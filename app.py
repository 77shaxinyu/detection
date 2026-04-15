import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import io
import json
import requests
import torch
import torch.nn as nn
import zipfile
from datetime import datetime
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block
from ultralytics.nn.modules.conv import Conv

# ==========================================
# 1. 系统初始化
# ==========================================
TEMP_DIR = "temp_results"
GITHUB_API_BASE = "https://api.github.com/repos/77shaxinyu/detection/contents/dataset_empty/"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 2. YOLO 模型组件注入
# ==========================================
class CBAM(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_sum = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c1, c1 // ratio, 1, bias=False),
            nn.ReLU(), nn.Conv2d(c1 // ratio, c1, 1, bias=False))
        self.channel_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Conv2d(c1, c1 // ratio, 1, bias=False),
            nn.ReLU(), nn.Conv2d(c1 // ratio, c1, 1, bias=False))
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False), nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        channel_att = self.sigmoid(self.channel_sum(x) + self.channel_max(x))
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([avg_out, max_out], dim=1))

class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class C2f_Custom(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.cbam = CBAM(c2)
        self.attn = SEAttention(c2)
        self.m = nn.ModuleList(block.Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        try:
            return self.cbam(self.cv2(torch.cat(y, 1)))
        except Exception:
            return self.attn(self.cv2(torch.cat(y, 1)))

block.C2f = tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM)
setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention)
setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 3. 核心算法 (SIFT + DBSCAN)
# ==========================================
def sift_count_logic_LOCAL(scene_gray, templates_list):
    sift = cv2.SIFT_create()
    kp_s, des_s = sift.detectAndCompute(scene_gray, None)
    if des_s is None:
        return 0
    best_count = 0
    max_good_matches = 0
    for img_t_bgr in templates_list:
        if img_t_bgr is None:
            continue
        img_t = cv2.cvtColor(img_t_bgr, cv2.COLOR_BGR2GRAY)
        kp_t, des_t = sift.detectAndCompute(img_t, None)
        if des_t is None:
            continue
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des_t, des_s, k=2)
        current_good_pts = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                current_good_pts.append(kp_s[m.trainIdx].pt)
        if len(current_good_pts) > max_good_matches:
            max_good_matches = len(current_good_pts)
            if len(current_good_pts) >= 5:
                X = np.array(current_good_pts)
                clustering = DBSCAN(eps=45, min_samples=4).fit(X)
                unique_labels = set(clustering.labels_)
                if -1 in unique_labels:
                    unique_labels.remove(-1)
                best_count = len(unique_labels)
    return int(best_count)

# ==========================================
# 4. 辅助函数
# ==========================================
def get_grid_pos(x_center, y_center, cell_h, cell_w):
    col = chr(ord('A') + int(x_center / cell_w))
    row = int(y_center / cell_h) + 1
    return f"{col}{row}"

def get_component_type(class_name):
    name_lower = class_name.lower()
    if "resistor" in name_lower:
        return "Resistor / 电阻"
    return "Capacitor / 电容"

@st.cache_data
def get_cloud_templates_list(file_name, path_map):
    rel_path = path_map.get(file_name)
    if not rel_path:
        return []
    api_url = f"{GITHUB_API_BASE}{rel_path.replace('\\', '/')}"
    templates = []
    try:
        res = requests.get(api_url, timeout=5).json()
        for item in res:
            if item["name"].lower().endswith((".jpg", ".png", ".jpeg")):
                data = requests.get(item["download_url"]).content
                img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
                if img is not None:
                    templates.append(img)
    except Exception:
        pass
    return templates

# ==========================================
# 5. UI 界面逻辑
# ==========================================
st.set_page_config(page_title="PCB Inspection System", layout="wide")

@st.cache_data
def load_path_map():
    if os.path.exists("path_index.json"):
        with open("path_index.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

path_map = load_path_map()

with st.sidebar:
    st.header("Configuration")
    proc_mode = st.radio("Mode", ["Interactive", "Fast Batch Scan"])
    model_choice = st.selectbox("Model", ["Model 1", "Model 2"])
    conf_thresh = st.slider("Confidence", 0.1, 1.0, 0.25)
    if st.button("Clear Records"):
        st.session_state.history = []
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        st.rerun()

@st.cache_resource
def load_pcb_model(choice):
    path = "models/se.pt" if "1" in choice else "models/cbam.pt"
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception:
            return None
    return None

model = load_pcb_model(model_choice)
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and model:
    if proc_mode == "Fast Batch Scan":
        st.success("Batch Scanning Started")
        progress_bar = st.progress(0)
        all_results = []
        for idx, file in enumerate(uploaded_files):
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            tpls = get_cloud_templates_list(file.name, path_map)
            s_count = sift_count_logic_LOCAL(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), tpls)
            yolo_res = model.predict(img, conf=conf_thresh, verbose=False)
            for r in yolo_res:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    cls = model.names[int(b.cls[0])]
                    all_results.append({
                        "File": file.name, "Type / 类型": get_component_type(cls),
                        "Class / 类别": cls, "Confidence / 置信度": f"{float(b.conf[0]):.2f}",
                        "Grid / 网格": get_grid_pos((x1+x2)/2, (y1+y2)/2, img.shape[0]/9, img.shape[1]/9),
                        "Coordinates / 坐标": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                        "S_Count": s_count
                    })
            progress_bar.progress((idx + 1) / len(uploaded_files))
        st.session_state.history = all_results
    else:
        for file in uploaded_files:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            tpls = get_cloud_templates_list(file.name, path_map)
            s_count = sift_count_logic_LOCAL(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), tpls)
            res = model.predict(img, conf=conf_thresh, verbose=False)
            canvas = img.copy()
            st.session_state.history = [d for d in st.session_state.history if d["File"] != file.name]
            for r in res:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
                    cls = model.names[int(b.cls[0])]
                    st.session_state.history.append({
                        "File": file.name, "Type / 类型": get_component_type(cls),
                        "Class / 类别": cls, "Confidence / 置信度": f"{float(b.conf[0]):.2f}",
                        "Grid / 网格": get_grid_pos((x1+x2)/2, (y1+y2)/2, img.shape[0]/9, img.shape[1]/9),
                        "Coordinates / 坐标": f"({x1},{y1},{x2},{y2})",
                        "S_Count": s_count
                    })
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        last_file = uploaded_files[-1].name
        df_curr = df[df["File"] == last_file]
        
        # 统计数据处理
        phys_total = int(df_curr["S_Count"].iloc[0]) if not df_curr.empty else 0
        res_boxes = len(df_curr[df_curr["Type / 类型"].str.contains("Resistor")])
        cap_boxes = len(df_curr[df_curr["Type / 类型"].str.contains("Capacitor")])
        total_boxes = res_boxes + cap_boxes
        res_phys = int(round(phys_total * (res_boxes / total_boxes))) if total_boxes > 0 else 0
        cap_phys = phys_total - res_phys

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.info(f"Resistors / 电阻: {res_phys}")
        c2.info(f"Capacitors / 电容: {cap_phys}")
        c3.success(f"Total: {phys_total}")
        
        # 显示表格并剔除隐藏列
        st.dataframe(df.drop(columns=["S_Count"], errors="ignore"), use_container_width=True)

        # 导出 ZIP 功能
        csv_data = df.drop(columns=["S_Count"], errors="ignore").to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("report.csv", csv_data)
        st.download_button("Download Result ZIP", data=zip_buffer.getvalue(), file_name="results.zip")
