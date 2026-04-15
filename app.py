import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import zipfile
import shutil
import io
import json
import requests
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block
from ultralytics.nn.modules.conv import Conv

# ==========================================
# 1. System Initialization / 系统初始化
# ==========================================
TEMP_DIR = "temp_results"
GITHUB_API_BASE = "https://api.github.com/repos/77shaxinyu/detection/contents/dataset_empty/"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 2. Custom Modules Injection / 模块注入
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
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        x = x * self.spatial(spatial_input)
        return x

class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
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
        try: return self.cbam(self.cv2(torch.cat(y, 1)))
        except: return self.attn(self.cv2(torch.cat(y, 1)))

block.C2f = tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM); setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention); setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 3. Core Algorithms / 核心算法 (DBSCAN + Helper)
# ==========================================
def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    cell_h, cell_w = h / 9, w / 9
    for i in range(1, 9):
        cv2.line(grid_img, (int(i * cell_w), 0), (int(i * cell_w), h), (0, 255, 0), 1)
        cv2.line(grid_img, (0, int(i * cell_h)), (w, int(i * cell_h)), (0, 255, 0), 1)
    return grid_img, cell_h, cell_w

def get_grid_pos(x_center, y_center, cell_h, cell_w):
    col = chr(ord('A') + int(x_center / cell_w))
    row = int(y_center / cell_h) + 1
    return f"{col}{row}"

def get_component_type(class_name):
    name_lower = class_name.lower()
    return "Resistor / 电阻" if "resistor" in name_lower else "Capacitor / 电容"

@st.cache_data
def get_cloud_templates_list(file_name, path_map):
    """获取 GitHub 对应文件夹下所有模板（模拟本地遍历）"""
    rel_path = path_map.get(file_name)
    if not rel_path: return []
    api_url = f"{GITHUB_API_BASE}{rel_path.replace('\\', '/')}"
    templates = []
    try:
        res = requests.get(api_url, timeout=5).json()
        for item in res:
            if item['name'].lower().endswith(('.jpg', '.png', '.jpeg')):
                data = requests.get(item['download_url']).content
                img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
                if img is not None: templates.append(img)
    except: pass
    return templates

def sift_dbscan_count(scene_img, templates):
    """严格按照你本地的 SIFT-DBSCAN 聚类计数逻辑"""
    scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_s, des_s = sift.detectAndCompute(scene_gray, None)
    if des_s is None: return 0

    best_count, max_good_matches = 0, 0
    for t_img in templates:
        t_gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
        kp_t, des_t = sift.detectAndCompute(t_gray, None)
        if des_t is None: continue

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
                if -1 in unique_labels: unique_labels.remove(-1)
                best_count = len(unique_labels)
    return best_count

# ==========================================
# 4. Streamlit UI / 界面显示
# ==========================================
st.set_page_config(page_title="PCB Detection System", layout="wide")

@st.cache_data
def load_path_map():
    if os.path.exists("path_index.json"):
        with open("path_index.json", "r", encoding="utf-8") as f: return json.load(f)
    return {}

path_map = load_path_map()

with st.sidebar:
    st.header("Configuration / 配置")
    proc_mode = st.radio("Processing Mode / 处理模式", ["Interactive (交互预览)", "Fast Batch Scan (快速批量扫描)"])
    model_choice = st.selectbox("Select Model / 选择检测模型", ["Model 1", "Model 2"])
    algo_choice = st.selectbox("Select Algorithm / 选择定位算法", ["Algorithm 1", "Algorithm 2"])
    conf_thresh = st.slider("Confidence Threshold / 置信度阈值", 0.1, 1.0, 0.25)

    if st.button("Clear Records / 清空记录"):
        st.session_state.history = []
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        st.rerun()

@st.cache_resource
def load_pcb_model(choice):
    path = "models/se.pt" if choice == "Model 1" else "models/cbam.pt"
    return YOLO(path) if os.path.exists(path) else None

model = load_pcb_model(model_choice)
if "history" not in st.session_state: st.session_state.history = []

uploaded_files = st.file_uploader("Upload PCB Images / 上传图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and model:
    if proc_mode == "Fast Batch Scan (快速批量扫描)":
        st.info("Fast scanning with DBSCAN clustering...")
        progress_bar = st.progress(0)
        current_scan_data = []

        for idx, file in enumerate(uploaded_files):
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            h, w = img.shape[:2]
            
            # SIFT 物理计数
            tpls = get_cloud_templates_list(file.name, path_map)
            sift_count = sift_dbscan_count(img, tpls)
            
            # YOLO 坐标获取 (表格记录用)
            results = model.predict(img, conf=conf_thresh, iou=0.45, agnostic_nms=True, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = model.names[int(box.cls[0])]
                    current_scan_data.append({
                        "File": file.name,
                        "Type / 类型": get_component_type(cls),
                        "Class / 类别": cls,
                        "Confidence / 置信度": f"{float(box.conf[0]):.2f}",
                        "Grid / 网格": get_grid_pos((x1 + x2) / 2, (y1 + y2) / 2, h / 9, w / 9),
                        "Coordinates / 坐标": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                        "DBSCAN_Count": sift_count # 隐藏记录物理计数
                    })
            progress_bar.progress((idx + 1) / len(uploaded_files))

        st.session_state.history.extend(current_scan_data)
        if st.session_state.history:
            st.divider()
            df_all = pd.DataFrame(st.session_state.history)
            st.subheader("Consolidated Defect Report / 缺陷汇总表格")
            st.dataframe(df_all, use_container_width=True)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                csv_data = df_all.to_csv(index=False, encoding='utf-8-sig')
                zip_file.writestr("total_defects_report.csv", csv_data)
            st.download_button(label="Download Consolidated ZIP", data=zip_buffer.getvalue(), file_name=f"Batch_Scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")

    else:
        # Interactive Mode Logic
        for file in uploaded_files:
            file_base = os.path.splitext(file.name)[0]
            target_path = os.path.join(TEMP_DIR, f"{file_base}_{model_choice}_annotated.jpg")

            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            grid_img, ch, cw = draw_grid_9x9(img)

            # 1. 物理计数逻辑 (严格按 SIFT 聚类)
            tpls = get_cloud_templates_list(file.name, path_map)
            sift_count = sift_dbscan_count(img, tpls)

            # 2. YOLO 检测用于标注
            results = model.predict(img, conf=conf_thresh, iou=0.45, agnostic_nms=True)
            st.session_state.history = [d for d in st.session_state.history if d['File'] != file.name]

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = model.names[int(box.cls[0])]
                    pos = get_grid_pos((x1 + x2) / 2, (y1 + y2) / 2, ch, cw)
                    entry = {
                        "File": file.name, "Type / 类型": get_component_type(cls),
                        "Class / 类别": cls, "Confidence / 置信度": f"{float(box.conf[0]):.2f}",
                        "Grid / 网格": pos, "Coordinates / 坐标": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                        "DBSCAN_Count": sift_count
                    }
                    st.session_state.history.append(entry)
                    cv2.rectangle(grid_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(grid_img, f"{cls} {pos}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

            cv2.imwrite(target_path, grid_img)

        if st.session_state.history:
            st.divider()
            df_all = pd.DataFrame(st.session_state.history)
            last_file_name = uploaded_files[-1].name
            df_curr = df_all[df_all["File"] == last_file_name]

            # --- 修改统计展示逻辑 ---
            # 统计当前文件不同类型的物理数量 (根据 DBSCAN 计数分配)
            res_df = df_curr[df_curr["Type / 类型"].str.contains("Resistor")]
            cap_df = df_curr[df_curr["Type / 类型"].str.contains("Capacitor")]
            
            # 由于 DBSCAN 计算的是该型号总元件数，此处我们取记录的 DBSCAN_Count 值
            physical_total = df_curr["DBSCAN_Count"].iloc[0] if not df_curr.empty else 0
            
            c1, c2, c3 = st.columns(3)
            # 此处我们将 SIFT 聚类得到的结果作为物理计数展示
            c1.info(f"Resistors / 电阻: 待定") # SIFT 目前是针对单型号全计数，此处为展示一致性
            c2.info(f"Capacitors / 电容: 待定")
            c3.success(f"Physical Total (SIFT): {physical_total}")

            display_path = os.path.join(TEMP_DIR, f"{os.path.splitext(last_file_name)[0]}_{model_choice}_annotated.jpg")
            if os.path.exists(display_path):
                st.image(cv2.cvtColor(cv2.imread(display_path), cv2.COLOR_BGR2RGB))
            st.dataframe(df_curr, use_container_width=True)
