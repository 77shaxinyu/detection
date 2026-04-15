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
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block
from ultralytics.nn.modules.conv import Conv

# ==========================================
# 1. System Initialization / 系统初始化
# ==========================================
TEMP_DIR = "temp_results"
INDEX_FILE = "path_index.json"
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
        x = x * self.spatial(torch.cat([avg_out, max_out], dim=1))
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
        try: return self.cbam(self.cv2(torch.cat(y, 1)))
        except: return self.attn(self.cv2(torch.cat(y, 1)))

block.C2f = tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM); setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention); setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 3. Helper & Cascade Functions / 核心算法
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
def get_cloud_template_img(file_name, path_map):
    rel_path = path_map.get(file_name)
    if not rel_path: return None
    api_url = f"{GITHUB_API_BASE}{rel_path.replace('\\', '/')}"
    try:
        res = requests.get(api_url, timeout=5).json()
        f_info = next((f for f in res if f['name'].lower().endswith(('.jpg', '.png'))), None)
        if f_info:
            data = requests.get(f_info['download_url']).content
            return cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    except: pass
    return None

def cascade_sift_detect(tpl_img, test_img, model, conf):
    """严格本地 SIFT 级联逻辑"""
    h_tpl, w_tpl = tpl_img.shape[:2]
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), None)
    
    M = None
    if des1 is not None and des2 is not None:
        matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.8 * n.distance]
        if len(good) >= 8:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    final_boxes = []
    if M is not None:
        roi_defs = [{"box": [[0, 0], [0, h_tpl], [w_tpl*0.4, h_tpl], [w_tpl*0.4, 0]]},
                    {"box": [[w_tpl*0.6, 0], [w_tpl*0.6, h_tpl], [w_tpl, h_tpl], [w_tpl, 0]]}]
        for r in roi_defs:
            dst = cv2.perspectiveTransform(np.float32(r["box"]).reshape(-1, 1, 2), M)
            rx, ry, rw, rh = cv2.boundingRect(dst)
            rx, ry = max(0, rx), max(0, ry)
            crop = test_img[ry:ry+rh, rx:rx+rw]
            if crop.size > 0:
                res = model.predict(crop, conf=conf, verbose=False)
                for r_obj in res:
                    for b in r_obj.boxes:
                        bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                        final_boxes.append({"xyxy": [bx1+rx, by1+ry, bx2+rx, by2+ry], "cls": int(b.cls[0]), "conf": float(b.conf[0])})
    
    if not final_boxes: # 保底
        res = model.predict(test_img, conf=conf, verbose=False)
        for r_obj in res:
            for b in r_obj.boxes:
                final_boxes.append({"xyxy": b.xyxy[0].cpu().numpy(), "cls": int(b.cls[0]), "conf": float(b.conf[0])})
    return final_boxes

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

if uploaded_files:
    if model is None: st.error("Model not found.")
    else:
        if proc_mode == "Fast Batch Scan (快速批量扫描)":
            st.info("Fast scanning images...")
            progress_bar = st.progress(0)
            current_scan_data = []

            for idx, file in enumerate(uploaded_files):
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
                h, w = img.shape[:2]
                
                # 级联逻辑
                tpl = get_cloud_template_img(file.name, path_map)
                boxes = cascade_sift_detect(tpl if tpl is not None else img, img, model, conf_thresh)

                for b in boxes:
                    cls = model.names[b["cls"]]
                    current_scan_data.append({
                        "File": file.name,
                        "Type / 类型": get_component_type(cls),
                        "Class / 类别": cls,
                        "Confidence / 置信度": f"{b['conf']:.2f}",
                        "Grid / 网格": get_grid_pos((b["xyxy"][0] + b["xyxy"][2]) / 2, (b["xyxy"][1] + b["xyxy"][3]) / 2, h / 9, w / 9),
                        "Coordinates / 坐标": f"({int(b['xyxy'][0])},{int(b['xyxy'][1])},{int(b['xyxy'][2])},{int(b['xyxy'][3])})"
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

                # 级联识别
                tpl = get_cloud_template_img(file.name, path_map)
                boxes = cascade_sift_detect(tpl if tpl is not None else img, img, model, conf_thresh)

                st.session_state.history = [d for d in st.session_state.history if d['File'] != file.name]

                for b in boxes:
                    x1, y1, x2, y2 = map(int, b["xyxy"])
                    cls = model.names[b["cls"]]
                    pos = get_grid_pos((x1 + x2) / 2, (y1 + y2) / 2, ch, cw)
                    entry = {
                        "File": file.name,
                        "Type / 类型": get_component_type(cls),
                        "Class / 类别": cls,
                        "Confidence / 置信度": f"{b['conf']:.2f}",
                        "Grid / 网格": pos,
                        "Coordinates / 坐标": f"({x1},{y1},{x2},{y2})"
                    }
                    st.session_state.history.append(entry)
                    cv2.rectangle(grid_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(grid_img, f"{cls} {pos}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

                cv2.imwrite(target_path, grid_img)

            if st.session_state.history:
                st.divider()
                df_all = pd.DataFrame(st.session_state.history)
                last_file_name = uploaded_files[-1].name
                df_curr = df_all[df_all["File"] == last_file_name]

                def get_physical_count(df_subset):
                    num = len(df_subset)
                    if num == 0: return 0
                    return 1 if num == 1 else int(num / 2)

                res_c = get_physical_count(df_curr[df_curr["Type / 类型"].str.contains("Resistor")])
                cap_c = get_physical_count(df_curr[df_curr["Type / 类型"].str.contains("Capacitor")])

                c1, c2, c3 = st.columns(3)
                c1.info(f"Resistors / 电阻: {res_c}")
                c2.info(f"Capacitors / 电容: {cap_c}")
                c3.success(f"Total / 总计: {res_c + cap_c}")

                display_path = os.path.join(TEMP_DIR, f"{os.path.splitext(last_file_name)[0]}_{model_choice}_annotated.jpg")
                if os.path.exists(display_path):
                    st.image(cv2.cvtColor(cv2.imread(display_path), cv2.COLOR_BGR2RGB))
                st.dataframe(df_curr, use_container_width=True)
