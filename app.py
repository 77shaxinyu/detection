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
# 3. Cascade Logic / 级联核心逻辑 (自动匹配与对齐)
# ==========================================
@st.cache_data
def get_cloud_template(file_name, path_map):
    """根据文件名自动从云端获取对应路径下的模板图"""
    rel_path = path_map.get(file_name)
    if not rel_path: return None, "Index not found"
    
    api_url = f"{GITHUB_API_BASE}{rel_path.replace('\\', '/')}"
    try:
        res = requests.get(api_url, timeout=5).json()
        tpl_info = next((f for f in res if f['name'].lower().endswith(('.jpg', '.png'))), None)
        if tpl_info:
            img_data = requests.get(tpl_info['download_url']).content
            return cv2.imdecode(np.frombuffer(img_data, np.uint8), 1), tpl_info['name']
    except: pass
    return None, "Cloud match failed"

def cascade_inspect(template_img, live_img, algorithm, model, conf):
    """级联巡检：特征对齐 -> ROI映射 -> YOLO检测"""
    h, w = template_img.shape[:2]
    # 1. 对齐
    detector = cv2.SIFT_create(2000) if "Algorithm 1" in algorithm else cv2.ORB_create(3000)
    kp1, des1 = detector.detectAndCompute(cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = detector.detectAndCompute(cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY), None)
    
    M = None
    if des1 is not None and des2 is not None:
        matcher = cv2.BFMatcher() if "Algorithm 1" in algorithm else cv2.BFMatcher(cv2.NORM_HAMMING, True)
        matches = matcher.knnMatch(des1, des2, k=2) if "Algorithm 1" in algorithm else matcher.match(des1, des2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance] if "Algorithm 1" in algorithm else sorted(matches, key=lambda x:x.distance)[:100]
        if len(good) >= 8:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 2. ROI 推理 (若对齐失败则全图推理)
    final_boxes = []
    if M is not None:
        rois = [{"box": [[0,0],[0,h],[w*0.4,h],[w*0.4,0]]}, {"box": [[w*0.6,0],[w*0.6,h],[w,h],[w,0]]}]
        for r in rois:
            dst = cv2.perspectiveTransform(np.float32(r["box"]).reshape(-1,1,2), M)
            rx, ry, rw, rh = cv2.boundingRect(dst)
            rx, ry = max(0, rx), max(0, ry)
            crop = live_img[ry:ry+rh, rx:rx+rw]
            if crop.size > 0:
                res = model.predict(crop, conf=conf, verbose=False)
                for r_obj in res:
                    for b in r_obj.boxes:
                        bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                        final_boxes.append({"xyxy":[bx1+rx, by1+ry, bx2+rx, by2+ry], "cls":int(b.cls[0]), "conf":float(b.conf[0])})
    else:
        res = model.predict(live_img, conf=conf, verbose=False)
        for r_obj in res:
            for b in r_obj.boxes: final_boxes.append({"xyxy":b.xyxy[0].cpu().numpy(), "cls":int(b.cls[0]), "conf":float(b.conf[0])})
    return final_boxes

# ==========================================
# 4. Streamlit UI / 界面显示
# ==========================================
st.set_page_config(page_title="PCB Intelligent Inspection System", layout="wide")

@st.cache_data
def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f: return json.load(f)
    return {}

path_map = load_index()

with st.sidebar:
    st.header("Configuration / 配置")
    proc_mode = st.radio("Processing Mode", ["Interactive (交互预览)", "Fast Batch Scan (快速批量扫描)"])
    model_choice = st.selectbox("Select Model", ["Model 1", "Model 2"])
    algo_choice = st.selectbox("Select Algorithm", ["Algorithm 1 (SIFT)", "Algorithm 2 (ORB)"])
    conf_thresh = st.slider("Confidence", 0.1, 1.0, 0.25)
    if st.button("Clear Records"):
        st.session_state.history = []
        st.rerun()

@st.cache_resource
def load_pcb_model(choice):
    path = f"models/{'se' if '1' in choice else 'cbam'}.pt"
    return YOLO(path) if os.path.exists(path) else None

model = load_pcb_model(model_choice)
if "history" not in st.session_state: st.session_state.history = []

uploaded_files = st.file_uploader("Upload PCB Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if not path_map: st.error("path_index.json not found!")
    elif model is None: st.error("Model not found!")
    else:
        if proc_mode == "Fast Batch Scan (快速批量扫描)":
            st.info("Batch scanning with cascade registration...")
            bar = st.progress(0)
            current_data = []
            for idx, file in enumerate(uploaded_files):
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
                tpl_img, _ = get_cloud_template(file.name, path_map)
                boxes = cascade_inspect(tpl_img if tpl_img is not None else img, img, algo_choice, model, conf_thresh)
                for b in boxes:
                    cls = model.names[b["cls"]]
                    current_data.append({
                        "File": file.name, "Type / 类型": "Resistor" if "resistor" in cls.lower() else "Capacitor",
                        "Class": cls, "Confidence": f"{b['conf']:.2f}",
                        "Grid": "Auto", "Coordinates": f"({int(b['xyxy'][0])},{int(b['xyxy'][1])})"
                    })
                bar.progress((idx + 1) / len(uploaded_files))
            st.session_state.history.extend(current_data)
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        else:
            # Interactive Mode
            for file in uploaded_files:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
                with st.spinner(f"Aligning {file.name}..."):
                    tpl_img, tpl_name = get_cloud_template(file.name, path_map)
                    boxes = cascade_inspect(tpl_img if tpl_img is not None else img, img, algo_choice, model, conf_thresh)
                
                canvas, ch, cw = draw_grid_9x9(img)
                st.session_state.history = [d for d in st.session_state.history if d['File'] != file.name]
                for b in boxes:
                    x1, y1, x2, y2 = map(int, b["xyxy"])
                    cls = model.names[b["cls"]]
                    pos = "Auto"
                    st.session_state.history.append({"File": file.name, "Class": cls, "Confidence": f"{b['conf']:.2f}"})
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(canvas, f"{cls}", (x1, y1 - 5), 0, 1.2, (255, 255, 0), 2)
                
                st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"Aligned Detection: {file.name}")
                st.dataframe(pd.DataFrame([d for d in st.session_state.history if d['File'] == file.name]))
