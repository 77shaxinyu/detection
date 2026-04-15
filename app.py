import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
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
# 1. 模型注入与系统初始化
# ==========================================
GITHUB_API_BASE = "https://api.github.com/repos/77shaxinyu/detection/contents/dataset_empty/"

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
        try: return self.cbam(self.cv2(torch.cat(y, 1)))
        except: return self.attn(self.cv2(torch.cat(y, 1)))

block.C2f = tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM); setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention); setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 2. 核心算法：你的本地 SIFT 级联逻辑
# ==========================================
def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    for i in range(1, 9):
        cv2.line(grid_img, (int(i * w / 9), 0), (int(i * w / 9), h), (0, 255, 0), 2)
        cv2.line(grid_img, (0, int(i * h / 9)), (w, int(i * h / 9)), (0, 255, 0), 2)
    return grid_img, h/9, w/9

def get_grid_pos(x, y, ch, cw):
    return f"{chr(ord('A') + int(x / cw))}{int(y / ch) + 1}"

@st.cache_data
def get_cloud_tpl(rel_path):
    api_url = f"{GITHUB_API_BASE}{rel_path.replace('\\', '/')}"
    try:
        res = requests.get(api_url, timeout=5).json()
        info = next((f for f in res if f['name'].lower().endswith(('.jpg', '.png'))), None)
        if info:
            data = requests.get(info['download_url']).content
            return cv2.imdecode(np.frombuffer(data, np.uint8), 1), info['name']
    except: pass
    return None, None

def cascade_logic(tpl_img, test_img, model, conf):
    """【你的本地 SIFT 逻辑】对齐 -> ROI 裁剪 -> YOLO -> 坐标还原"""
    h, w = tpl_img.shape[:2]
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

    boxes = []
    if M is not None:
        # 按照你的 roi_defs 逻辑
        roi_defs = [{"box": [[0,0],[0,h],[w*0.4,h],[w*0.4,0]]}, {"box": [[w*0.6,0],[w*0.6,h],[w,h],[w,0]]}]
        for r in roi_defs:
            dst = cv2.perspectiveTransform(np.float32(r["box"]).reshape(-1,1,2), M)
            rx, ry, rw, rh = cv2.boundingRect(dst)
            rx, ry = max(0, rx), max(0, ry)
            crop = test_img[ry:ry+rh, rx:rx+rw]
            if crop.size > 0:
                res = model.predict(crop, conf=conf, verbose=False)
                for r_obj in res:
                    for b in r_obj.boxes:
                        bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                        boxes.append({"xyxy":[bx1+rx, by1+ry, bx2+rx, by2+ry], "cls":int(b.cls[0]), "conf":float(b.conf[0])})
    else:
        res = model.predict(test_img, conf=conf, verbose=False)
        for r_obj in res:
            for b in r_obj.boxes: boxes.append({"xyxy":b.xyxy[0].cpu().numpy(), "cls":int(b.cls[0]), "conf":float(b.conf[0])})
    return boxes

# ==========================================
# 3. Streamlit UI (保持你原本的 GUI 结构)
# ==========================================
st.set_page_config(page_title="PCB SIFT-YOLO System", layout="wide")

@st.cache_data
def load_index():
    if os.path.exists("path_index.json"):
        with open("path_index.json", "r", encoding="utf-8") as f: return json.load(f)
    return {}

path_map = load_index()

with st.sidebar:
    st.header("Configuration / 配置")
    proc_mode = st.radio("Processing Mode", ["Interactive (交互预览)", "Fast Batch Scan (快速批量扫描)"])
    model_choice = st.selectbox("Select Model", ["Model 1 (SE)", "Model 2 (CBAM)"])
    conf_thresh = st.slider("Confidence", 0.1, 1.0, 0.25)
    if st.button("Clear Records"):
        st.session_state.history = []; st.rerun()

@st.cache_resource
def load_pcb_model(choice):
    path = f"models/{'se' if '1' in choice else 'cbam'}.pt"
    return YOLO(path) if os.path.exists(path) else None

model = load_pcb_model(model_choice)
if "history" not in st.session_state: st.session_state.history = []

files = st.file_uploader("Upload PCB Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if files and model:
    if proc_mode == "Fast Batch Scan (快速批量扫描)":
        bar = st.progress(0)
        curr_data = []
        for idx, f in enumerate(files):
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
            tpl, _ = get_cloud_tpl(path_map.get(f.name, ""))
            # 调用级联逻辑
            found = cascade_logic(tpl if tpl is not None else img, img, model, conf_thresh)
            for b in found:
                cls = model.names[b["cls"]]
                curr_data.append({"File": f.name, "Class": cls, "Confidence": f"{b['conf']:.2f}"})
            bar.progress((idx + 1) / len(files))
        st.session_state.history.extend(curr_data)
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        # Interactive Mode (交互预览)
        for f in files:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
            rel_path = path_map.get(f.name, "")
            with st.spinner(f"Processing {f.name}..."):
                tpl, tpl_n = get_cloud_tpl(rel_path)
                found = cascade_logic(tpl if tpl is not None else img, img, model, conf_thresh)
            
            canvas, ch, cw = draw_grid_9x9(img)
            st.session_state.history = [d for d in st.session_state.history if d['File'] != f.name]
            for b in found:
                x1, y1, x2, y2 = map(int, b["xyxy"])
                cls = model.names[b["cls"]]
                st.session_state.history.append({"File": f.name, "Class": cls, "Confidence": f"{b['conf']:.2f}"})
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(canvas, f"{cls}", (x1, y1 - 10), 0, 1.2, (0, 255, 255), 2)
            
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"Result: {f.name}")
            st.dataframe(pd.DataFrame([d for d in st.session_state.history if d['File'] == f.name]))
