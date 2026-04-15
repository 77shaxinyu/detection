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
import shutil
from datetime import datetime
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block
from ultralytics.nn.modules.conv import Conv

# ==========================================
# 1. 模型架构定义 (SE/CBAM)
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
# 2. 核心算法逻辑 (定位与检测)
# ==========================================
def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    for i in range(1, 9):
        cv2.line(grid_img, (int(i * w / 9), 0), (int(i * w / 9), h), (0, 255, 0), 2)
        cv2.line(grid_img, (0, int(i * h / 9)), (w, int(i * h / 9)), (0, 255, 0), 2)
    return grid_img, h / 9, w / 9

def get_alignment_matrix(tpl_img, test_img, mode="ORB"):
    if mode == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        ratio = 0.85
    else:
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher()
        ratio = 0.75

    kp1, des1 = detector.detectAndCompute(cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = detector.detectAndCompute(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None:
        return None

    matches = matcher.knnMatch(des1, des2, k=2)
    
    # 显式循环 Ratio Test 以防止作用域错误
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio * n.distance:
                good.append(m)
    
    if len(good) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M
    return None

def get_roi_detect(img, M, model, conf):
    h, w = img.shape[:2]
    roi_defs = [{"box": [[0, 0], [0, h], [w * 0.4, h], [w * 0.4, 0]]},
                {"box": [[w * 0.6, 0], [w * 0.6, h], [w, h], [w, 0]]}]
    final_boxes = []
    for r in roi_defs:
        pts = np.float32(r["box"]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        rx, ry, rw, rh = cv2.boundingRect(dst)
        rx, ry = max(0, rx), max(0, ry)
        crop = img[ry:ry + rh, rx:rx + rw]
        if crop.size > 0:
            res = model.predict(crop, conf=conf, verbose=False)
            for r_obj in res:
                for b in r_obj.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                    final_boxes.append({
                        "xyxy": [bx1 + rx, by1 + ry, bx2 + rx, by2 + ry], 
                        "cls": int(b.cls[0]), 
                        "conf": float(b.conf[0])
                    })
    return final_boxes

# ==========================================
# 3. 辅助函数
# ==========================================
def get_grid_pos(x_center, y_center, cell_h, cell_w):
    col = chr(ord("A") + int(x_center / cell_w))
    row = int(y_center / cell_h) + 1
    return f"{col}{row}"

def get_component_type(class_name):
    if "resistor" in class_name.lower():
        return "Resistor / 电阻"
    return "Capacitor / 电容"

@st.cache_data
def get_cloud_templates(file_name, path_map):
    rel_path = path_map.get(file_name)
    if not rel_path:
        return []
    api_url = f"https://api.github.com/repos/77shaxinyu/detection/contents/dataset_empty/{rel_path.replace('\\', '/')}"
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
# 4. UI 界面逻辑
# ==========================================
st.set_page_config(page_title="PCB Inspection System", layout="wide")

@st.cache_data
def load_path_map():
    if os.path.exists("path_index.json"):
        with open("path_index.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

path_map = load_path_map()
TEMP_DIR = "temp_results"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

with st.sidebar:
    st.header("Configuration")
    proc_mode = st.radio("Processing Mode", ["Interactive", "Fast Batch Scan"])
    model_choice = st.selectbox("DL Model", ["Model 1 (SE)", "Model 2 (CBAM)"])
    algo_choice = st.selectbox("Alignment Algorithm", ["ORB", "SIFT"])
    conf_thresh = st.slider("Confidence", 0.1, 1.0, 0.25)
    if st.button("Clear Records"):
        st.session_state.history = []
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        st.rerun()

@st.cache_resource
def load_pcb_model(choice):
    path = "models/se.pt" if "SE" in choice else "models/cbam.pt"
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
    for f in uploaded_files:
        img_bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
        tpls = get_cloud_templates(f.name, path_map)
        
        with st.spinner(f"Analyzing {f.name}..."):
            final_boxes = []
            if tpls:
                M = get_alignment_matrix(tpls[0], img_bgr, mode=algo_choice)
                if M is not None:
                    final_boxes = get_roi_detect(img_bgr, M, model, conf_thresh)
            
            if not final_boxes:
                res = model.predict(img_bgr, conf=conf_thresh, verbose=False)
                for r in res:
                    for b in r.boxes:
                        final_boxes.append({
                            "xyxy": b.xyxy[0].cpu().numpy(), 
                            "cls": int(b.cls[0]), 
                            "conf": float(b.conf[0])
                        })

        canvas, ch, cw = draw_grid_9x9(img_bgr)
        st.session_state.history = [d for d in st.session_state.history if d["File"] != f.name]
        
        for box in final_boxes:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            cls_name = model.names[box["cls"]]
            pos = get_grid_pos((x1+x2)/2, (y1+y2)/2, ch, cw)
            
            st.session_state.history.append({
                "File": f.name,
                "Type / 类型": get_component_type(cls_name),
                "Class / 类别": cls_name,
                "Confidence": f"{box['conf']:.2f}",
                "Grid / 网格": pos,
                "Coordinates": f"({x1},{y1},{x2},{y2})"
            })
            
            if proc_mode == "Interactive":
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(canvas, f"{cls_name} {pos}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    if st.session_state.history:
        df_all = pd.DataFrame(st.session_state.history)
        
        if proc_mode == "Interactive":
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            df_curr = df_all[df_all["File"] == uploaded_files[-1].name]
            r_c = len(df_curr[df_curr["Type / 类型"].str.contains("Resistor")])
            c_c = len(df_curr[df_curr["Type / 类型"].str.contains("Capacitor")])
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.info(f"Resistors / 电阻: {r_c}")
            col2.info(f"Capacitors / 电容: {c_c}")
            col3.success(f"Total: {r_c + c_c}")

        st.subheader("Inspection Report")
        st.dataframe(df_all, use_container_width=True)

        csv_data = df_all.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("report.csv", csv_data)
        st.download_button("Download Report ZIP", data=zip_buffer.getvalue(), file_name="results.zip", use_container_width=True)
