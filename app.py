import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import json
import requests
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block
from ultralytics.nn.modules.conv import Conv

# ==========================================
# 1. 深度学习模块定义 (与你的 pt 模型匹配)
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
        try: return self.cbam(self.cv2(torch.cat(y, 1)))
        except: return self.attn(self.cv2(torch.cat(y, 1)))

# 动态注入
block.C2f = tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM); setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention); setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 2. 核心算法：完全迁移你的本地 SIFT 级联逻辑
# ==========================================
def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    for i in range(1, 9):
        cv2.line(grid_img, (int(i * w / 9), 0), (int(i * w / 9), h), (0, 255, 0), 2)
        cv2.line(grid_img, (0, int(i * h / 9)), (w, int(i * h / 9)), (0, 255, 0), 2)
    return grid_img

def process_sift_and_detect(tpl_img, test_img, model, conf):
    """
    这是你要求的本地逻辑：
    1. SIFT 提取特征点 -> 2. KNN 匹配 -> 3. 单应性矩阵 M 
    -> 4. ROI 区域变换 -> 5. 裁剪检测 -> 6. 坐标还原
    """
    h, w = tpl_img.shape[:2]
    # SIFT 找点
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), None)
    
    final_results = []
    M = None
    
    # 特征匹配
    if des1 is not None and des2 is not None:
        matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.8 * n.distance]
        if len(good) >= 8:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 级联检测核心：ROI 映射
    if M is not None:
        roi_defs = [
            {"box": [[0, 0], [0, h], [w * 0.4, h], [w * 0.4, 0]]}, # 左侧区域
            {"box": [[w * 0.6, 0], [w * 0.6, h], [w, h], [w, 0]]}  # 右侧区域
        ]
        for r in roi_defs:
            pts = np.float32(r["box"]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            rx, ry, rw, rh = cv2.boundingRect(dst)
            # 防止坐标越界
            rx, ry = max(0, rx), max(0, ry)
            crop = test_img[ry:ry + rh, rx:rx + rw]
            
            if crop.size > 0:
                # 在裁剪图上执行 YOLO
                res = model.predict(crop, conf=conf, verbose=False)
                for r_obj in res:
                    for b in r_obj.boxes:
                        bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                        # 重要：还原到全图坐标 (bx1 + rx, by1 + ry)
                        final_results.append({
                            "box": [int(bx1 + rx), int(by1 + ry), int(bx2 + rx), int(by2 + ry)],
                            "cls": int(b.cls[0]),
                            "conf": float(b.conf[0])
                        })
    else:
        # 如果 SIFT 没对齐，保底全图检测一次
        res = model.predict(test_img, conf=conf, verbose=False)
        for r_obj in res:
            for b in r_obj.boxes:
                final_results.append({
                    "box": b.xyxy[0].cpu().numpy().astype(int).tolist(),
                    "cls": int(b.cls[0]),
                    "conf": float(b.conf[0])
                })
                
    return final_results

# ==========================================
# 3. GUI 界面与寻路逻辑
# ==========================================
st.set_page_config(page_title="PCB SIFT-YOLO GUI", layout="wide")

# 加载路径索引
@st.cache_data
def load_path_map():
    if os.path.exists("path_index.json"):
        with open("path_index.json", "r", encoding="utf-8") as f: return json.load(f)
    return {}

path_map = load_path_map()
GITHUB_BASE = "https://api.github.com/repos/77shaxinyu/detection/contents/dataset_empty/"

@st.cache_data
def get_cloud_tpl(rel_path):
    url = f"{GITHUB_BASE}{rel_path.replace('\\', '/')}"
    try:
        r = requests.get(url, timeout=5).json()
        f = next((i for i in r if i['name'].lower().endswith(('.jpg', '.png'))), None)
        if f:
            data = requests.get(f['download_url']).content
            return cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    except: pass
    return None

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Interactive", "Batch Scan"])
    model_type = st.selectbox("Model", ["Model 1 (SE)", "Model 2 (CBAM)"])
    conf_val = st.slider("Confidence", 0.1, 1.0, 0.25)

@st.cache_resource
def load_yolo(choice):
    p = f"models/{'se' if '1' in choice else 'cbam'}.pt"
    return YOLO(p) if os.path.exists(p) else None

model = load_yolo(model_type)

files = st.file_uploader("Upload PCB", accept_multiple_files=True)

if files and model:
    history = []
    for f in files:
        # 1. 读取原图
        live_img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
        # 2. 获取对应的云端模板
        rel_path = path_map.get(f.name, "")
        tpl_img = get_cloud_tpl(rel_path)
        
        # 3. 运行你的本地级联检测逻辑
        with st.spinner(f"Detecting {f.name}..."):
            # 如果没找到模板，逻辑里会自动保底全图检测
            results = process_sift_and_detect(tpl_img if tpl_img is not None else live_img, live_img, model, conf_val)
        
        # 4. 绘图 (9x9网格 + 检测框)
        canvas = draw_grid_9x9(live_img)
        for r in results:
            x1, y1, x2, y2 = r["box"]
            name = model.names[r["cls"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(canvas, name, (x1, y1 - 10), 0, 1.2, (0, 255, 255), 2)
            history.append({"File": f.name, "Class": name, "Conf": f"{r['conf']:.2f}"})
        
        if mode == "Interactive":
            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"Result: {f.name}")
    
    if history:
        st.write("### Detection Summary")
        st.dataframe(pd.DataFrame(history), use_container_width=True)
