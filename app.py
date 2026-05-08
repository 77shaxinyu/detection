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
import time
from datetime import datetime
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block
from ultralytics.nn.modules.conv import Conv
from docx import Document

# ==========================================
# 1. 模型架构定义 (保持底层支持，防止加载报错)
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
        except Exception: return self.attn(self.cv2(torch.cat(y, 1)))

block.C2f = tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM)
setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention)
setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 2. 核心性能分析逻辑
# ==========================================
def get_alignment_metrics(tpl_img, test_img, algo_name):
    metrics = {}
    t_start = time.perf_counter()
    
    if "Algorithm 2" in algo_name: # ORB
        detector = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        ratio = 0.85
    else: # SIFT
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher()
        ratio = 0.75

    kp1, des1 = detector.detectAndCompute(cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = detector.detectAndCompute(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), None)
    t_detect = time.perf_counter()
    
    metrics['Keypoints'] = len(kp2)
    metrics['Det_Time_ms'] = round((t_detect - t_start) * 1000, 2)

    if des1 is None or des2 is None: return metrics

    t_match_start = time.perf_counter()
    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio * n.distance:
                good.append(m)
    t_match_end = time.perf_counter()
    
    metrics['Match_Time_ms'] = round((t_match_end - t_match_start) * 1000, 2)
    metrics['Good_Matches'] = len(good)

    if len(good) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            metrics['Inlier_Ratio'] = round(np.sum(mask) / len(good), 4)
            metrics['Match_Acc'] = f"{round((np.sum(mask) / len(matches)) * 100, 2)}%" if len(matches) > 0 else "0%"
    return metrics

@st.cache_data
def get_cloud_templates(file_name, path_map):
    # 提取纯文件名用于路径映射
    base_name = os.path.basename(file_name)
    rel_path = path_map.get(base_name)
    if not rel_path: return None
    api_url = f"https://api.github.com/repos/77shaxinyu/detection/contents/dataset_empty/{rel_path.replace('\\', '/')}"
    try:
        res = requests.get(api_url, timeout=5).json()
        for item in res:
            if item["name"].lower().endswith((".jpg", ".png", ".jpeg")):
                data = requests.get(item["download_url"]).content
                return cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    except: pass
    return None

# ==========================================
# 3. Streamlit UI (多级目录全扫描版)
# ==========================================
st.set_page_config(page_title="Deep Folder Scanner", layout="wide")

if os.path.exists("path_index.json"):
    with open("path_index.json", "r", encoding="utf-8") as f: path_map = json.load(f)
else: path_map = {}

with st.sidebar:
    st.header("Algorithm Control")
    algo_choice = st.selectbox("Algorithm", ["Algorithm 1 (SIFT)", "Algorithm 2 (ORB)"])
    if st.button("Reset Records"):
        st.session_state.perf_history = []
        st.rerun()

if "perf_history" not in st.session_state: st.session_state.history = []

st.title("📂 PCB Multilevel Folder Performance Scanner")
st.warning("⚠️ **重要操作说明**：请直接将**整个父文件夹**拖入下方的上传框，或点击上传后进入该文件夹，按 **Ctrl+A** 全选所有内容（包括子文件夹）。Streamlit 会自动展平所有子目录下的图片。")

# 核心：允许批量上传，这是扫描多目录图片的唯一途径
uploaded_files = st.file_uploader("Drop Parent Folder / 拖入整个文件夹", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Start Global Scan / 开始全目录扫描"):
        progress_bar = st.progress(0)
        st.session_state.perf_history = [] # 每次扫描清空旧数据
        
        for i, f in enumerate(uploaded_files):
            # 将文件读取为 OpenCV 格式
            file_bytes = np.frombuffer(f.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, 1)
            
            # 获取对应的模板
            tpl = get_cloud_templates(f.name, path_map)
            
            if tpl is not None:
                perf = get_alignment_metrics(tpl, img_bgr, algo_choice)
                perf['Filename'] = f.name # 保留相对路径名
                perf['Algorithm'] = algo_choice
                st.session_state.perf_history.append(perf)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        st.success(f"Global Scan Complete! Processed {len(st.session_state.perf_history)} images from all subdirectories.")

# 只输出 Algorithm Performance Metrics
if "perf_history" in st.session_state and st.session_state.perf_history:
    st.subheader("📊 Algorithm Performance Metrics (All Subdirectories)")
    df = pd.DataFrame(st.session_state.perf_history)
    
    # 定义展示列
    target_cols = ['Filename', 'Algorithm', 'Keypoints', 'Good_Matches', 'Match_Acc', 'Inlier_Ratio', 'Det_Time_ms', 'Match_Time_ms']
    df = df[[c for c in target_cols if c in df.columns]]
    
    st.dataframe(df, use_container_width=True)

    # 导出逻辑
    col1, col2 = st.columns(2)
    csv = df.to_csv(index=False).encode('utf-8-sig')
    col1.download_button("Download CSV Metrics", csv, "batch_performance.csv", "text/csv", use_container_width=True)
    
    # 导出 Word
    doc = Document()
    doc.add_heading('Multilevel Folder Performance Report', 0)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Table Grid'
    for i, col in enumerate(df.columns): table.rows[0].cells[i].text = col
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row): cells[i].text = str(val)
    bio = io.BytesIO()
    doc.save(bio)
    col2.download_button("Download Word Summary", bio.getvalue(), "Subdirectory_Performance.docx", use_container_width=True)
