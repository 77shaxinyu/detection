import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import zipfile
import shutil
import io
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block
from ultralytics.nn.modules.conv import Conv

# ==========================================
# 1. System Initialization / 系统初始化
# ==========================================
TEMP_DIR = "temp_results"
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
        try:
            return self.cbam(self.cv2(torch.cat(y, 1)))
        except:
            return self.attn(self.cv2(torch.cat(y, 1)))

# 修正模块注入逻辑
block.C2f = C2f_Custom
tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM)
setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention)
setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 3. Helper Functions / 工具函数
# ==========================================
def calculate_clustered_count(df_subset, dist_threshold=65):
    """
    基于中心点距离的聚类算法：
    - 两个框距离近算1个物理元件
    - 单个框周围没邻居也算1个物理元件（防止漏检导致计数归零）
    """
    if df_subset.empty: return 0
    centers = []
    for coord_str in df_subset["Coordinates / 坐标"]:
        c = [int(v) for v in coord_str.strip("()").split(",")]
        centers.append([(c[0] + c[2]) / 2, (c[1] + c[3]) / 2])
    
    centers = np.array(centers)
    n = len(centers)
    visited = [False] * n
    actual_count = 0
    for i in range(n):
        if not visited[i]:
            actual_count += 1
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j]:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < dist_threshold:
                        visited[j] = True
    return actual_count

def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    ch, cw = h / 9, w / 9
    for i in range(1, 9):
        cv2.line(grid_img, (int(i * cw), 0), (int(i * cw), h), (0, 255, 0), 1)
        cv2.line(grid_img, (0, int(i * ch)), (w, int(i * ch)), (0, 255, 0), 1)
    return grid_img, ch, cw

def get_grid_pos(x_c, y_c, ch, cw):
    col = chr(ord('A') + int(x_c / cw))
    row = int(y_c / ch) + 1
    return f"{col}{row}"

def get_component_type(class_name):
    return "电阻" if "resistor" in class_name.lower() else "电容"

# ==========================================
# 4. Streamlit UI / 界面显示
# ==========================================
st.set_page_config(page_title="PCB智能检测系统", layout="wide")

with st.sidebar:
    st.header("Configuration / 配置")
    proc_mode = st.radio("处理模式", ["交互预览", "快速批量扫描"])
    model_choice = st.selectbox("选择模型", ["Model 1", "Model 2"])
    conf_thresh = st.slider("置信度阈值", 0.1, 1.0, 0.25)
    dist_thresh = st.slider("聚类判定距离(px)", 10, 200, 65)
    
    if st.button("清空系统记录"):
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

# 文件上传
uploaded_files = st.file_uploader("上传PCB图片", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files and model:
    for file in uploaded_files:
        # 批量模式防重复逻辑
        if any(d['File'] == file.name for d in st.session_state.history if proc_mode == "快速批量扫描"):
            continue
            
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h, w = img.shape[:2]
        
        results = model.predict(img, conf=conf_thresh, verbose=False)
        
        if proc_mode == "交互预览":
            grid_img, ch, cw = draw_grid_9x9(img)
            st.session_state.history = [d for d in st.session_state.history if d['File'] != file.name]

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = model.names[int(box.cls[0])]
                pos = get_grid_pos((x1+x2)/2, (y1+y2)/2, h/9, w/9)
                
                st.session_state.history.append({
                    "File": file.name,
                    "Type / 类型": get_component_type(cls),
                    "Coordinates / 坐标": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                    "Grid / 网格": pos
                })
                
                if proc_mode == "交互预览":
                    cv2.rectangle(grid_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(grid_img, pos, (int(x1), int(y1)-10), 0, 0.6, (0, 255, 0), 2)

        if proc_mode == "交互预览":
            cv2.imwrite(os.path.join(TEMP_DIR, f"{file.name}_res.jpg"), grid_img)

    # 结果展示
    st.divider()
    df_all = pd.DataFrame(st.session_state.history)

    if proc_mode == "交互预览" and not df_all.empty:
        last_name = uploaded_files[-1].name
        df_curr = df_all[df_all["File"] == last_name]
        
        # 核心：物理计数
        res_c = calculate_clustered_count(df_curr[df_curr["Type / 类型"]=="电阻"], dist_thresh)
        cap_c = calculate_clustered_count(df_curr[df_curr["Type / 类型"]=="电容"], dist_thresh)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("电阻(物理元件数)", res_c)
        c2.metric("电容(物理元件数)", cap_c)
        c3.metric("总计", res_c + cap_c)
        
        st.image(os.path.join(TEMP_DIR, f"{last_name}_res.jpg"), use_container_width=True)
        st.dataframe(df_curr, use_container_width=True)
    else:
        st.dataframe(df_all, use_container_width=True)
