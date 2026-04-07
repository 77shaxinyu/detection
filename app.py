import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import sys
import zipfile
import shutil
import io
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
st.set_page_config(page_title="PCB Detection System", layout="wide")

# 方法一：注入自定义 CSS 实现黑底白字 [cite: 322, 369]
st.markdown("""
    <style>
    /* 主背景与文字颜色 */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363d;
    }
    /* 强制所有层级标题、标签和文本为白色 */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #FFFFFF !important;
    }
    /* 输入框与下拉菜单样式 */
    .stSelectbox div[data-baseweb="select"], .stNumberInput input {
        background-color: #0D1117 !important;
        color: white !important;
    }
    /* 数据表格深色适配 */
    .stDataFrame, [data-testid="stTable"] {
        background-color: #161B22;
        border: 1px solid #30363d;
    }
    /* 按钮样式 */
    .stButton>button {
        background-color: #21262d;
        color: white;
        border: 1px solid #30363d;
        width: 100%;
    }
    .stButton>button:hover {
        border-color: #8b949e;
        color: #58a6ff;
    }
    /* 分割线 */
    hr {
        border-color: #30363d;
    }
    /* 文件上传器样式 */
    [data-testid="stFileUploadDropzone"] {
        background-color: #161B22;
        border: 1px dashed #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

TEMP_DIR = "temp_results"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)


# ==========================================
# 2. Custom Modules Injection / 模块注入 [cite: 93, 345-348]
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
        self.m = nn.ModuleList(
            block.Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        try:
            return self.cbam(self.cv2(torch.cat(y, 1)))
        except:
            return self.attn(self.cv2(torch.cat(y, 1)))

# 运行时模块注入 [cite: 348-350]
block.C2f = C2f_Custom
tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM)
setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention)
setattr(tasks, 'SEAttention', SEAttention)


# ==========================================
# 3. Helper Functions / 工具函数 [cite: 342, 344]
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

def process_features(image, algorithm):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    out_img = image.copy()
    if "Algorithm 1" in algorithm:  # SIFT [cite: 131, 150]
        sift = cv2.SIFT_create()
        kp, _ = sift.detectAndCompute(gray, None)
        for p in kp: cv2.circle(out_img, (int(p.pt[0]), int(p.pt[1])), 3, (0, 255, 255), -1)
    else:  # ORB [cite: 131, 175]
        orb = cv2.ORB_create(nfeatures=2000)
        kp, _ = orb.detectAndCompute(gray, None)
        for p in kp: cv2.circle(out_img, (int(p.pt[0]), int(p.pt[1])), 3, (255, 0, 255), -1)
    return out_img


# ==========================================
# 4. Main Controller / 核心控制逻辑
# ==========================================
with st.sidebar:
    st.header("Configuration / 配置")
    proc_mode = st.radio("Processing Mode / 处理模式", ["Interactive (交互预览)", "Fast Batch Scan (快速批量扫描)"])
    model_choice = st.selectbox("Select Model / 选择检测模型", ["Model 1", "Model 2"])
    algo_choice = st.selectbox("Select Algorithm / 选择定位算法", ["Algorithm 1", "Algorithm 2"])
    conf_thresh = st.slider("Confidence Threshold / 置信度阈值", 0.1, 1.0, 0.25)

    if st.button("Clear Records / 清空记录"):
        st.session_state.history = []
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        st.rerun()

@st.cache_resource
def load_pcb_model(choice):
    path = "models/se.pt" if choice == "Model 1" else "models/cbam.pt"
    return YOLO(path) if os.path.exists(path) else None

model = load_pcb_model(model_choice)
if "history" not in st.session_state:
    st.session_state.history = []

st.title("PCB Defect Detection & Analysis System")
uploaded_files = st.file_uploader("Upload PCB Images / 上传图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if model is None:
        st.error("Model not found in models/ folder.")
    else:
        if proc_mode == "Fast Batch Scan (快速批量扫描)": [cite: 333-335]
            st.info(f"Fast scanning images...")
            progress_bar = st.progress(0)
            current_scan_data = []

            for idx, file in enumerate(uploaded_files):
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                h, w = img.shape[:2]
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
                            "Coordinates / 坐标": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})"
                        })
                progress_bar.progress((idx + 1) / len(uploaded_files))

            st.session_state.history.extend(current_scan_data)
            if st.session_state.history:
                st.divider()
                df_all = pd.DataFrame(st.session_state.history)
                st.subheader("Consolidated Defect Report / 缺陷汇总表格") [cite: 335, 407]
                st.dataframe(df_all, use_container_width=True)

                zip_buffer = io.BytesIO() [cite: 336]
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    csv_data = df_all.to_csv(index=False, encoding='utf-8-sig')
                    zip_file.writestr("total_defects_report.csv", csv_data)
                
                st.download_button(
                    label="Download Consolidated ZIP (CSV Report) / 导出汇总报告",
                    data=zip_buffer.getvalue(),
                    file_name=f"Batch_Scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

        else: # Interactive Preview Mode [cite: 340-344]
            for file in uploaded_files:
                file_base = os.path.splitext(file.name)[0]
                target_path = os.path.join(TEMP_DIR, f"{file_base}_{model_choice}_annotated.jpg")

                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_feat = process_features(img_rgb.copy(), algo_choice)
                grid_img, ch, cw = draw_grid_9x9(img_feat)

                results = model.predict(img, conf=conf_thresh, iou=0.45, agnostic_nms=True)
                st.session_state.history = [d for d in st.session_state.history if d['File'] != file.name]

                img_data_list = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = model.names[int(box.cls[0])]
                        pos = get_grid_pos((x1 + x2) / 2, (y1 + y2) / 2, ch, cw)
                        entry = {
                            "File": file.name, "Type / 类型": get_component_type(cls),
                            "Class / 类别": cls, "Confidence / 置信度": f"{float(box.conf[0]):.2f}",
                            "Grid / 网格": pos, "Coordinates / 坐标": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})"
                        }
                        st.session_state.history.append(entry)
                        img_data_list.append(entry)
                        cv2.rectangle(grid_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(grid_img, f"{cls} {pos}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                cv2.imwrite(target_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))

            if st.session_state.history:
                st.divider()
                df_all = pd.DataFrame(st.session_state.history)
                last_file_name = uploaded_files[-1].name
                df_curr = df_all[df_all["File"] == last_file_name]

                # 物理计数算法
                def get_physical_count(df_subset):
                    num_boxes = len(df_subset)
                    if num_boxes == 0: return 0
                    return 1 if num_boxes == 1 else int(num_boxes / 2)

                res_c = get_physical_count(df_curr[df_curr["Type / 类型"].str.contains("Resistor")])
                cap_c = get_physical_count(df_curr[df_curr["Type / 类型"].str.contains("Capacitor")])

                c1, c2, c3 = st.columns(3)
                c1.info(f"Resistors / 电阻: {res_c}")
                c2.info(f"Capacitors / 电容: {cap_c}")
                c3.success(f"Total / 总计: {res_c + cap_c}")

                f_base = os.path.splitext(last_file_name)[0]
                display_path = os.path.join(TEMP_DIR, f"{f_base}_{model_choice}_annotated.jpg")
                if os.path.exists(display_path):
                    st.image(cv2.cvtColor(cv2.imread(display_path), cv2.COLOR_BGR2RGB))
                st.dataframe(df_curr, use_container_width=True)
