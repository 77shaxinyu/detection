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
# 1. System Initialization
# ==========================================
TEMP_DIR = "temp_results"
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 2. Custom Modules Injection (CBAM/SE)
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

block.C2f = C2f_Custom
tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM)
setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention)
setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 3. Helper Functions & Feature Algo
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

def sanitize_name(name):
    return name.replace(" ", "_")

def get_component_type(class_name):
    name_lower = class_name.lower()
    if "resistor" in name_lower:
        return "Resistor / 电阻"
    else:
        return "Capacitor / 电容"

def process_features(image, algorithm):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    out_img = image.copy()
    if "Algorithm 1" in algorithm:
        try:
            sift = cv2.SIFT_create()
            keypoints, _ = sift.detectAndCompute(gray, None)
            for kp in keypoints:
                cv2.circle(out_img, (int(kp.pt[0]), int(kp.pt[1])), 3, (0, 255, 255), -1)
        except: return image, 0
    elif "Algorithm 2" in algorithm:
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints, _ = orb.detectAndCompute(gray, None)
        for kp in keypoints:
            cv2.circle(out_img, (int(kp.pt[0]), int(kp.pt[1])), 3, (255, 0, 255), -1)
    return out_img, len(keypoints)

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="PCB Detection System", layout="wide")

with st.sidebar:
    st.header("Configuration / 配置")
    model_choice = st.selectbox("Select Model / 选择检测模型", ["Model 1", "Model 2"])
    algo_choice = st.selectbox("Select Algorithm / 选择定位算法", ["Algorithm 1", "Algorithm 2"])
    conf_thresh = st.slider("Confidence Threshold / 置信度阈值", 0.1, 1.0, 0.25)
    st.divider()
    if st.button("Clear Records / 清空记录"):
        st.session_state.history = []
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        st.rerun()

st.title("PCB Detection System / PCB检测系统")

@st.cache_resource
def load_pcb_model(choice):
    # 更新后的文件名路径
    path = "models/se.pt" if choice == "Model 1" else "models/cbam.pt"
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_pcb_model(model_choice)
if "history" not in st.session_state: st.session_state.history = []

uploaded_files = st.file_uploader("Upload PCB Images / 上传图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if model is None:
        st.error("Model file not found in 'models/' folder. / 未在models文件夹中找到模型文件。")
    else:
        for file in uploaded_files:
            if not any(d['File'] == file.name for d in st.session_state.history):
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_feat, _ = process_features(img_rgb.copy(), algo_choice)
                grid_img, ch, cw = draw_grid_9x9(img_feat)
                
                # 强化NMS去重逻辑
                results = model.predict(img, conf=conf_thresh, iou=0.45, agnostic_nms=True)
                
                img_data_list = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls_name = model.names[int(box.cls[0])]
                        pos = get_grid_pos((x1+x2)/2, (y1+y2)/2, ch, cw)
                        entry = {
                            "File": file.name,
                            "Type / 类型": get_component_type(cls_name),
                            "Class / 类别": cls_name,
                            "Confidence / 置信度": f"{float(box.conf[0]):.2f}",
                            "Grid / 网格": pos,
                            "Coordinates / 坐标": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})"
                        }
                        st.session_state.history.append(entry)
                        img_data_list.append(entry)
                        cv2.rectangle(grid_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(grid_img, f"{cls_name} {pos}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                file_base = os.path.splitext(file.name)[0]
                cv2.imwrite(os.path.join(TEMP_DIR, f"{file_base}_annotated.jpg"), cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
                pd.DataFrame(img_data_list).to_csv(os.path.join(TEMP_DIR, f"{file_base}_data.csv"), index=False, encoding='utf-8-sig')

        if st.session_state.history:
            st.divider()
            df_all = pd.DataFrame(st.session_state.history)
            df_curr = df_all[df_all["File"] == uploaded_files[-1].name]
            res_c = int(len(df_curr[df_curr["Type / 类型"].str.contains("Resistor")]) / 2)
            cap_c = int(len(df_curr[df_curr["Type / 类型"].str.contains("Capacitor")]) / 2)
            c1, c2, c3 = st.columns(3)
            c1.info(f"Resistors / 电阻: {res_c}")
            c2.info(f"Capacitors / 电容: {cap_c}")
            c3.success(f"Total / 总计: {res_c + cap_c}")
            
            f_base = os.path.splitext(uploaded_files[-1].name)[0]
            st.image(cv2.cvtColor(cv2.imread(os.path.join(TEMP_DIR, f"{f_base}_annotated.jpg")), cv2.COLOR_BGR2RGB), use_column_width=True)
            st.dataframe(df_curr)

if st.session_state.history:
    st.divider()
    folder_struct = f"{sanitize_name(model_choice)}_{sanitize_name(algo_choice)}"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        u_files = list(set([h['File'] for h in st.session_state.history]))
        for f in u_files:
            b = os.path.splitext(f)[0]
            if os.path.exists(os.path.join(TEMP_DIR, f"{b}_annotated.jpg")):
                zf.write(os.path.join(TEMP_DIR, f"{b}_annotated.jpg"), f"{folder_struct}/{b}/{b}_annotated.jpg")
                zf.write(os.path.join(TEMP_DIR, f"{b}_data.csv"), f"{folder_struct}/{b}/{b}_data.csv")
    st.download_button("Download Results (ZIP) / 下载所有结果", zip_buf.getvalue(), f"{folder_struct}_Results.zip", "application/zip")
