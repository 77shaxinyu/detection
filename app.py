import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# 1. Register Custom Modules
# This allows the cloud environment to recognize your CBAM and SE layers
try:
    from my_blocks import CBAM, SEAttention
    tasks.CBAM = CBAM
    tasks.SEAttention = SEAttention
except ImportError:
    st.error("Error: my_blocks.py not found in root directory.")

# 2. Page Configuration
st.set_page_config(page_title="PCB Detection System", layout="wide")

# 3. UI Language Dictionary
LANG = {
    "zh": {
        "title": "PCB 缺陷智能检测系统",
        "config": "配置中心",
        "model_sel": "选择检测模型",
        "algo_sel": "选择定位算法",
        "conf": "置信度阈值",
        "upload": "上传 PCB 图片 (支持多张)",
        "result_table": "检测结果汇总",
        "download": "下载检测报告",
        "clear": "清空记录",
        "pos": "网格位置",
        "cls": "组件类别",
        "time": "检测时间"
    },
    "en": {
        "title": "PCB Defect Detection System",
        "config": "Configuration",
        "model_sel": "Select Model",
        "algo_sel": "Select Algorithm",
        "conf": "Confidence Threshold",
        "upload": "Upload PCB Images (Multiple)",
        "result_table": "Detection Summary",
        "download": "Download Report",
        "clear": "Clear Records",
        "pos": "Grid Position",
        "cls": "Class",
        "time": "Time"
    }
}

# Language Selector
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# 4. Helper Functions
def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    cell_h, cell_w = h / 9, w / 9
    for i in range(1, 9):
        cv2.line(grid_img, (int(i * cell_w), 0), (int(i * cell_w), h), (0, 255, 0), 2)
        cv2.line(grid_img, (0, int(i * cell_h)), (w, int(i * cell_h)), (0, 255, 0), 2)
    return grid_img, cell_h, cell_w

def get_grid_pos(x_center, y_center, cell_h, cell_w):
    col = chr(ord('A') + int(x_center / cell_w))
    row = int(y_center / cell_h) + 1
    return f"{col}{row}"

# 5. Sidebar Configuration
with st.sidebar:
    st.session_state.lang = st.radio("Language / 语言", ["zh", "en"])
    L = LANG[st.session_state.lang]
    st.header(L["config"])
    
    model_choice = st.selectbox(L["model_sel"], ["Model 1 (SE)", "Model 2 (CBAM)"])
    algo_choice = st.selectbox(L["algo_sel"], ["Algorithm 1 (SIFT)", "Algorithm 2 (ORB)"])
    conf_thresh = st.slider(L["conf"], 0.1, 1.0, 0.25)
    
    if st.button(L["clear"]):
        st.session_state.history = []
        st.rerun()

st.title(L["title"])

# 6. Model Loading
@st.cache_resource
def load_model(choice):
    # Match the file names in your /models folder
    name = "models/best_se.pt" if "SE" in choice else "models/best_cbam.pt"
    if os.path.exists(name):
        return YOLO(name)
    return None

model = load_model(model_choice)

# 7. Main Logic
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader(L["upload"], type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and model:
    for file in uploaded_files:
        # Image Processing
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Grid and Detection
        grid_img, ch, cw = draw_grid_9x9(img_rgb)
        results = model.predict(img, conf=conf_thresh)
        
        current_data = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cls = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                pos = get_grid_pos(cx, cy, ch, cw)
                
                entry = {
                    "File": file.name,
                    L["cls"]: cls,
                    "Confidence": f"{conf:.2f}",
                    L["pos"]: pos,
                    L["time"]: datetime.now().strftime("%H:%M:%S")
                }
                current_data.append(entry)
                st.session_state.history.append(entry)
                
                # Draw boxes
                cv2.rectangle(grid_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

        # UI Layout
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(grid_img, use_column_width=True)
        with col2:
            st.dataframe(pd.DataFrame(current_data))

# 8. Export Section
if st.session_state.history:
    st.divider()
    st.subheader(L["result_table"])
    df_all = pd.DataFrame(st.session_state.history)
    st.dataframe(df_all)
    
    csv = df_all.to_csv(index=False).encode('utf-8-sig')
    st.download_button(L["download"], data=csv, file_name="pcb_report.csv", mime="text/csv")
