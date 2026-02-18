import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block

# ==========================================================
# 1. 核心修复：强行注入自定义模块 (必须在加载模型前执行)
# ==========================================================
try:
    # 确保 my_blocks.py 在根目录
    import my_blocks
    from my_blocks import CBAM, SEAttention
    
    # 注入到任务模块，解决加载权重时的 pickle.load 问题
    setattr(tasks, 'CBAM', CBAM)
    setattr(tasks, 'SEAttention', SEAttention)
    
    # 注入到模块定义中，解决网络构建时的实例化问题
    setattr(block, 'CBAM', CBAM)
    setattr(block, 'SEAttention', SEAttention)
    
    # 兼容性补丁：将类直接挂载到 sys.modules 的顶级路径
    sys.modules['ultralytics.nn.tasks'].CBAM = CBAM
    sys.modules['ultralytics.nn.tasks'].SEAttention = SEAttention
except Exception as e:
    st.error(f"Class Injection Error: {e}")

# ==========================================================
# 2. 页面配置与多语言字典
# ==========================================================
st.set_page_config(page_title="PCB Detection System", layout="wide")

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

if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# ==========================================================
# 3. 核心工具函数
# ==========================================================
def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    cell_h, cell_w = h / 9, w / 9
    for i in range(1, 9):
        # 绘制绿色网格线
        cv2.line(grid_img, (int(i * cell_w), 0), (int(i * cell_w), h), (0, 255, 0), 2)
        cv2.line(grid_img, (0, int(i * cell_h)), (w, int(i * cell_h)), (0, 255, 0), 2)
    return grid_img, cell_h, cell_w

def get_grid_pos(x_center, y_center, cell_h, cell_w):
    # 将坐标映射为 A1-I9
    col = chr(ord('A') + int(x_center / cell_w))
    row = int(y_center / cell_h) + 1
    return f"{col}{row}"

# ==========================================================
# 4. 侧边栏与模型加载
# ==========================================================
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

@st.cache_resource
def load_pcb_model(choice):
    path = "models/best_se.pt" if "SE" in choice else "models/best_cbam.pt"
    if os.path.exists(path):
        # 加载时，由于已经注入了类，这里不会再报 AttributeError
        return YOLO(path)
    return None

model = load_pcb_model(model_choice)
st.title(L["title"])

# ==========================================================
# 5. 主逻辑：检测与定位
# ==========================================================
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader(L["upload"], type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if model is None:
        st.warning(f"Model weight file for {model_choice} not found in /models folder.")
    else:
        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            grid_img, ch, cw = draw_grid_9x9(img_rgb)
            results = model.predict(img, conf=conf_thresh)
            
            current_data = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    cls_name = model.names[int(box.cls[0])]
                    conf_val = float(box.conf[0])
                    pos_val = get_grid_pos(cx, cy, ch, cw)
                    
                    entry = {
                        "File": file.name,
                        L["cls"]: cls_name,
                        "Confidence": f"{conf_val:.2f}",
                        L["pos"]: pos_val,
                        L["time"]: datetime.now().strftime("%H:%M:%S")
                    }
                    current_data.append(entry)
                    st.session_state.history.append(entry)
                    
                    # 绘制检测框
                    cv2.rectangle(grid_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                    cv2.putText(grid_img, f"{cls_name} {pos_val}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(grid_img, use_column_width=True)
            with col2:
                st.dataframe(pd.DataFrame(current_data))

# ==========================================================
# 6. 数据导出
# ==========================================================
if st.session_state.history:
    st.divider()
    st.subheader(L["result_table"])
    df_all = pd.DataFrame(st.session_state.history)
    st.dataframe(df_all)
    
    csv_data = df_all.to_csv(index=False).encode('utf-8-sig')
    st.download_button(L["download"], data=csv_data, file_name="pcb_report.csv", mime="text/csv")
