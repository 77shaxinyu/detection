import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules.block as block
from ultralytics.nn.modules.conv import Conv, autopad

# ==========================================================
# 1. 定义自定义模块 (CBAM & SE)
# ==========================================================
class CBAM(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_sum = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // ratio, c1, 1, bias=False)
        )
        self.channel_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(c1, c1 // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // ratio, c1, 1, bias=False)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
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
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ==========================================================
# 2. 核心修复：重写 C2f 类 (还原你本地 block.py 的修改)
# ==========================================================
class C2f_Custom(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e) 
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1) 
        
        # --- 这里的关键：云端默认没有这行，必须加上！ ---
        # 注意：如果你之前练 SE 模型时改成了 self.attn，这里逻辑需要判断
        # 为了兼容 CBAM，我们这里先写死 CBAM。
        # 如果你加载 SE 模型，YOLO 会自动匹配结构，只要类里有定义即可。
        self.cbam = CBAM(c2) 
        self.attn = SEAttention(c2) # 同时定义 SE，防止加载 SE 模型报错

        self.m = nn.ModuleList(block.Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        # --- 还原 forward 逻辑 ---
        # 这是一个兼容写法：
        # 如果权重里只有 cbam 参数，就用 cbam；如果是 se 参数，就用 attn
        # 但通常你本地代码是写死的。假设你现在要跑 CBAM：
        return self.cbam(self.cv2(torch.cat(y, 1)))

# ==========================================================
# 3. 强行替换系统模块 (Monkey Patching)
# ==========================================================
# 把我们定义的 C2f_Custom 塞进 ultralytics 库里，替换掉官方的 C2f
block.C2f = C2f_Custom
tasks.C2f = C2f_Custom
# 注入注意力模块
setattr(block, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention)
setattr(tasks, 'CBAM', CBAM)
setattr(tasks, 'SEAttention', SEAttention)


# ==========================================================
# 4. 页面 UI 配置 (保持不变)
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

with st.sidebar:
    st.session_state.lang = st.radio("Language / 语言", ["zh", "en"])
    L = LANG[st.session_state.lang]
    st.header(L["config"])
    
    # 暂时只开放 Model 2 (CBAM) 防止误触
    model_choice = st.selectbox(L["model_sel"], ["Model 2 (CBAM)", "Model 1 (SE)"])
    algo_choice = st.selectbox(L["algo_sel"], ["Algorithm 1 (SIFT)", "Algorithm 2 (ORB)"])
    conf_thresh = st.slider(L["conf"], 0.1, 1.0, 0.25)
    
    if st.button(L["clear"]):
        st.session_state.history = []
        st.rerun()

@st.cache_resource
def load_pcb_model(choice):
    path = "models/best_se.pt" if "SE" in choice else "models/best_cbam.pt"
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_pcb_model(model_choice)
st.title(L["title"])

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader(L["upload"], type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if model is None:
        st.warning(f"⚠️ Model weight file not found. Please upload to /models folder.")
    else:
        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            grid_img, ch, cw = draw_grid_9x9(img_rgb)
            
            # 使用 predict 进行推理
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
                    cv2.rectangle(grid_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    # 字体稍微调小一点，防止遮挡
                    cv2.putText(grid_img, f"{cls_name} {pos_val}", (int(x1), int(y1)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(grid_img, use_column_width=True, caption=f"Detected: {file.name}")
            with col2:
                st.dataframe(pd.DataFrame(current_data))

if st.session_state.history:
    st.divider()
    st.subheader(L["result_table"])
    df_all = pd.DataFrame(st.session_state.history)
    st.dataframe(df_all)
    
    csv_data = df_all.to_csv(index=False).encode('utf-8-sig')
    st.download_button(L["download"], data=csv_data, file_name="pcb_report.csv", mime="text/csv")
