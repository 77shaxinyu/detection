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
# 1. 模型层定义与动态注入 (CBAM/SE/C2f)
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
        x = x * self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x

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

# 覆盖官方 YOLO 模块
block.C2f = tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM); setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention); setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 2. 核心：云端自动寻址 + 缓存逻辑
# ==========================================
GITHUB_API_BASE = "https://api.github.com/repos/77shaxinyu/detection/contents/dataset_empty/"

@st.cache_data
def get_template_img(rel_path):
    """自动通过 API 获取文件夹下第一张图并缓存"""
    api_url = f"{GITHUB_API_BASE}{rel_path.replace('\\', '/')}"
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            files = response.json()
            tpl_info = next((f for f in files if f['name'].lower().endswith(('.jpg', '.png'))), None)
            if tpl_info:
                img_res = requests.get(tpl_info['download_url'])
                nparr = np.frombuffer(img_res.content, np.uint8)
                return cv2.imdecode(nparr, 1), tpl_info['name']
    except: pass
    return None, None

# ==========================================
# 3. 算法处理：本地 SIFT 级联逻辑
# ==========================================
def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    for i in range(1, 9):
        cv2.line(grid_img, (int(i * w / 9), 0), (int(i * w / 9), h), (0, 255, 0), 2)
        cv2.line(grid_img, (0, int(i * h / 9)), (w, int(i * h / 9)), (0, 255, 0), 2)
    return grid_img

def cascade_process(tpl_img, test_img, model, conf):
    """执行你要求的本地 SIFT + ROI 检测逻辑"""
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

    boxes_list = []
    if M is not None:
        # ROI 定义 (左右 40%)
        roi_defs = [{"box": [[0, 0], [0, h], [w * 0.4, h], [w * 0.4, 0]]},
                    {"box": [[w * 0.6, 0], [w * 0.6, h], [w, h], [w, 0]]}]
        for r in roi_defs:
            pts = np.float32(r["box"]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            rx, ry, rw, rh = cv2.boundingRect(dst)
            rx, ry = max(0, rx), max(0, ry)
            crop = test_img[ry:ry + rh, rx:rx + rw]
            if crop.size > 0:
                res = model.predict(crop, conf=conf, verbose=False)
                for r_obj in res:
                    for b in r_obj.boxes:
                        bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                        boxes_list.append({
                            "xyxy": [bx1 + rx, by1 + ry, bx2 + rx, by2 + ry],
                            "cls": int(b.cls[0]), "conf": float(b.conf[0])
                        })
    else:
        # 对齐失败，全图检测
        res = model.predict(test_img, conf=conf, verbose=False)
        for r_obj in res:
            for b in r_obj.boxes:
                boxes_list.append({"xyxy": b.xyxy[0].cpu().numpy(), "cls": int(b.cls[0]), "conf": float(b.conf[0])})
    return boxes_list

# ==========================================
# 4. Streamlit UI 
# ==========================================
st.set_page_config(page_title="PCB SIFT-YOLO Cascade", layout="wide")

if os.path.exists("path_index.json"):
    with open("path_index.json", "r", encoding="utf-8") as f:
        path_index = json.load(f)
else:
    st.error("缺失索引文件 path_index.json")
    st.stop()

@st.cache_resource
def load_model(choice):
    path = f"models/{'se' if 'SE' in choice else 'cbam'}.pt"
    return YOLO(path) if os.path.exists(path) else None

with st.sidebar:
    st.header("⚙️ 参数")
    model_choice = st.selectbox("选择模型", ["Model 1 (SE)", "Model 2 (CBAM)"])
    conf_val = st.slider("置信度", 0.1, 1.0, 0.25)
    if st.button("清空历史记录"):
        st.session_state.history = []
        st.rerun()

model = load_model(model_choice)
if "history" not in st.session_state: st.session_state.history = []

files = st.file_uploader("上传 PCB 图片", accept_multiple_files=True)

if files and model:
    for f in files:
        rel_path = path_index.get(f.name)
        if not rel_path: continue
        
        with st.spinner(f"正在级联处理 {f.name}..."):
            tpl_img, tpl_name = get_template_img(rel_path)
            
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            live_img = cv2.imdecode(file_bytes, 1)
            
            if tpl_img is not None:
                final_boxes = cascade_process(tpl_img, live_img, model, conf_val)
                
                # 绘制结果
                canvas = draw_grid_9x9(live_img)
                for b in final_boxes:
                    x1, y1, x2, y2 = map(int, b["xyxy"])
                    cls = model.names[b["cls"]]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(canvas, f"{cls} {b['conf']:.2f}", (x1, y1-10), 0, 1.2, (0, 255, 255), 2)
                    st.session_state.history.append({"File": f.name, "Class": cls, "Conf": f"{b['conf']:.2f}"})
                
                st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"识别成功: {tpl_name}")
            else:
                st.error(f"无法匹配模板路径: {rel_path}")

    if st.session_state.history:
        st.divider()
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
