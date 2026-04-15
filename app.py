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
# 1. 模型注入 (保持你的 CBAM/SE 架构)
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
        self.c = int(c2 * e); self.cv1 = Conv(c1, 2 * self.c, 1, 1); self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.cbam = CBAM(c2); self.attn = SEAttention(c2)
        self.m = nn.ModuleList(block.Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1)); y.extend(m(y[-1]) for m in self.m)
        try: return self.cbam(self.cv2(torch.cat(y, 1)))
        except: return self.attn(self.cv2(torch.cat(y, 1)))

block.C2f = tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM); setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention); setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 2. 严格遵循你本地的 SIFT 逻辑函数
# ==========================================
def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    for i in range(1, 9):
        cv2.line(grid_img, (int(i * w / 9), 0), (int(i * w / 9), h), (0, 255, 0), 2)
        cv2.line(grid_img, (0, int(i * h / 9)), (w, int(i * h / 9)), (0, 255, 0), 2)
    return grid_img

def process_sift_alignment_STRICT(tpl_img, test_img):
    """【严格按照你本地的 SIFT 逻辑实现】"""
    sift = cv2.SIFT_create(nfeatures=2000)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None: return 0, None
    
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    # 严格使用你代码中的 0.8 比例
    good = [m for m, n in matches if m.distance < 0.8 * n.distance]
    
    M = None
    if len(good) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return len(good), M

def get_roi_detect_STRICT(img, M, model, conf, tpl_h, tpl_w):
    """【严格按照你本地的 ROI 裁剪与坐标还原逻辑】"""
    # 你的 ROI 定义：左右 40% 区域
    roi_defs = [
        {"box": [[0, 0], [0, tpl_h], [tpl_w * 0.4, tpl_h], [tpl_w * 0.4, 0]]},
        {"box": [[tpl_w * 0.6, 0], [tpl_w * 0.6, tpl_h], [tpl_w, tpl_h], [tpl_w, 0]]}
    ]
    boxes = []
    for r in roi_defs:
        pts = np.float32(r["box"]).reshape(-1, 1, 2)
        # 单应性映射
        dst = cv2.perspectiveTransform(pts, M)
        x, y, rw, rh = cv2.boundingRect(dst)
        x, y = max(0, x), max(0, y)
        crop = img[y:y + rh, x:x + rw]
        if crop.size > 0:
            res = model.predict(crop, conf=conf, verbose=False)
            for r_obj in res:
                for b in r_obj.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                    # 关键坐标还原：裁剪偏移量叠加
                    boxes.append({
                        "xyxy": [bx1 + x, by1 + y, bx2 + x, by2 + y], 
                        "cls": int(b.cls[0]), 
                        "conf": float(b.conf[0])
                    })
    return boxes

# ==========================================
# 3. GUI 与云端获取逻辑
# ==========================================
st.set_page_config(page_title="PCB Cascade Inspector", layout="wide")

@st.cache_data
def get_cloud_tpl(rel_path):
    api_url = f"https://api.github.com/repos/77shaxinyu/detection/contents/dataset_empty/{rel_path.replace('\\', '/')}"
    try:
        r = requests.get(api_url, timeout=5).json()
        f = next((i for i in r if i['name'].lower().endswith(('.jpg', '.png'))), None)
        if f:
            data = requests.get(f['download_url']).content
            return cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    except: pass
    return None

# 加载路径索引
if os.path.exists("path_index.json"):
    with open("path_index.json", "r", encoding="utf-8") as f:
        path_map = json.load(f)
else:
    st.error("Missing path_index.json"); st.stop()

with st.sidebar:
    st.header("⚙️ 参数")
    model_choice = st.selectbox("模型", ["Model 1 (SE)", "Model 2 (CBAM)"])
    conf_thresh = st.slider("置信度", 0.1, 1.0, 0.25)

@st.cache_resource
def load_yolo(choice):
    p = f"models/{'se' if 'SE' in choice else 'cbam'}.pt"
    return YOLO(p) if os.path.exists(p) else None

model = load_yolo(model_choice)
files = st.file_uploader("上传待检图", accept_multiple_files=True)

if files and model:
    history = []
    for f in files:
        # 1. 读入测试图
        live_img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
        rel_path = path_map.get(f.name, "")
        
        with st.spinner(f"正在匹配云端模板并级联检测: {f.name}..."):
            # 2. 抓取模板图
            tpl_img = get_cloud_tpl(rel_path)
            
            final_boxes = []
            if tpl_img is not None:
                # 3. 严格运行本地 SIFT 逻辑
                tpl_h, tpl_w = tpl_img.shape[:2]
                match_cnt, M = process_sift_alignment_STRICT(tpl_img, live_img)
                
                if M is not None and match_cnt >= 8:
                    # 4. 严格运行本地 ROI 检测逻辑
                    final_boxes = get_roi_detect_STRICT(live_img, M, model, conf_thresh, tpl_h, tpl_w)
            
            # 5. 保底策略：若对齐失败则全图检测
            if not final_boxes:
                res = model.predict(live_img, conf=conf_thresh, verbose=False)
                for r_obj in res:
                    for b in r_obj.boxes:
                        final_boxes.append({"xyxy": b.xyxy[0].cpu().numpy(), "cls": int(b.cls[0]), "conf": float(b.conf[0])})

        # 6. 绘图展示
        canvas = draw_grid_9x9(live_img)
        for b in final_boxes:
            x1, y1, x2, y2 = map(int, b["xyxy"])
            cls_name = model.names[b["cls"]]
            # 严格按照你本地的绘图颜色和样式
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(canvas, f"{cls_name} {b['conf']:.2f}", (x1, y1 - 10), 0, 1.2, (0, 255, 255), 2)
            history.append({"File": f.name, "Class": cls_name, "Confidence": f"{b['conf']:.2f}"})
        
        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"Result: {f.name}")
    
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
