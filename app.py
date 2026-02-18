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
# Define modules locally to prevent Cloud errors
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

# Apply Injection
block.C2f = C2f_Custom
tasks.C2f = C2f_Custom
setattr(block, 'CBAM', CBAM)
setattr(tasks, 'CBAM', CBAM)
setattr(block, 'SEAttention', SEAttention)
setattr(tasks, 'SEAttention', SEAttention)

# ==========================================
# 3. Helper Functions
# ==========================================
def draw_grid_9x9(image):
    h, w = image.shape[:2]
    grid_img = image.copy()
    cell_h, cell_w = h / 9, w / 9
    for i in range(1, 9):
        # Green lines
        cv2.line(grid_img, (int(i * cell_w), 0), (int(i * cell_w), h), (0, 255, 0), 1)
        cv2.line(grid_img, (0, int(i * cell_h)), (w, int(i * cell_h)), (0, 255, 0), 1)
    return grid_img, cell_h, cell_w

def get_grid_pos(x_center, y_center, cell_h, cell_w):
    col = chr(ord('A') + int(x_center / cell_w))
    row = int(y_center / cell_h) + 1
    return f"{col}{row}"

def sanitize_name(name):
    if "(" in name:
        return name.split("(")[1].split(")")[0]
    return name.replace(" ", "_")

def get_component_type(class_name):
    """
    Classify component type based on label name.
    Logic: 
    - Contains 'resistor' -> Resistor
    - Contains 'capacitor' or 'unused' -> Capacitor
    """
    name_lower = class_name.lower()
    if "resistor" in name_lower:
        return "Resistor / 电阻"
    else:
        # Assuming 'unused' classes are also capacitors as per user context
        return "Capacitor / 电容"

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="PCB Detection System", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration / 配置中心")
    
    model_choice = st.selectbox(
        "Select Model / 选择检测模型", 
        ["Model 2 (CBAM)", "Model 1 (SE)"]
    )
    
    algo_choice = st.selectbox(
        "Select Algorithm / 选择定位算法", 
        ["Algorithm 1 (SIFT)", "Algorithm 2 (ORB)"]
    )
    
    conf_thresh = st.slider(
        "Confidence Threshold / 置信度阈值", 
        0.1, 1.0, 0.25
    )
    
    st.divider()
    if st.button("Clear Records / 清空记录"):
        st.session_state.history = []
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        st.rerun()

# Main Title
st.title("PCB Component Counter & Locator / PCB元件计数与定位系统")

# Load Model
@st.cache_resource
def load_pcb_model(choice):
    path = "models/best_se.pt" if "SE" in choice else "models/best_cbam.pt"
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_pcb_model(model_choice)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

# File Uploader
uploaded_files = st.file_uploader(
    "Upload PCB Images / 上传图片 (支持批量)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    if model is None:
        st.error("Model file not found. Please check 'models' folder in GitHub. / 未找到模型文件，请检查GitHub仓库。")
    else:
        for file in uploaded_files:
            # Process new files only (simple check)
            if not any(d['File'] == file.name for d in st.session_state.history):
                
                # Decode Image
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Draw Grid
                grid_img, ch, cw = draw_grid_9x9(img_rgb)
                
                # Predict
                results = model.predict(img, conf=conf_thresh)
                
                # Data Collection for this image
                img_data_list = []
                
                for r in results:
                    for box in r.boxes:
                        # Extract Box Info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf_val = float(box.conf[0])
                        
                        # Process Logic
                        pos_val = get_grid_pos(cx, cy, ch, cw)
                        comp_type = get_component_type(cls_name)
                        coord_str = f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})"
                        
                        entry = {
                            "File": file.name,
                            "Type / 类型": comp_type,
                            "Class / 类别": cls_name,
                            "Confidence / 置信度": f"{conf_val:.2f}",
                            "Grid / 网格": pos_val,
                            "Coordinates / 坐标": coord_str,
                            "Time": datetime.now().strftime("%H:%M:%S")
                        }
                        
                        # Save to history
                        st.session_state.history.append(entry)
                        img_data_list.append(entry)
                        
                        # Draw on Image
                        cv2.rectangle(grid_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        # Text: Class + Grid
                        label_text = f"{cls_name} {pos_val}"
                        cv2.putText(grid_img, label_text, (int(x1), int(y1)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # --- Save Temp Files for Export ---
                file_base = os.path.splitext(file.name)[0]
                
                # Save Image (BGR for opencv)
                save_img_path = os.path.join(TEMP_DIR, f"{file_base}_annotated.jpg")
                cv2.imwrite(save_img_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
                
                # Save CSV
                save_csv_path = os.path.join(TEMP_DIR, f"{file_base}_data.csv")
                pd.DataFrame(img_data_list).to_csv(save_csv_path, index=False, encoding='utf-8-sig')

        # --- Display Results ---
        if st.session_state.history:
            st.divider()
            
            # Get data for the last uploaded/processed file
            df_all = pd.DataFrame(st.session_state.history)
            last_file_name = uploaded_files[-1].name
            df_curr = df_all[df_all["File"] == last_file_name]
            
            # --- Counting Logic (Divide by 2) ---
            # Filter by Type string defined in get_component_type
            res_df = df_curr[df_curr["Type / 类型"].str.contains("Resistor")]
            cap_df = df_curr[df_curr["Type / 类型"].str.contains("Capacitor")]
            
            # Count boxes
            res_boxes = len(res_df)
            cap_boxes = len(cap_df)
            
            # Calculate Components (Boxes / 2)
            res_comps = res_boxes / 2
            cap_comps = cap_boxes / 2
            total_comps = res_comps + cap_comps
            
            # Display Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info(f"Resistors / 电阻: {res_comps:.1f}")
            with c2:
                st.info(f"Capacitors / 电容: {cap_comps:.1f}")
            with c3:
                st.success(f"Total / 总计: {total_comps:.1f}")
            
            # Display Image and Table
            # Note: We need to reload the temp image to show the latest processed one
            file_base = os.path.splitext(last_file_name)[0]
            temp_img_show_path = os.path.join(TEMP_DIR, f"{file_base}_annotated.jpg")
            
            if os.path.exists(temp_img_show_path):
                # Using opencv to read and st.image to display
                show_img = cv2.imread(temp_img_show_path)
                st.image(cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB), 
                         caption=f"Result: {last_file_name}", use_column_width=True)
            
            st.dataframe(df_curr)

# ==========================================
# 5. Export Logic (ZIP)
# ==========================================
if st.session_state.history:
    st.divider()
    
    # Create ZIP name based on config
    m_name = sanitize_name(model_choice)
    a_name = sanitize_name(algo_choice)
    folder_struct = f"{m_name}_{a_name}" # e.g. CBAM_SIFT
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Get unique files from history
        unique_files = list(set([h['File'] for h in st.session_state.history]))
        
        for fname in unique_files:
            base = os.path.splitext(fname)[0]
            
            # Paths in temp dir
            t_img = os.path.join(TEMP_DIR, f"{base}_annotated.jpg")
            t_csv = os.path.join(TEMP_DIR, f"{base}_data.csv")
            
            if os.path.exists(t_img) and os.path.exists(t_csv):
                # Structure: Model_Algo / ImageName / Files
                z_img = f"{folder_struct}/{base}/{base}_annotated.jpg"
                z_csv = f"{folder_struct}/{base}/{base}_data.csv"
                
                zf.write(t_img, z_img)
                zf.write(t_csv, z_csv)
    
    st.download_button(
        label="Download All Results (ZIP) / 下载所有结果",
        data=zip_buffer.getvalue(),
        file_name=f"{folder_struct}_Results.zip",
        mime="application/zip"
    )
