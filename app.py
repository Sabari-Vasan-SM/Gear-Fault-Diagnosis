# ------------------- app.py -------------------

# 1️⃣ Streamlit page config MUST come first
import streamlit as st
st.set_page_config(
    page_title="⚙️ Gear Fault Detection", 
    page_icon="🛠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2️⃣ Imports
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO 
from collections import Counter

# ------------------- Load YOLO model -------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # ensure best.pt is in the root folder

model = load_model()

# ------------------- Custom CSS -------------------
def load_custom_css():
    st.markdown("""
    <style>
    /* Global black theme */
    .stApp { background: #000000; color: #ffffff; }
    .main .block-container { padding: 1rem; max-width: 100%; }
    
    /* Grid layout */
    .grid-container { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; min-height: 100vh; background: #000; padding: 1rem; }
    .header-section { grid-column: 1 / -1; text-align: center; background: linear-gradient(45deg, #1a1a1a, #2d2d2d); border-radius: 15px; padding: 2rem; margin-bottom: 1rem; border: 1px solid #333; animation: fadeInDown 0.8s ease-out; }
    .main-title { background: linear-gradient(45deg, #00ff88, #00ccff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 700; margin: 0; }
    .subtitle { color: #888; font-size: 1.1rem; margin-top: 0.5rem; }
    
    /* Panels */
    .left-panel, .right-panel { background: linear-gradient(145deg, #111, #1a1a1a); border-radius: 15px; padding: 1.5rem; border: 1px solid #333; height: fit-content; }
    .left-panel { animation: slideInLeft 0.8s ease-out; }
    .right-panel { animation: slideInRight 0.8s ease-out; }
    
    /* Upload area */
    .upload-area { background: #0a0a0a; border: 2px dashed #333; border-radius: 10px; padding: 2rem; text-align: center; margin: 1rem 0; transition: all 0.3s ease; }
    .upload-area:hover { border-color: #00ff88; background: #0f0f0f; }
    
    /* Buttons */
    .stButton > button { background: linear-gradient(45deg, #00ff88, #00ccff); color: #000; border: none; border-radius: 10px; padding: 0.8rem 2rem; font-size: 1.1rem; font-weight: 700; width: 100%; margin: 1rem 0; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,255,136,0.3); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,255,136,0.5); background: linear-gradient(45deg, #00ccff, #00ff88); }

    /* Image card */
    .image-card { background: #0a0a0a; border-radius: 12px; padding: 1rem; margin: 1rem 0; border: 1px solid #333; transition: all 0.3s ease; }
    .image-title { color: #00ccff; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }

    /* Animations */
    @keyframes fadeInDown { from {opacity:0; transform:translateY(-30px);} to {opacity:1; transform:translateY(0);} }
    @keyframes slideInLeft { from {opacity:0; transform:translateX(-30px);} to {opacity:1; transform:translateX(0);} }
    @keyframes slideInRight { from {opacity:0; transform:translateX(30px);} to {opacity:1; transform:translateX(0);} }

    /* Hide Streamlit default elements */
    #MainMenu, footer, header, .stDeployButton { display:none; }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ------------------- Class Colors -------------------
class_colors = {
    "kp": (0, 255, 255),
    "hp_cm": (255, 0, 255),
    "hp_cd": (128, 0, 255)
}

def bgr_to_rgb(color_bgr):
    return (color_bgr[2]/255.0, color_bgr[1]/255.0, color_bgr[0]/255.0)

# ------------------- Detection Function -------------------
def predict_and_overlay(img):
    result = model.predict(source=img, imgsz=640, conf=0.25, save=False)[0]
    image = result.orig_img.copy()
    detected_labels = []

    for box in result.obb:
        cls_id = int(box.cls[0])
        label = result.names[cls_id]
        detected_labels.append(label)
        pts = box.xyxyxyxy[0].cpu().numpy().astype(int).reshape(-1, 2)

        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], class_colors[label])
        image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        cv2.polylines(image, [pts], isClosed=True, color=class_colors[label], thickness=2)
        x, y = pts[0]
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[label], 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detected_labels

# ------------------- UI -------------------
# Header
st.markdown('''
<div class="header-section">
    <h1 class="main-title">⚙️ GEAR FAULT DETECTION</h1>
    <p class="subtitle">Advanced AI-powered gear fault detection system</p>
</div>
''', unsafe_allow_html=True)

# Columns
col_left, col_right = st.columns([1,1], gap="large")

# ------------------- Left Panel -------------------
with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.markdown('<div class="image-card"><div class="image-title">📸 UPLOADED IMAGE</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        detect_button = st.button("🔍 START DETECTION", key="detect_btn")
    else:
        st.markdown('<p style="color:#888;text-align:center;">Drag & drop or click to upload JPG/PNG</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Right Panel -------------------
with col_right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    
    # Legend
    st.markdown('<div class="section-title">🏷️ FAULT TYPES</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-grid">', unsafe_allow_html=True)
    legend_data = [
        ("kp", "Key Point (Severe Damage)", "#ffff00"),
        ("hp_cm", "Corroded/Material Loss", "#ff00ff"),
        ("hp_cd", "Chipped/Damaged Tooth", "#ff0080")
    ]
    for code, desc, color in legend_data:
        st.markdown(f'''
        <div class="legend-item">
            <div class="legend-color" style="background-color:{color};"></div>
            <span class="legend-text">{code}:</span>
            <span class="legend-desc">{desc}</span>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detection
    if uploaded_file and detect_button:
        with st.spinner("Analyzing..."):
            time.sleep(1.5)
            result_img, detected_labels = predict_and_overlay(img)
        
        st.markdown('<div class="image-card"><div class="image-title">🛠 DETECTION RESULTS</div>', unsafe_allow_html=True)
        st.image(result_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Histogram
        if detected_labels:
            from collections import Counter
            counts = Counter(detected_labels)
            fig, ax = plt.subplots(figsize=(8,5))
            fig.patch.set_facecolor('#0a0a0a')
            ax.set_facecolor('#0a0a0a')
            for label, count in counts.items():
                ax.bar(label, count, color=bgr_to_rgb(class_colors[label]), edgecolor="#333", linewidth=2, alpha=0.9)
            ax.set_title("FAULT DISTRIBUTION", color="#00ff88")
            ax.set_xlabel("FAULT TYPE", color="#fff")
            ax.set_ylabel("FREQUENCY", color="#fff")
            ax.grid(axis='y', alpha=0.2, color='#333', linestyle='--')
            ax.tick_params(colors='#fff')
            st.pyplot(fig)
        
        # Download
        save_path = "detection_result.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        with open(save_path, "rb") as f:
            st.download_button("⬇️ DOWNLOAD RESULT", f, file_name="gear_fault_detection_result.jpg", mime="image/jpeg", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown('''
<div style="text-align:center; margin-top:2rem; padding:1rem; color:#555; border-top:1px solid #333;">
    🚀 Powered by YOLO & Streamlit | Advanced Gear Fault Detection System
</div>
''', unsafe_allow_html=True)
