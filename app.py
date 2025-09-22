import streamlit as st
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO 
from collections import Counter

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # make sure best.pt is in same folder

model = load_model()

# Custom CSS for complete black theme and grid layout
def load_custom_css():
    st.markdown("""
    <style>
    /* Global black theme */
    .stApp {
        background: #000000;
        color: #ffffff;
    }
    
    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Main grid container */
    .grid-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-gap: 1.5rem;
        min-height: 100vh;
        background: #000000;
        padding: 1rem;
    }
    
    /* Header section spanning full width */
    .header-section {
        grid-column: 1 / -1;
        text-align: center;
        background: linear-gradient(45deg, #1a1a1a, #2d2d2d);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 1rem;
        border: 1px solid #333;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .main-title {
        background: linear-gradient(45deg, #00ff88, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .subtitle {
        color: #888;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Left panel - Upload and controls */
    .left-panel {
        background: linear-gradient(145deg, #111111, #1a1a1a);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #333;
        height: fit-content;
        animation: slideInLeft 0.8s ease-out;
    }
    
    /* Right panel - Results */
    .right-panel {
        background: linear-gradient(145deg, #111111, #1a1a1a);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #333;
        height: fit-content;
        animation: slideInRight 0.8s ease-out;
    }
    
    /* Section headers */
    .section-title {
        color: #00ff88;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #333;
    }
    
    /* Upload area */
    .upload-area {
        background: #0a0a0a;
        border: 2px dashed #333;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #00ff88;
        background: #0f0f0f;
    }
    
    /* Instruction card */
    .instruction-card {
        background: linear-gradient(145deg, #0a0a0a, #111111);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid #333;
        transition: all 0.3s ease;
    }
    
    .instruction-card:hover {
        border-color: #00ff88;
        box-shadow: 0 5px 20px rgba(0, 255, 136, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #00ff88, #00ccff);
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.5);
        background: linear-gradient(45deg, #00ccff, #00ff88);
    }
    
    /* Image containers */
    .image-card {
        background: #0a0a0a;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #333;
        transition: all 0.3s ease;
    }
    
    .image-card:hover {
        border-color: #00ff88;
        box-shadow: 0 5px 20px rgba(0, 255, 136, 0.1);
    }
    
    .image-title {
        color: #00ccff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Legend styling */
    .legend-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.6rem;
        background: #0a0a0a;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0 2rem 0;
        border: 1px solid #333;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        padding: 0.6rem;
        background: #111;
        border-radius: 6px;
        border-left: 3px solid #00ff88;
        transition: all 0.2s ease;
    }
    
    .legend-item:hover {
        background: #1a1a1a;
        transform: translateX(3px);
    }
    
    .legend-color {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        margin-right: 0.8rem;
        border: 2px solid #333;
        flex-shrink: 0;
    }
    
    .legend-text {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.9rem;
        min-width: 60px;
    }
    
    .legend-desc {
        color: #bbb;
        margin-left: 0.3rem;
        font-size: 0.85rem;
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: linear-gradient(45deg, #0a0a0a, #111111);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #333;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        border-color: #00ff88;
        transform: translateY(-2px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #00ff88;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #888;
        font-size: 0.9rem;
    }
    
    /* Chart container */
    .chart-container {
        background: #0a0a0a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    
    /* Download section */
    .download-card {
        background: linear-gradient(45deg, #001a00, #002600);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #00ff88;
        margin: 1rem 0;
    }
    
    /* Success message */
    .success-banner {
        background: linear-gradient(45deg, #00ff88, #00ccff);
        color: #000000;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0 1.5rem 0;
        animation: slideInDown 0.5s ease-out;
    }
    
    /* Loading spinner */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1.5rem;
        color: #00ff88;
        background: #0a0a0a;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 1rem 0;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 3px solid #333;
        border-top: 3px solid #00ff88;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    .loading-text {
        margin-top: 1rem;
        color: #00ff88;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: #0a0a0a;
        border: 2px dashed #333;
        border-radius: 10px;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #00ff88;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Remove white bars and scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #111;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .grid-container {
            grid-template-columns: 1fr;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Class colors (BGR for OpenCV, but we need HEX/RGB for Matplotlib)
class_colors = {
    "kp": (0, 255, 255),     # Yellow
    "hp_cm": (255, 0, 255),  # Magenta
    "hp_cd": (128, 0, 255)   # Pinkish
}

# Convert OpenCV BGR to Matplotlib RGB
def bgr_to_rgb(color_bgr):
    return (color_bgr[2] / 255.0, color_bgr[1] / 255.0, color_bgr[0] / 255.0)

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


# ---------------- Streamlit UI ----------------
st.set_page_config(
    page_title="⚙️ Gear Fault Detection", 
    page_icon="🛠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
load_custom_css()

# Header section
st.markdown('''
<div class="header-section">
    <h1 class="main-title">⚙️ GEAR FAULT DETECTION</h1>
    <p class="subtitle">Advanced AI-powered gear fault detection and analysis system</p>
</div>
''', unsafe_allow_html=True)

# Create main grid layout
col_left, col_right = st.columns([1, 1], gap="large")

# Left panel - Upload and controls
with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="section-title">📂 UPLOAD IMAGE</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if not uploaded_file:
        st.markdown('<p style="color: #888; text-align: center;">Drag and drop or click to upload<br>Supported: JPG, JPEG, PNG</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show uploaded image if available
    if uploaded_file:
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown('<div class="image-title">📸 UPLOADED IMAGE</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detection button
        detect_button = st.button("🔍 START DETECTION", key="detect_btn")
    
    # Instructions for users when no image is uploaded
    else:
        st.markdown('''
            <div class="instruction-card">
                <div style="text-align: center; padding: 2rem; color: #888;">
                    <h3 style="color: #00ccff; margin-bottom: 1rem;">📸 UPLOAD YOUR GEAR IMAGE</h3>
                    <p style="margin-bottom: 0.5rem;">• Select a clear image of the gear</p>
                    <p style="margin-bottom: 0.5rem;">• Supported formats: JPG, JPEG, PNG</p>
                    <p>• Our AI will detect fault types automatically</p>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right panel - Results
with col_right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    
    # Always show the fault types legend on the right side
    st.markdown('<div class="section-title">🏷️ FAULT TYPES</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend-grid">', unsafe_allow_html=True)
    
    legend_data = [
        ("kp", "Key Point (Severe Damage)", "#ffff00"),
        ("hp_cm", "Corroded/Material Loss", "#ff00ff"), 
        ("hp_cd", "Chipped/Damaged Tooth", "#ff0080")
    ]
    
    for code, description, color in legend_data:
        st.markdown(f'''
            <div class="legend-item">
                <div class="legend-color" style="background-color: {color};"></div>
                <span class="legend-text">{code}:</span>
                <span class="legend-desc">{description}</span>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file and detect_button:
        # Show loading animation during processing
        loading_placeholder = st.empty()
        with loading_placeholder:
            st.markdown('''
                <div class="loading-container">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">🔄 ANALYZING GEAR IMAGE...</div>
                </div>
            ''', unsafe_allow_html=True)
        
        # Processing with spinner
        with st.spinner(""):
            time.sleep(1.5)  # Reduced time for faster response
            result_img, detected_labels = predict_and_overlay(img)
        
        # Clear the loading animation
        loading_placeholder.empty()

        # Brief success message that fades quickly
        success_placeholder = st.empty()
        with success_placeholder:
            st.markdown('<div class="success-banner">✅ DETECTION COMPLETED</div>', unsafe_allow_html=True)
        
        # Clear success message after a brief moment for cleaner UI
        time.sleep(1)
        success_placeholder.empty()
        
        # Statistics cards
        st.markdown('<div class="section-title">📊 DETECTION STATISTICS</div>', unsafe_allow_html=True)
        st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
        
        total_faults = len(detected_labels)
        unique_types = len(set(detected_labels)) if detected_labels else 0
        
        st.markdown(f'''
            <div class="stat-card">
                <div class="stat-number">{total_faults}</div>
                <div class="stat-label">TOTAL FAULTS</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{unique_types}</div>
                <div class="stat-label">FAULT TYPES</div>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detection result image
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown('<div class="image-title">🛠 DETECTION RESULTS</div>', unsafe_allow_html=True)
        st.image(result_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Histogram chart
        if detected_labels:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 FAULT DISTRIBUTION</div>', unsafe_allow_html=True)
            
            counts = Counter(detected_labels)
            
            # Create black-themed histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0a0a0a')
            ax.set_facecolor('#0a0a0a')
            
            bars = []
            for label, count in counts.items():
                color_rgb = bgr_to_rgb(class_colors[label])
                bar = ax.bar(label, count, color=color_rgb, edgecolor="#333", linewidth=2, alpha=0.9)
                bars.append(bar)
                
                # Add value labels on bars
                ax.text(label, count + 0.05, str(count), ha='center', va='bottom', 
                       fontweight='bold', fontsize=12, color='#ffffff')
            
            # Style the chart for black theme
            ax.set_title("FAULT DISTRIBUTION ANALYSIS", fontsize=14, fontweight='bold', 
                        color='#00ff88', pad=15)
            ax.set_xlabel("FAULT TYPE", fontsize=11, fontweight='600', color='#ffffff')
            ax.set_ylabel("FREQUENCY", fontsize=11, fontweight='600', color='#ffffff')
            
            # Grid and styling
            ax.grid(axis='y', alpha=0.2, color='#333', linestyle='--')
            ax.tick_params(colors='#ffffff')
            ax.spines['bottom'].set_color('#333')
            ax.spines['top'].set_color('#333')
            ax.spines['right'].set_color('#333')
            ax.spines['left'].set_color('#333')
            
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Download section
        st.markdown('<div class="download-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💾 SAVE RESULTS</div>', unsafe_allow_html=True)
        
        # Save the result
        save_path = "detection_result.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        with open(save_path, "rb") as f:
            st.download_button(
                "⬇️ DOWNLOAD DETECTION RESULT", 
                f, 
                file_name="gear_fault_detection_result.jpg", 
                mime="image/jpeg",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif uploaded_file and not detect_button:
        st.markdown('''
            <div style="text-align: center; padding: 2rem; color: #888;">
                <h3 style="color: #00ccff; margin-bottom: 1rem;">🔍 READY FOR ANALYSIS</h3>
                <p style="margin-bottom: 0.5rem;">Image uploaded successfully!</p>
                <p style="font-size: 0.9rem;">Click "START DETECTION" to analyze faults</p>
            </div>
        ''', unsafe_allow_html=True)
    
    else:
        st.markdown('''
            <div style="text-align: center; padding: 2rem; color: #888;">
                <h3 style="color: #00ccff; margin-bottom: 1rem;">� DETECTION READY</h3>
                <p style="margin-bottom: 0.5rem;">Upload an image to start analysis</p>
                <p style="font-size: 0.9rem;">Results will appear here after detection</p>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('''
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #555; border-top: 1px solid #333;">
        <p>🚀 Powered by YOLO & Streamlit | Advanced Gear Fault Detection System</p>
    </div>
''', unsafe_allow_html=True)
