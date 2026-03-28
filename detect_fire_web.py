import cv2
import datetime
import pandas as pd
import streamlit as st
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
import csv

# --- Configuration & Constants ---
MODEL_PATH = 'runs/detect/train8/weights/best.pt'
LOG_FILE = 'detection_logs.csv'

# --- Logging Functions ---
def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Class', 'Confidence'])

def log_detection(cls_name, conf):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cls_name, f"{conf:.2f}"])

# --- Streamlit Setup ---
st.set_page_config(page_title="Forest Fire System", page_icon="🔥", layout="wide")

# CSS Styling
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; font-weight: bold; }
.metric-card { background-color: #2e2e2e; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
.metric-value { font-size: 32px; font-weight: bold; color: #ff4b4b; margin-bottom: 5px; }
.metric-label { font-size: 16px; color: #aaaaaa; }
.main-header { display: flex; align-items: center; gap: 15px; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1 style="margin:0;">🔥 Automated Forest Fire Detection Hub</h1></div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model from `{MODEL_PATH}`. Ensure the model file exists.")
    st.stop()

init_log_file()

# --- Shared Processing Function ---
def process_frame(frame, conf_threshold):
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()
    
    fire_detected = False
    
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id] if model.names else str(cls_id)
        
        if conf > 0.6: 
            fire_detected = True
            log_detection(cls_name, conf)
            break # Only process highest confidence per frame for logging
            
    return annotated_frame, fire_detected

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard Overview", "🎥 Live Camera", "📹 Video Scanner", "🖼️ Image Scanner"])

# ----------------- TAB 1: DASHBOARD OVERVIEW -----------------
with tab1:
    st.markdown("### System Security Status")
    
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
    else:
        df = pd.DataFrame(columns=['Timestamp', 'Class', 'Confidence'])
        
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Detections</div>
        </div>
        """, unsafe_allow_html=True)
        
    with colB:
        last_alert = df['Timestamp'].iloc[-1] if not df.empty else "None"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ffa500; font-size: 24px;">{last_alert}</div>
            <div class="metric-label">Last Registration</div>
        </div>
        """, unsafe_allow_html=True)
        
    with colC:
        status_color = "#00ff00" if df.empty or (datetime.datetime.now() - pd.to_datetime(df['Timestamp'].iloc[-1])).total_seconds() > 3600 else "#ff4b4b"
        status_text = "SAFE" if status_color == "#00ff00" else "ALERT ACTIVE"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {status_color};">{status_text}</div>
            <div class="metric-label">Real-Time Status</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    if not df.empty:
        chart_col, table_col = st.columns([2, 1])
        with chart_col:
            st.markdown("**Detection Timeline**")
            df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
            daily_counts = df['Date'].value_counts().sort_index()
            st.bar_chart(daily_counts, use_container_width=True)
        with table_col:
            st.markdown("**Recent Logs**")
            st.dataframe(df.tail(15).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.success("The system is monitoring safely. No hazards have been recorded.")

# ----------------- TAB 2: LIVE CAMERA -----------------
with tab2:
    st.markdown("### Live Browser Inference")
    st.markdown("Use this tab to stream your local camera directly into the secure YOLO model.")
    
    conf_thresh_live = st.slider("Confidence Threshold (Live)", 0.0, 1.0, 0.50, 0.05)
    
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
    import av

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated_frame, _ = process_frame(img, conf_thresh_live)
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="fire-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False}
    )

# ----------------- TAB 3: VIDEO SCANNER -----------------
with tab3:
    st.markdown("### Upload & Scan Video")
    conf_thresh_vid = st.slider("Confidence Threshold (Video)", 0.0, 1.0, 0.50, 0.05)
    
    uploaded_video = st.file_uploader("Select an MP4, AVI, or MOV file", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()
        
        split1, split2 = st.columns(2)
        with split1:
            st.markdown("**Original Media**")
            st.video(tfile.name)
            process_btn = st.button("Start Analysis", type="primary", use_container_width=True)
            
        with split2:
            st.markdown("**AI Assessment**")
            stframe = st.empty()
        
        if process_btn:
            vid_cap = cv2.VideoCapture(tfile.name)
            total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            frame_count = 0
            
            while vid_cap.isOpened():
                ret, frame = vid_cap.read()
                if not ret: break
                
                annotated_frame, _ = process_frame(frame, conf_thresh_vid)
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame, channels="RGB", use_container_width=True)
                
                frame_count += 1
                if total_frames > 0: progress_bar.progress(min(frame_count / total_frames, 1.0))
                
            vid_cap.release()
            os.remove(tfile.name)
            st.success("File processed successfully.")

# ----------------- TAB 4: IMAGE SCANNER -----------------
with tab4:
    st.markdown("### Static Image Analysis")
    conf_thresh_img = st.slider("Confidence Threshold (Image)", 0.0, 1.0, 0.50, 0.05)
    
    uploaded_file = st.file_uploader("Select an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(image, caption="Original Image", use_container_width=True)
            analyze_btn = st.button("Scan for Fire", type="primary", use_container_width=True)
            
        if analyze_btn:
            with st.spinner('Model is evaluating...'):
                import numpy as np
                open_cv_image = np.array(image.convert('RGB')) 
                open_cv_image = open_cv_image[:, :, ::-1].copy() 
                
                annotated_img, fire_detected = process_frame(open_cv_image, conf_thresh_img)
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                out_image = Image.fromarray(annotated_img)
                
            with img_col2:
                st.image(out_image, caption="YOLO Result", use_container_width=True)
                if fire_detected:
                    st.error("🚨 Hazard warning! Fire/Smoke detected.")
                else:
                    st.success("✅ Clear.")
