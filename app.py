from flask import Flask, render_template, Response, jsonify, request, url_for
import cv2
import datetime
from ultralytics import YOLO
import csv
import os
import threading
import numpy as np
import base64
from werkzeug.utils import secure_filename
from alert import send_mobile_alert

app = Flask(__name__)

# Constants
MODEL_PATH = 'runs/detect/train8/weights/best.pt'
LOG_FILE = 'detection_logs.csv'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global state
global_conf = 0.80 # Increased default to avoid false positives
current_frame = None # For capturing
file_video_path = None # For uploading videos

# Initialize model
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}. Ensure weights exist.")
    model = None

# Initialize logs
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Class', 'Confidence'])

def log_detection(cls_name, conf):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cls_name, f"{conf:.2f}"])

def generate_frames():
    global current_frame
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if model:
                results = model.predict(source=frame, conf=global_conf, verbose=False)
                frame = results[0].plot()
                current_frame = frame.copy()
                
                # Check for high confidence detections
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    # High confidence threshold relative to user setting
                    if conf > global_conf:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id] if model.names else str(cls_id)
                        
                        def trigger_actions():
                            log_detection(cls_name, conf)
                            send_mobile_alert(f"🚨 URGENT: {cls_name} detected with {conf:.2f} confidence!")
                        
                        threading.Thread(target=trigger_actions).start()
                        break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    global global_conf
    data = request.json
    if 'conf' in data:
        global_conf = float(data['conf'])
        return jsonify({"status": "success", "conf": global_conf})
    return jsonify({"status": "error"}), 400

@app.route('/capture', methods=['POST'])
def capture():
    global current_frame
    if current_frame is not None:
        ret, buffer = cv2.imencode('.jpg', current_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"status": "success", "image": jpg_as_text})
    return jsonify({"status": "error", "message": "No frame available"}), 400

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file and model:
        # Read image
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Infer
        results = model.predict(source=img, conf=global_conf, verbose=False)
        annotated_img = results[0].plot()
        
        # Log if detected
        fire_detected = False
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > global_conf:
                fire_detected = True
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] if model.names else str(cls_id)
                log_detection(cls_name, conf)
                break
        
        # Encode back
        ret, buffer = cv2.imencode('.jpg', annotated_img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({"status": "success", "image": jpg_as_text, "detected": fire_detected})

@app.route('/get_logs')
def get_logs():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                logs.append(row)
    return jsonify(logs[-15:][::-1])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
