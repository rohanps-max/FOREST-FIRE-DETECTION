import cv2
import csv
import datetime
import threading
import smtplib
from email.message import EmailMessage
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# Constants
MODEL_PATH = 'runs/detect/train8/weights/best.pt'
LOG_FILE = 'detection_logs.csv'

# Email Configuration (Update these with real values to use)
SENDER_EMAIL = "23bsds167rohanps@skacas.ac.in"
SENDER_PASSWORD = "WELCOME@123" # Use an App Password if using Gmail
RECEIVER_EMAIL = "psrohan122705@gmail.com"
EMAIL_COOLDOWN = 60 # seconds between emails

# Initialize log file
def init_log_file():
    try:
        with open(LOG_FILE, mode='x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Class', 'Confidence'])
    except FileExistsError:
        pass

def log_detection(cls_name, conf):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cls_name, f"{conf:.2f}"])

def send_email_alert(frame, timestamp):
    if SENDER_EMAIL == "your_email@gmail.com":
        print("Warning: Email not sent. Please configure credentials in detect_fire_gui.py.")
        return
        
    try:
        msg = EmailMessage()
        msg['Subject'] = f"URGENT: FIRE DETECTED at {timestamp}"
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg.set_content(f"CRITICAL ALERT: The Forest Fire Detection system has identified a fire at {timestamp}.\n\nPlease find the attached snapshot.")
        
        # Attach the image
        _, buffer = cv2.imencode('.jpg', frame)
        msg.add_attachment(buffer.tobytes(), maintype='image', subtype='jpeg', filename='detection.jpg')
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        print("Alert email sent successfully!")
    except Exception as e:
        print(f"Failed to send email alert: {e}")

class FireDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("700x550")
        
        # Load Model
        try:
            self.model = YOLO(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model from {MODEL_PATH}\nEnsure you have trained the model first.")
            self.window.destroy()
            return
            
        self.vid = None
        self.is_running = False
        self.last_log_time = datetime.datetime.now()
        self.last_email_time = None
        
        # Create GUI elements
        self.btn_frame = tk.Frame(window)
        self.btn_frame.pack(side=tk.TOP, pady=10)
        
        self.btn_webcam = tk.Button(self.btn_frame, text="Start Webcam", width=15, command=self.start_webcam)
        self.btn_webcam.pack(side=tk.LEFT, padx=5)
        
        self.btn_video = tk.Button(self.btn_frame, text="Select Video/Image", width=15, command=self.select_file)
        self.btn_video.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(self.btn_frame, text="Stop", width=15, command=self.stop_stream)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        # Canvas for video display
        self.canvas_frame = tk.Frame(window, bg='black', width=640, height=480)
        self.canvas_frame.pack(padx=10, pady=10)
        self.canvas_frame.pack_propagate(False) # don't shrink
        
        self.canvas = tk.Canvas(self.canvas_frame, width=640, height=480, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True)
        
        init_log_file()
        
    def start_webcam(self):
        self.start_stream(0)
        
    def select_file(self):
        file_path = filedialog.askopenfilename(title="Select Video or Image", filetypes=[("Media files", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png")])
        if file_path:
            # Check if it's an image
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.process_image(file_path)
            else:
                self.start_stream(file_path)
                
    def process_image(self, path):
        self.stop_stream()
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Could not read image.")
            return
            
        frame = self.detect_and_draw(frame)
        self.display_frame(frame)

    def start_stream(self, source):
        self.stop_stream()
        self.vid = cv2.VideoCapture(source)
        if not self.vid.isOpened():
            messagebox.showerror("Error", "Could not open video source.")
            return
        self.is_running = True
        self.update_video()
        
    def stop_stream(self):
        self.is_running = False
        if self.vid:
            self.vid.release()
            self.vid = None
            
    def detect_and_draw(self, frame):
        # Optimization: Process at a lower resolution (320) to significantly speed up inference
        results = self.model.predict(source=frame, conf=0.5, verbose=False, imgsz=320)
        annotated_frame = results[0].plot()
        
        # Log detections (throttle to 1 log per second maximum)
        now = datetime.datetime.now()
        if (now - self.last_log_time).total_seconds() > 1.0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id] if self.model.names else str(cls_id)
                if conf > 0.6: # high confidence threshold for logging
                    log_detection(cls_name, conf)
                    self.last_log_time = now
                    
                    # Trigger Email Alert (with Cooldown)
                    if self.last_email_time is None or (now - self.last_email_time).total_seconds() > EMAIL_COOLDOWN:
                        threading.Thread(target=send_email_alert, args=(annotated_frame.copy(), now.strftime("%H:%M:%S"))).start()
                        self.last_email_time = now
                        
                    break # Only log the highest confidence per second to avoid spam
                
        return annotated_frame
        
    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize for canvas to maintain aspect ratio keeping max dimensions 640x480
        h, w, _ = frame.shape
        scale = min(640/w, 480/h)
        new_w, new_h = int(w*scale), int(h*scale)
        frame = cv2.resize(frame, (new_w, new_h))
            
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.delete("all")
        self.canvas.create_image(640//2, 480//2, image=self.photo, anchor=tk.CENTER)

    def update_video(self):
        if self.is_running and self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame = self.detect_and_draw(frame)
                self.display_frame(frame)
                self.window.after(30, self.update_video) # approx 30 fps
            else:
                self.stop_stream()
                
    def on_close(self):
        self.stop_stream()
        self.window.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = FireDetectionApp(root, "Real-Time Forest Fire Detection")
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
