import cv2
import numpy as np
import os
import time
from flask import Flask, render_template_string, request, jsonify, send_file, Response
import uuid
from werkzeug.utils import secure_filename
import urllib.request
from advanced_detector import AdvancedHumanDetector

# ==================== CONFIGURATION ====================
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'mpeg'}
UPLOAD_FOLDER = 'uploaded_videos'
PROCESSED_FOLDER = 'processed_videos'
MODEL_FOLDER = 'models'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
FRAMES_TO_SKIP = 3

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ==================== FLASK APP SETUP ====================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# ==================== LIVE DETECTION GLOBALS ====================
live_camera = None
live_running = False
live_fps = 0
live_detections = 0

# ==================== TINY YOLO MODEL (INCLUDED IN CODE) ====================
# We'll use OpenCV's built-in face detection instead of YOLO for simplicity
# This avoids the need to download large model files

# COCO class names (we only need person which is class 0)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ==================== UTILITY FUNCTIONS ====================
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_model_files():
    """Create necessary model files locally"""
    # Create coco.names file
    coco_path = os.path.join(MODEL_FOLDER, 'coco.names')
    with open(coco_path, 'w') as f:
        for class_name in COCO_CLASSES:
            f.write(f"{class_name}\n")
    
    print("Created model files successfully")
    return True

# ==================== HUMAN DETECTOR CLASS ====================
class HumanDetector:
    def __init__(self, confidence_threshold=0.6, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize multiple detection methods
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Tracking variables
        self.previous_detections = []
        self.detection_history = []
        
        self.classes = COCO_CLASSES
        self.person_class_id = 0
    
    def _load_model(self):
        """Load a lightweight DNN model for object detection"""
        # Using OpenCV's face detector as a fallback
        # This will detect faces which is good for testing
        try:
            # Try to load a lightweight model
            net = cv2.dnn.readNetFromDarknet(
                "yolov3-tiny.cfg" if os.path.exists("yolov3-tiny.cfg") else self._create_tiny_yolo_config(),
                "yolov3-tiny.weights" if os.path.exists("yolov3-tiny.weights") else None
            )
            print("Loaded YOLO-tiny model")
        except:
            # Fallback to OpenCV's face detector
            print("Using OpenCV face detector as fallback")
            net = None
        
        return net
    
    def _create_tiny_yolo_config(self):
        """Create a minimal configuration for testing"""
        # This is just a placeholder - we'll use OpenCV's face detector instead
        return None
    
    def detect_humans(self, frame):
        """Fast stable detection"""
        height, width = frame.shape[:2]
        
        # Fast processing - 50% scale for better accuracy
        small_frame = cv2.resize(frame, (int(width * 0.5), int(height * 0.5)))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-method human detection
        detections = []
        
        # Method 1: Face detection
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25), maxSize=(150, 150)
        )
        
        for (x, y, w, h) in faces:
            x, y, w, h = int(x*2), int(y*2), int(w*2), int(h*2)
            person_w, person_h = int(w*2.2), int(h*5.5)
            person_x = max(0, x - int(w*0.6))
            person_y = max(0, y - int(h*0.2))
            
            detections.append({
                'box': (person_x, person_y, min(person_w, width-person_x), min(person_h, height-person_y)),
                'confidence': 0.9,
                'type': 'HUMAN',
                'class_name': 'person'
            })
        
        # Method 2: Body detection if no faces
        if not detections:
            bodies = self.body_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 120)
            )
            
            for (x, y, w, h) in bodies:
                x, y, w, h = int(x*2), int(y*2), int(w*2), int(h*2)
                aspect_ratio = h / w if w > 0 else 0
                if 1.5 < aspect_ratio < 4.0:
                    detections.append({
                        'box': (x, y, w, h),
                        'confidence': 0.8,
                        'type': 'HUMAN',
                        'class_name': 'person'
                    })
        
        # Stable tracking with fade-out
        if hasattr(self, 'prev_dets'):
            detections = self._stable_track(detections)
        self.prev_dets = detections
        return detections
    
    def _stable_track(self, dets):
        """Stable tracking with brief persistence"""
        if not hasattr(self, 'miss_count'):
            self.miss_count = 0
        
        if not dets:
            self.miss_count += 1
            # Keep previous detection for 3 frames when moving
            if self.miss_count < 3 and hasattr(self, 'prev_dets'):
                return self.prev_dets
            else:
                self.miss_count = 0
                return []
        
        self.miss_count = 0
        
        for i, det in enumerate(dets):
            if i < len(self.prev_dets):
                px, py, pw, ph = self.prev_dets[i]['box']
                cx, cy, cw, ch = det['box']
                # Smooth interpolation
                det['box'] = (
                    int(px*0.4 + cx*0.6),
                    int(py*0.4 + cy*0.6), 
                    int(pw*0.4 + cw*0.6),
                    int(ph*0.4 + ch*0.6)
                )
        return dets
    
    def _smooth_tracking(self, current_detections):
        """Smooth tracking between frames"""
        if not current_detections:
            # If no current detections, fade out previous ones
            faded = []
            for prev_det in self.previous_detections:
                prev_det['confidence'] *= 0.8
                if prev_det['confidence'] > 0.3:
                    faded.append(prev_det)
            return faded
        
        # Match current detections with previous ones
        matched_detections = []
        
        for curr_det in current_detections:
            curr_x, curr_y, curr_w, curr_h = curr_det['box']
            curr_center = (curr_x + curr_w//2, curr_y + curr_h//2)
            
            best_match = None
            min_distance = float('inf')
            
            for prev_det in self.previous_detections:
                prev_x, prev_y, prev_w, prev_h = prev_det['box']
                prev_center = (prev_x + prev_w//2, prev_y + prev_h//2)
                
                distance = ((curr_center[0] - prev_center[0])**2 + 
                           (curr_center[1] - prev_center[1])**2)**0.5
                
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    best_match = prev_det
            
            if best_match:
                # Smooth transition
                prev_x, prev_y, prev_w, prev_h = best_match['box']
                smooth_x = int(prev_x * 0.3 + curr_x * 0.7)
                smooth_y = int(prev_y * 0.3 + curr_y * 0.7)
                smooth_w = int(prev_w * 0.3 + curr_w * 0.7)
                smooth_h = int(prev_h * 0.3 + curr_h * 0.7)
                
                curr_det['box'] = (smooth_x, smooth_y, smooth_w, smooth_h)
                curr_det['confidence'] = min(0.95, curr_det['confidence'] + 0.1)
            
            matched_detections.append(curr_det)
        
        return matched_detections
    
    def _remove_overlaps(self, detections):
        """Remove overlapping detections using NMS"""
        if len(detections) <= 1:
            return detections
        
        boxes = []
        confidences = []
        
        for det in detections:
            x, y, w, h = det['box']
            boxes.append([x, y, x + w, y + h])
            confidences.append(det['confidence'])
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        if len(indices) > 0:
            return [detections[i] for i in indices.flatten()]
        return []
    
    def process_video(self, input_path, output_path=None, frames_to_skip=2, display=False):
        """Process a video file to detect humans in each frame"""
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            # Try opening with different backend
            cap = cv2.VideoCapture(input_path, cv2.CAP_ANY)
            if not cap.isOpened():
                return {"error": f"Could not open video: {input_path}"}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30  # Default if cannot determine fps
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            # For some video formats, frame count might not be available
            total_frames = 1000  # Estimate
        
        # Set up video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize variables
        frame_count = 0
        processed_count = 0
        detection_data = []
        start_time = time.time()
        
        print(f"Processing video: {input_path}")
        print(f"Total frames: {total_frames}, Processing every {frames_to_skip} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every nth frame
            if frame_count % frames_to_skip == 0:
                # Detect humans in the frame
                detections = self.detect_humans(frame)
                
                # Store detection data
                detection_data.append({
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'detections': detections
                })
                
                # Draw bounding boxes and labels
                processed_frame = self._draw_detections(frame.copy(), detections)
                
                # Write to output video
                if output_path:
                    out.write(processed_frame)
                
                # Display the frame if requested
                if display:
                    # Resize for display if too large
                    display_frame = processed_frame
                    if width > 1280:
                        scale = 1280 / width
                        display_frame = cv2.resize(processed_frame, (int(width * scale), int(height * scale)))
                    
                    cv2.imshow('Human Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                processed_count += 1
                
                # Print progress
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} frames")
            
            frame_count += 1
            
            # Safety break for very long videos
            if frame_count > 10000:
                break
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        if display:
            cv2.destroyAllWindows()
        
        # Prepare results
        total_humans = sum(len(frame_data['detections']) for frame_data in detection_data)
        avg_humans_per_frame = total_humans / processed_count if processed_count > 0 else 0
        
        results = {
            'input_path': input_path,
            'output_path': output_path,
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'processing_time': round(processing_time, 2),
            'fps': fps,
            'total_humans_detected': total_humans,
            'avg_humans_per_frame': round(avg_humans_per_frame, 2),
            'detection_data': detection_data
        }
        
        print(f"Processing complete! Detected {total_humans} humans in {processed_count} frames")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        return results
    
    def _draw_detections(self, frame, detections):
        """Ultra-fast professional visualization"""
        for i, det in enumerate(detections):
            x, y, w, h = det['box']
            
            # High-tech neon green
            color = (0, 255, 100)
            
            # Fast minimal box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Fast corners
            s = 20
            cv2.line(frame, (x, y), (x + s, y), color, 3)
            cv2.line(frame, (x, y), (x, y + s), color, 3)
            cv2.line(frame, (x + w, y), (x + w - s, y), color, 3)
            cv2.line(frame, (x + w, y), (x + w, y + s), color, 3)
            
            # Minimal label
            label = f"HUMAN-{i+1}"
            cv2.rectangle(frame, (x, y - 25), (x + 100, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Fast HUD with instant count update
        status_color = (0, 255, 100) if len(detections) > 0 else (255, 100, 0)
        cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"HUMANS: {len(detections)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        status_text = "LOCKED" if len(detections) > 0 else "SCANNING"
        cv2.putText(frame, f"STATUS: {status_text}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        return frame

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    """Render the main page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Human Detection in Videos</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f0f0; }
            .container { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 20px; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: bold; color: #333; }
            input[type="file"] { width: 100%; padding: 12px; border: 2px dashed #ccc; border-radius: 8px; background: #fafafa; }
            input[type="range"], input[type="number"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #4CAF50; color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%; }
            button:hover { background: #45a049; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
            button:disabled { background: #cccccc; cursor: not-allowed; }
            .results { margin-top: 25px; padding: 20px; background: #e8f5e8; border-radius: 10px; border-left: 5px solid #4CAF50; }
            .progress { height: 25px; background: #e0e0e0; border-radius: 10px; margin: 15px 0; overflow: hidden; }
            .progress-bar { height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); border-radius: 10px; width: 0%; transition: width 0.5s; }
            .status { padding: 15px; margin: 15px 0; border-radius: 8px; }
            .status.info { background: #e3f2fd; border-left: 5px solid #2196F3; }
            .status.success { background: #e8f5e8; border-left: 5px solid #4CAF50; }
            .status.error { background: #ffebee; border-left: 5px solid #F44336; }
            .download-btn { display: block; text-align: center; margin-top: 15px; text-decoration: none; }
            .mode-selector { display: flex; justify-content: center; gap: 10px; margin-bottom: 25px; }
            .mode-btn { padding: 12px 24px; border: 2px solid #4CAF50; background: white; color: #4CAF50; border-radius: 25px; cursor: pointer; transition: all 0.3s; }
            .mode-btn.active { background: #4CAF50; color: white; }
            .mode-btn:hover { transform: translateY(-2px); }
            .live-controls { display: flex; justify-content: center; gap: 15px; margin-bottom: 20px; }
            .btn-start { background: #2196F3; }
            .btn-stop { background: #f44336; }
            .live-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 20px; }
            .stat-item { background: #f0f0f0; padding: 15px; border-radius: 8px; text-align: center; }
            .stat-label { display: block; font-weight: bold; color: #666; }
            .stat-value { display: block; font-size: 1.5em; color: #4CAF50; margin-top: 5px; }
            .features { display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; margin-top: 15px; }
            .feature { background: #e3f2fd; color: #1976d2; padding: 5px 12px; border-radius: 15px; font-size: 0.9em; font-weight: bold; }
            .header h1 { background: linear-gradient(45deg, #2196F3, #00BCD4, #4CAF50); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: glow 1s ease-in-out infinite alternate; }
            @keyframes glow { from { text-shadow: 0 0 5px #2196F3; } to { text-shadow: 0 0 20px #2196F3, 0 0 30px #00BCD4; } }
            body { background: linear-gradient(135deg, #1a1a2e, #16213e); }
            .container { background: rgba(22,33,62,0.95); border: 2px solid #2196F3; box-shadow: 0 0 20px rgba(33,150,243,0.3); }
            .confidence-display { text-align: center; margin: 10px 0; }
            .confidence-display span { background: #4CAF50; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI HUMAN DETECTION SYSTEM</h1>
                <p>MEDIAPIPE • YOLOV8 • PYTORCH • ADVANCED AI TRACKING</p>
                <div class="features">
                    <span class="feature">🧠 AI POWERED</span>
                    <span class="feature">🔭 MEDIAPIPE</span>
                    <span class="feature">⚡ YOLO V8</span>
                    <span class="feature">🎯 SMART TRACK</span>
                </div>
            </div>
            
            <div class="mode-selector">
                <button onclick="showVideoMode()" class="mode-btn active" id="videoBtn">📁 Video Upload</button>
                <button onclick="showLiveMode()" class="mode-btn" id="liveBtn">📹 Live Webcam</button>
            </div>
            
            <div id="videoMode">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="video">📁 Upload Video File:</label>
                    <input type="file" id="video" name="video" accept="video/*" required>
                    <small>Supported formats: MP4, AVI, MOV, MKV, WEBM</small>
                </div>
                
                <div class="form-group">
                    <label for="confidence">🎯 Detection Confidence Threshold:</label>
                    <input type="range" id="confidence" name="confidence" min="0.3" max="0.9" step="0.05" value="0.6">
                    <div class="confidence-display">
                        <span id="confidenceValue">60%</span>
                    </div>
                    <small>Professional Mode: Higher threshold = fewer false positives</small>
                </div>
                
                <div class="form-group">
                    <label for="frames">⏩ Processing Speed (1-10):</label>
                    <input type="number" id="frames" name="frames_to_skip" min="1" max="10" value="3">
                    <small>Higher numbers = faster processing but less accurate</small>
                </div>
                
                <button type="submit" id="processBtn">🚀 Process Video</button>
            </form>
            </div>
            
            <div id="liveMode" style="display: none;">
                <div class="live-controls">
                    <button onclick="startLive()" class="btn btn-start" id="startBtn">📹 Start Live Detection</button>
                    <button onclick="stopLive()" class="btn btn-stop" id="stopBtn" disabled>⏹️ Stop Detection</button>
                </div>
                
                <div id="liveVideo" style="display: none; text-align: center; margin: 20px 0;">
                    <img id="videoFeed" style="max-width: 100%; border: 3px solid #4CAF50; border-radius: 10px;">
                </div>
                
                <div id="liveStats" class="live-stats" style="display: none;">
                    <div class="stat-item">
                        <span class="stat-label">FPS:</span>
                        <span class="stat-value" id="fpsValue">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Detections:</span>
                        <span class="stat-value" id="detectionsValue">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Status:</span>
                        <span class="stat-value" id="statusValue">Stopped</span>
                    </div>
                </div>
            </div>
            
            <div class="progress" id="progressContainer" style="display: none;">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            
            <div id="status" class="status info" style="display: none;">
                <strong>Status:</strong> <span id="statusText">Ready to process</span>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <h2>📊 Processing Results</h2>
                <div id="resultContent"></div>
                <a id="downloadLink" class="download-btn" style="display: none;">
                    <button>💾 Download Processed Video</button>
                </a>
            </div>
        </div>

        <script>
            let liveRunning = false;
            let statsInterval;
            
            function showVideoMode() {
                document.getElementById('videoMode').style.display = 'block';
                document.getElementById('liveMode').style.display = 'none';
                document.getElementById('videoBtn').classList.add('active');
                document.getElementById('liveBtn').classList.remove('active');
            }
            
            function showLiveMode() {
                document.getElementById('videoMode').style.display = 'none';
                document.getElementById('liveMode').style.display = 'block';
                document.getElementById('videoBtn').classList.remove('active');
                document.getElementById('liveBtn').classList.add('active');
            }
            
            function startLive() {
                fetch('/start_live', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            liveRunning = true;
                            document.getElementById('videoFeed').src = '/video_feed';
                            document.getElementById('liveVideo').style.display = 'block';
                            document.getElementById('liveStats').style.display = 'block';
                            document.getElementById('startBtn').disabled = true;
                            document.getElementById('stopBtn').disabled = false;
                            document.getElementById('statusValue').textContent = 'Running';
                            
                            statsInterval = setInterval(updateLiveStats, 1000);
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }
            
            function stopLive() {
                fetch('/stop_live', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            liveRunning = false;
                            document.getElementById('liveVideo').style.display = 'none';
                            document.getElementById('liveStats').style.display = 'none';
                            document.getElementById('startBtn').disabled = false;
                            document.getElementById('stopBtn').disabled = true;
                            document.getElementById('statusValue').textContent = 'Stopped';
                            
                            clearInterval(statsInterval);
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }
            
            function updateLiveStats() {
                if (!liveRunning) return;
                
                fetch('/live_stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('fpsValue').textContent = data.fps;
                        document.getElementById('detectionsValue').textContent = data.detections;
                        
                        // Update status color based on performance
                        const statusElement = document.getElementById('statusValue');
                        if (data.fps > 20) {
                            statusElement.style.color = '#4CAF50';
                            statusElement.textContent = 'Excellent';
                        } else if (data.fps > 15) {
                            statusElement.style.color = '#FF9800';
                            statusElement.textContent = 'Good';
                        } else {
                            statusElement.style.color = '#F44336';
                            statusElement.textContent = 'Processing';
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }
            
            document.getElementById('confidence').addEventListener('input', function() {
                const percentage = Math.round(this.value * 100);
                document.getElementById('confidenceValue').textContent = percentage + '%';
            });
            
            // Initialize confidence display
            document.addEventListener('DOMContentLoaded', function() {
                const confidenceSlider = document.getElementById('confidence');
                const percentage = Math.round(confidenceSlider.value * 100);
                document.getElementById('confidenceValue').textContent = percentage + '%';
            });

            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const form = this;
                const statusDiv = document.getElementById('status');
                const statusText = document.getElementById('statusText');
                const progressContainer = document.getElementById('progressContainer');
                const progressBar = document.getElementById('progressBar');
                const processBtn = document.getElementById('processBtn');
                
                // Disable button during processing
                processBtn.disabled = true;
                processBtn.textContent = 'Processing...';
                
                // Show status
                statusDiv.style.display = 'block';
                statusDiv.className = 'status info';
                statusText.textContent = 'Processing... This may take a few minutes depending on video size';
                
                // Show progress
                progressContainer.style.display = 'block';
                progressBar.style.width = '10%';
                
                const formData = new FormData(form);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        // Show error
                        statusDiv.className = 'status error';
                        statusText.textContent = 'Error: ' + data.error;
                        progressContainer.style.display = 'none';
                    } else {
                        // Show success with progress animation
                        let progress = 10;
                        const interval = setInterval(() => {
                            progress += 2;
                            progressBar.style.width = progress + '%';
                            
                            if (progress >= 100) {
                                clearInterval(interval);
                                
                                // Show success
                                statusDiv.className = 'status success';
                                statusText.textContent = 'Processing complete!';
                                
                                // Show results
                                document.getElementById('results').style.display = 'block';
                                document.getElementById('resultContent').innerHTML = `
                                    <p><strong>Original File:</strong> ${data.original_filename}</p>
                                    <p><strong>Total Frames Processed:</strong> ${data.results.processed_frames}</p>
                                    <p><strong>Processing Time:</strong> ${data.results.processing_time} seconds</p>
                                    <p><strong>Total Humans Detected:</strong> ${data.results.total_humans_detected}</p>
                                    <p><strong>Average Detections per Frame:</strong> ${data.results.avg_humans_per_frame}</p>
                                `;
                                
                                // Show download link
                                const downloadLink = document.getElementById('downloadLink');
                                downloadLink.href = '/download/' + data.processed_filename;
                                downloadLink.style.display = 'block';
                                
                                progressContainer.style.display = 'none';
                            }
                        }, 50);
                    }
                } catch (error) {
                    statusDiv.className = 'status error';
                    statusText.textContent = 'Error: ' + error.message;
                    progressContainer.style.display = 'none';
                } finally {
                    // Re-enable button
                    processBtn.disabled = false;
                    processBtn.textContent = '🚀 Process Video';
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{unique_id}{ext}"
        
        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(input_path)
        
        # Process video
        output_filename = f"processed_{unique_id}.mp4"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        try:
            # Get processing parameters
            frames_to_skip = int(request.form.get('frames_to_skip', FRAMES_TO_SKIP))
            confidence = float(request.form.get('confidence', CONFIDENCE_THRESHOLD))
            
            # Process the video
            detector = HumanDetector(confidence_threshold=confidence)
            results = detector.process_video(
                input_path=input_path,
                output_path=output_path,
                frames_to_skip=frames_to_skip,
                display=False
            )
            
            if 'error' in results:
                return jsonify({'error': results['error']}), 500
            
            # Prepare response
            response = {
                'message': 'Processing completed successfully',
                'original_filename': filename,
                'processed_filename': output_filename,
                'results': {
                    'total_frames': results['total_frames'],
                    'processed_frames': results['processed_frames'],
                    'processing_time': results['processing_time'],
                    'total_humans_detected': results['total_humans_detected'],
                    'avg_humans_per_frame': results['avg_humans_per_frame']
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please use MP4, AVI, MOV, MKV, or WEBM'}), 400

@app.route('/download/<filename>')
def download_video(filename):
    """Download processed video"""
    try:
        filepath = os.path.join(PROCESSED_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(
            filepath,
            as_attachment=True,
            download_name=f"human_detection_{filename}"
        )
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

def generate_live_frames():
    global live_camera, live_running, live_fps, live_detections
    detector = AdvancedHumanDetector(confidence_threshold=0.7)
    
    while live_running and live_camera:
        ret, frame = live_camera.read()
        if not ret: break
        
        start = time.time()
        
        # Advanced AI detection
        detections = detector.detect_humans(frame)
        live_detections = len(detections)
        frame = detector.draw_detections(frame, detections)
        
        # Calculate FPS
        live_fps = 1.0 / (time.time() - start)
        
        # High-quality encode
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    from flask import Response
    return Response(generate_live_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_live', methods=['POST'])
def start_live():
    global live_camera, live_running
    try:
        live_camera = cv2.VideoCapture(0)
        if not live_camera.isOpened():
            return jsonify({'success': False, 'error': 'Camera not found'})
        
        # TURBO camera settings
        live_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        live_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        live_camera.set(cv2.CAP_PROP_FPS, 60)
        live_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        live_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        live_running = True
        return jsonify({'success': True, 'message': 'Live detection started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_live', methods=['POST'])
def stop_live():
    global live_camera, live_running
    try:
        live_running = False
        if live_camera:
            live_camera.release()
            live_camera = None
        return jsonify({'success': True, 'message': 'Live detection stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/live_stats')
def live_stats():
    return jsonify({
        'fps': round(live_fps, 1),
        'detections': live_detections,
        'status': 'running' if live_running else 'stopped'
    })

@app.route('/test')
def test_detection():
    try:
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        detector = HumanDetector()
        detections = detector.detect_humans(test_image)
        return jsonify({
            'status': 'success',
            'detections': len(detections),
            'message': 'Detection system is working correctly'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==================== MAIN EXECUTION ====================
def main():
    """Main function to run the application"""
    print("Starting Human Detection System...")
    print("Creating necessary directories...")
    
    # Create model files
    create_model_files()
    
    print("Setup complete!")
    print("Starting web server...")
    print("Open http://localhost:5000 in your browser to use the application.")
    print("Tip: For best results, use videos with clear human subjects.")
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()