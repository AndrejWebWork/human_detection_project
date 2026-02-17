#!/usr/bin/env python3
"""
Human Detection System - Complete Setup and Launch Script
This script ensures all dependencies are installed and starts the web server
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_status(message, status="INFO"):
    """Print status messages (Windows compatible)"""
    try:
        print(f"{status}: {message}")
    except UnicodeEncodeError:
        # Fallback for Windows console encoding issues
        print(f"{status}: {message.encode('ascii', 'ignore').decode('ascii')}")

def check_python_version():
    """Check if Python version is compatible"""
    print_status("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_status(f"Python {version.major}.{version.minor} detected. Python 3.8+ required!", "ERROR")
        return False
    print_status(f"Python {version.major}.{version.minor}.{version.micro} - OK", "SUCCESS")
    return True

def install_requirements():
    """Install all required packages"""
    print_status("Installing required packages...")
    
    # Essential packages for the human detection system
    packages = [
        "opencv-python>=4.8.0",
        "numpy>=1.21.0", 
        "flask>=2.3.0",
        "werkzeug>=2.3.0",
        "pillow>=10.0.0"
    ]
    
    for package in packages:
        try:
            print_status(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--upgrade"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print_status(f"✓ {package} installed", "SUCCESS")
        except subprocess.CalledProcessError:
            print_status(f"Failed to install {package}", "ERROR")
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print_status("Creating project directories...")
    
    directories = [
        "uploaded_videos",
        "processed_videos", 
        "models",
        "screenshots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print_status(f"✓ Directory '{directory}' ready", "SUCCESS")

def create_advanced_detector():
    """Create the advanced detector module"""
    print_status("Creating advanced detector module...")
    
    advanced_detector_code = '''import cv2
import numpy as np
import time

class AdvancedHumanDetector:
    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        
        # Initialize detection cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
    def detect_humans(self, frame):
        """Advanced human detection using multiple methods"""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = []
        
        # Method 1: Face detection (most reliable)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Expand to full person
            person_h = int(h * 3.5)
            person_y = max(0, y - int(h * 0.3))
            person_h = min(height - person_y, person_h)
            
            detections.append({
                'box': (x, person_y, w, person_h),
                'confidence': 0.9,
                'type': 'face'
            })
        
        # Method 2: Body detection if no faces
        if not detections:
            bodies = self.body_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100)
            )
            
            for (x, y, w, h) in bodies:
                detections.append({
                    'box': (x, y, w, h),
                    'confidence': 0.8,
                    'type': 'body'
                })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels"""
        for i, detection in enumerate(detections):
            x, y, w, h = detection['box']
            confidence = detection['confidence']
            det_type = detection['type']
            
            # Color based on detection type
            color = (0, 255, 0) if det_type == 'face' else (255, 165, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"Human {i+1} ({det_type}): {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw info panel
        cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Detections: {len(detections)}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {'ACTIVE' if detections else 'SCANNING'}", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
'''
    
    with open("advanced_detector.py", "w") as f:
        f.write(advanced_detector_code)
    
    print_status("✓ Advanced detector module created", "SUCCESS")

def test_opencv():
    """Test if OpenCV is working properly"""
    print_status("Testing OpenCV installation...")
    
    try:
        import cv2
        print_status(f"✓ OpenCV {cv2.__version__} loaded successfully", "SUCCESS")
        
        # Test cascade classifiers
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print_status("Face cascade not loaded properly", "ERROR")
            return False
        
        print_status("✓ Face detection cascade loaded", "SUCCESS")
        return True
        
    except ImportError as e:
        print_status(f"OpenCV import failed: {e}", "ERROR")
        return False

def test_flask():
    """Test if Flask is working"""
    print_status("Testing Flask installation...")
    
    try:
        import flask
        print_status(f"✓ Flask {flask.__version__} loaded successfully", "SUCCESS")
        return True
    except ImportError as e:
        print_status(f"Flask import failed: {e}", "ERROR")
        return False

def start_web_server():
    """Start the Flask web server"""
    print_status("Starting Human Detection Web Server...")
    
    try:
        # Import and start the main application
        import human_detector
        
        print_status("=" * 60, "SUCCESS")
        print_status("HUMAN DETECTION SYSTEM READY!", "SUCCESS")
        print_status("=" * 60, "SUCCESS")
        print_status("Web interface: http://localhost:5000", "INFO")
        print_status("Press Ctrl+C to stop the server", "INFO")
        print_status("=" * 60, "SUCCESS")
        
        # Open browser automatically
        time.sleep(2)
        webbrowser.open("http://localhost:5000")
        
        # Start the Flask app
        human_detector.app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print_status("\nServer stopped by user", "INFO")
    except Exception as e:
        print_status(f"Failed to start server: {e}", "ERROR")
        return False
    
    return True

def main():
    """Main setup and launch function"""
    print_status("=" * 60, "INFO")
    print_status("HUMAN DETECTION SYSTEM - SETUP & LAUNCH", "INFO") 
    print_status("=" * 60, "INFO")
    
    # Step 1: Check Python version
    if not check_python_version():
        print_status("Please install Python 3.8 or higher", "ERROR")
        return False
    
    # Step 2: Install requirements
    if not install_requirements():
        print_status("Failed to install required packages", "ERROR")
        return False
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Create advanced detector
    create_advanced_detector()
    
    # Step 5: Test installations
    if not test_opencv():
        print_status("OpenCV test failed", "ERROR")
        return False
    
    if not test_flask():
        print_status("Flask test failed", "ERROR") 
        return False
    
    print_status("✓ All systems ready!", "SUCCESS")
    
    # Step 6: Start web server
    return start_web_server()

if __name__ == "__main__":
    success = main()
    if not success:
        print_status("Setup failed. Please check the errors above.", "ERROR")
        sys.exit(1)