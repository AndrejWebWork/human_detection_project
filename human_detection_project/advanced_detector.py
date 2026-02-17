import cv2
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
