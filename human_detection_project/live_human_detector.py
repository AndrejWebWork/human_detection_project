import cv2
import numpy as np
import time
import threading
from collections import deque
import argparse

class LiveHumanDetector:
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.detection_history = deque(maxlen=10)
        
        # Person tracking
        self.person_id = 0
        self.tracked_persons = {}
        
    def detect_humans(self, frame):
        """Enhanced human detection using multiple methods"""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = []
        
        # Method 1: Face detection (most reliable)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Expand bounding box to include more of the person
            expanded_h = int(h * 2.5)  # Assume body is 2.5x face height
            expanded_y = max(0, y - int(h * 0.2))  # Start slightly above face
            expanded_h = min(height - expanded_y, expanded_h)
            
            detections.append({
                'box': (x, expanded_y, w, expanded_h),
                'confidence': 0.85,
                'type': 'face',
                'center': (x + w//2, expanded_y + expanded_h//2)
            })
        
        # Method 2: Full body detection (if no faces found)
        if len(detections) == 0:
            bodies = self.body_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100)
            )
            
            for (x, y, w, h) in bodies:
                detections.append({
                    'box': (x, y, w, h),
                    'confidence': 0.7,
                    'type': 'body',
                    'center': (x + w//2, y + h//2)
                })
        
        # Method 3: Motion-based detection (backup)
        if len(detections) == 0 and hasattr(self, 'background_subtractor'):
            fg_mask = self.background_subtractor.apply(frame)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Filter by reasonable human size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Human-like aspect ratio
                    if 1.5 < aspect_ratio < 4.0:
                        detections.append({
                            'box': (x, y, w, h),
                            'confidence': 0.6,
                            'type': 'motion',
                            'center': (x + w//2, y + h//2)
                        })
        
        # Remove overlapping detections
        detections = self._remove_overlaps(detections)
        
        # Track persons across frames
        detections = self._track_persons(detections)
        
        return detections
    
    def _remove_overlaps(self, detections):
        """Remove overlapping detections, keeping the one with higher confidence"""
        if len(detections) <= 1:
            return detections
        
        filtered = []
        for i, det1 in enumerate(detections):
            is_duplicate = False
            x1, y1, w1, h1 = det1['box']
            
            for j, det2 in enumerate(detections):
                if i != j:
                    x2, y2, w2, h2 = det2['box']
                    
                    # Calculate overlap
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    if overlap_area > 0.3 * min(area1, area2):  # 30% overlap threshold
                        if det2['confidence'] > det1['confidence']:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered.append(det1)
        
        return filtered
    
    def _track_persons(self, detections):
        """Simple person tracking across frames"""
        current_time = time.time()
        
        # Remove old tracked persons
        to_remove = []
        for person_id, person_data in self.tracked_persons.items():
            if current_time - person_data['last_seen'] > 2.0:  # 2 second timeout
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.tracked_persons[person_id]
        
        # Match detections to existing persons
        for detection in detections:
            best_match = None
            best_distance = float('inf')
            
            for person_id, person_data in self.tracked_persons.items():
                distance = np.sqrt(
                    (detection['center'][0] - person_data['center'][0])**2 +
                    (detection['center'][1] - person_data['center'][1])**2
                )
                
                if distance < 100 and distance < best_distance:  # 100 pixel threshold
                    best_match = person_id
                    best_distance = distance
            
            if best_match:
                # Update existing person
                self.tracked_persons[best_match]['center'] = detection['center']
                self.tracked_persons[best_match]['last_seen'] = current_time
                detection['person_id'] = best_match
            else:
                # New person
                self.person_id += 1
                self.tracked_persons[self.person_id] = {
                    'center': detection['center'],
                    'last_seen': current_time,
                    'first_seen': current_time
                }
                detection['person_id'] = self.person_id
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw enhanced bounding boxes and labels"""
        for detection in detections:
            x, y, w, h = detection['box']
            confidence = detection['confidence']
            detection_type = detection['type']
            person_id = detection.get('person_id', 0)
            
            # Color based on detection type
            if detection_type == 'face':
                color = (0, 255, 0)  # Green for face detection
            elif detection_type == 'body':
                color = (255, 165, 0)  # Orange for body detection
            else:
                color = (0, 165, 255)  # Blue for motion detection
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            center = detection['center']
            cv2.circle(frame, center, 5, color, -1)
            
            # Prepare label
            label = f"Person {person_id} ({detection_type}): {confidence:.2f}"
            
            # Calculate label size
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def draw_info_panel(self, frame, detections, fps):
        """Draw information panel on the frame"""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Info text
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Detections: {len(detections)}",
            f"Active Persons: {len(self.tracked_persons)}",
            f"Resolution: {width}x{height}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (15, 35 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection history graph
        if len(self.detection_history) > 1:
            points = []
            for i, count in enumerate(self.detection_history):
                x = 320 + i * 10
                y = 100 - count * 10
                points.append((x, max(20, y)))
            
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], (0, 255, 255), 2)
        
        return frame
    
    def process_live_stream(self, source=0, display=True, save_output=None):
        """Process live video stream from webcam or video file"""
        # Initialize video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Set camera properties for better performance
        if isinstance(source, int):  # Webcam
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize background subtractor for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Video writer setup
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        
        print("Starting live human detection...")
        print("Press 'q' to quit, 's' to save screenshot, 'r' to reset tracking")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Detect humans
                detections = self.detect_humans(frame)
                
                # Update detection history
                self.detection_history.append(len(detections))
                
                # Draw detections
                frame = self.draw_detections(frame, detections)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
                current_fps = np.mean(self.fps_counter)
                
                # Draw info panel
                frame = self.draw_info_panel(frame, detections, current_fps)
                
                # Save frame if recording
                if save_output:
                    out.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Live Human Detection', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_name = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_name, frame)
                        print(f"Screenshot saved: {screenshot_name}")
                    elif key == ord('r'):
                        # Reset tracking
                        self.tracked_persons.clear()
                        self.person_id = 0
                        print("Tracking reset")
                
                frame_count += 1
                
                # Print stats every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    print(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            # Cleanup
            cap.release()
            if save_output:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            # Final stats
            total_time = time.time() - start_time
            print(f"\nSession complete:")
            print(f"Total frames: {frame_count}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Average FPS: {frame_count/total_time:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Live Human Detection System')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (0.1-0.9)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display (for headless systems)')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number (webcam)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Initialize detector
    detector = LiveHumanDetector(confidence_threshold=args.confidence)
    
    # Start processing
    detector.process_live_stream(
        source=source,
        display=not args.no_display,
        save_output=args.output
    )

if __name__ == '__main__':
    main()