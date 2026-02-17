#!/usr/bin/env python3
"""
Quick launcher for the Live Human Detection System
"""

import sys
import os
from live_human_detector import LiveHumanDetector

def main():
    print("🚀 Live Human Detection System")
    print("=" * 40)
    
    # Simple menu
    print("Choose detection mode:")
    print("1. Webcam (default)")
    print("2. Video file")
    print("3. Webcam with recording")
    
    choice = input("\nEnter choice (1-3) or press Enter for webcam: ").strip()
    
    detector = LiveHumanDetector(confidence_threshold=0.5)
    
    if choice == "2":
        # Video file mode
        video_path = input("Enter video file path: ").strip()
        if not os.path.exists(video_path):
            print(f"Error: File not found: {video_path}")
            return
        
        print(f"Processing video: {video_path}")
        detector.process_live_stream(source=video_path, display=True)
    
    elif choice == "3":
        # Webcam with recording
        output_file = f"recorded_detection_{int(__import__('time').time())}.mp4"
        print(f"Recording to: {output_file}")
        detector.process_live_stream(source=0, display=True, save_output=output_file)
    
    else:
        # Default webcam mode
        print("Starting webcam detection...")
        detector.process_live_stream(source=0, display=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)