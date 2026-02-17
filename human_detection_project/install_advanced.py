#!/usr/bin/env python3
"""
Advanced AI Detection Libraries Installer
Installs the latest and most powerful detection libraries
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
        print(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def main():
    print("Installing Advanced AI Detection Libraries...")
    print("=" * 50)
    
    # Core packages
    packages = [
        "opencv-python>=4.9.0",
        "numpy>=1.24.0", 
        "flask>=3.0.0",
        "werkzeug>=3.0.0",
        "pillow>=10.0.0"
    ]
    
    # AI/ML packages
    ai_packages = [
        "mediapipe>=0.10.9",
        "ultralytics>=8.1.0", 
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "onnxruntime>=1.16.0"
    ]
    
    print("Installing core packages...")
    for package in packages:
        install_package(package)
    
    print("\nInstalling AI/ML packages...")
    for package in ai_packages:
        install_package(package)
    
    print("\nDownloading YOLO models...")
    try:
        from ultralytics import YOLO
        # Download YOLOv8 nano model (fastest)
        model = YOLO('yolov8n.pt')
        print("YOLOv8 nano model downloaded")
    except Exception as e:
        print(f"YOLO model download failed: {e}")
    
    print("\nInstallation complete!")
    print("Your system now has the most advanced detection capabilities!")

if __name__ == "__main__":
    main()