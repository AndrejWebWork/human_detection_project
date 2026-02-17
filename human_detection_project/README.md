# 🎥 Live Human Detection System

An upgraded real-time human detection system that processes live video streams, detects and tracks people, and displays results with enhanced visualization.

## ✨ Features

### 🔥 **NEW UPGRADES:**
- **Real-time live video processing** from webcam or video files
- **Person tracking** across frames with unique IDs
- **Multi-method detection** (face, body, motion-based)
- **Performance monitoring** with FPS counter and detection history
- **Interactive controls** (save screenshots, reset tracking)
- **Enhanced visualization** with info panels and detection graphs
- **Overlap removal** to prevent duplicate detections
- **Configurable confidence thresholds**

### 📊 **Detection Methods:**
1. **Face Detection** (Primary) - Most reliable, expands to full person
2. **Full Body Detection** (Secondary) - When faces aren't visible
3. **Motion Detection** (Backup) - Detects moving persons

## 🚀 Quick Start

### Option 1: Simple Launcher
```bash
python run_detection.py
```

### Option 2: Command Line
```bash
# Webcam detection
python live_human_detector.py

# Video file detection
python live_human_detector.py --source "path/to/video.mp4"

# Webcam with recording
python live_human_detector.py --output "recorded_output.mp4"

# Custom confidence threshold
python live_human_detector.py --confidence 0.7
```

## 🎮 Controls

- **Q** - Quit the application
- **S** - Save screenshot with detections
- **R** - Reset person tracking
- **ESC** - Exit

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- OpenCV Python (>=4.8.0)
- NumPy (>=1.21.0)
- Flask (>=2.3.0) - for web interface
- Werkzeug (>=2.3.0)

## 🖥️ Display Information

The live display shows:
- **Bounding boxes** around detected persons
- **Person IDs** for tracking across frames
- **Detection type** (face/body/motion)
- **Confidence scores**
- **Real-time FPS**
- **Active person count**
- **Detection history graph**

## 🎯 Detection Types & Colors

- 🟢 **Green** - Face detection (most accurate)
- 🟠 **Orange** - Body detection
- 🔵 **Blue** - Motion-based detection

## ⚙️ Configuration Options

```python
detector = LiveHumanDetector(
    confidence_threshold=0.5,  # Detection sensitivity (0.1-0.9)
    nms_threshold=0.4         # Non-maximum suppression
)
```

## 📁 Project Structure

```
human_detection_project/
├── live_human_detector.py    # Main upgraded detection system
├── run_detection.py          # Simple launcher
├── human_detector.py         # Original web-based system
├── requirements.txt          # Dependencies
├── models/                   # Model files
├── uploaded_videos/          # Input videos
└── processed_videos/         # Output videos
```

## 🔧 Advanced Usage

### Command Line Arguments
- `--source` - Video source (0 for webcam, path for file)
- `--confidence` - Detection threshold (0.1-0.9)
- `--output` - Save processed video to file
- `--no-display` - Run headless (no GUI)

### Examples
```bash
# High sensitivity detection
python live_human_detector.py --confidence 0.3

# Process video and save result
python live_human_detector.py --source "input.mp4" --output "detected.mp4"

# Headless processing (servers)
python live_human_detector.py --source "video.mp4" --no-display --output "result.mp4"
```

## 🚨 Troubleshooting

### Common Issues:
1. **Camera not found** - Check if webcam is connected and not used by other apps
2. **Low FPS** - Reduce resolution or increase confidence threshold
3. **No detections** - Ensure good lighting and clear view of persons
4. **Video file errors** - Check file format (MP4, AVI, MOV supported)

### Performance Tips:
- Use lower resolution for better FPS
- Increase confidence threshold to reduce false positives
- Close other applications using the camera

## 🆚 Comparison: Original vs Upgraded

| Feature | Original | Upgraded |
|---------|----------|----------|
| Processing | Video files only | Live stream + files |
| Detection | Basic face detection | Multi-method detection |
| Tracking | None | Person tracking with IDs |
| Interface | Web-based | Real-time display |
| Performance | Batch processing | Real-time with FPS monitoring |
| Controls | None | Interactive keyboard controls |
| Visualization | Basic boxes | Enhanced with info panels |

## 🎯 Use Cases

- **Security monitoring** - Real-time person detection
- **Crowd counting** - Track people in spaces
- **Video analysis** - Process recorded footage
- **Research** - Study human movement patterns
- **Development** - Test computer vision algorithms

Ready to detect humans in real-time! 🚀👥