# Cricket Ball Tracking - Refactored Project

A modern, modular computer vision pipeline for detecting and tracking cricket balls in video.

## ✨ Features

- **Per-frame Detection**: YOLO 11n-based cricket ball detection in each video frame
- **Trajectory Tracking**: Centroid-based tracking across frames
- **CSV Export**: Frame-by-frame annotations with confidence scores
- **Video Overlay**: Processed videos with visual tracking annotations
- **Flexible Configuration**: YAML-based configuration for training and inference
- **Modular Design**: Clean, reusable code components

## 📁 Project Structure

```
Cricket-Ball-Tracking-Refactored/
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── detection/              # Ball detection module
│   │   ├── __init__.py
│   │   └── detector.py         # YOLO detector wrapper
│   ├── tracking/               # Ball tracking module
│   │   ├── __init__.py
│   │   └── tracker.py          # Centroid tracker
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       └── helpers.py          # Video processing & export
│
├── data/                       # Data management
│   ├── raw/                    # Raw input videos
│   ├── processed/              # Organized dataset
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/            # Additional annotations
│
├── models/                     # Model storage
│   ├── pretrained/             # Pre-trained weights
│   └── checkpoints/            # Training checkpoints
│
├── scripts/                    # Executable scripts
│   ├── train.py                # Training script
│   ├── inference.py            # Inference/tracking script
│   └── preprocess.py           # Data preprocessing
│
├── config/                     # Configuration files
│   ├── training.yaml           # Training configuration
│   ├── inference.yaml          # Inference configuration
│   └── dataset.yaml            # Dataset configuration
│
├── experiments/                # Results and logs
│   ├── logs/                   # Training logs
│   ├── results/                # Inference results
│   └── metrics/                # Evaluation metrics
│
├── docs/                       # Documentation
│   ├── SETUP.md               # Setup instructions
│   ├── API.md                 # API documentation
│   └── USAGE.md               # Usage examples
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_detector.py        # Detector tests
│   └── test_tracker.py         # Tracker tests
│
├── notebooks/                  # Jupyter notebooks
│   └── analysis.ipynb          # Analysis & visualization
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project
cd Cricket-Ball-Tracking-Refactored

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config config/training.yaml
```

### 3. Inference

```bash
# Run inference on all videos in data/raw/
python scripts/inference.py

# Run inference on specific video
python scripts/inference.py --video path/to/video.mp4

# Use custom configuration
python scripts/inference.py --config config/inference.yaml
```

## 📊 Configuration Files

### Training Configuration (`config/training.yaml`)
- Model selection and pre-trained weights
- Training hyperparameters (epochs, batch size, learning rate)
- Data augmentation settings
- Output paths for checkpoints

### Inference Configuration (`config/inference.yaml`)
- Model weights path
- Detection confidence threshold
- Tracking parameters
- Output format (CSV, video, etc.)
- Input/output directories

### Dataset Configuration (`config/dataset.yaml`)
- Dataset paths and splits
- Number of classes
- Class names

## 🔧 API Usage

### Ball Detection

```python
from src.detection import BallDetector
import cv2

# Initialize detector
detector = BallDetector('models/checkpoints/best.pt', conf_threshold=0.35)

# Load frame
frame = cv2.imread('frame.jpg')

# Detect ball
result = detector.detect(frame)
print(result)
# Output: {
#     'detected': True,
#     'centroid': [320, 240],
#     'bbox': [300, 220, 340, 260],
#     'confidence': 0.95
# }
```

### Ball Tracking

```python
from src.tracking import BallTracker

# Initialize tracker
tracker = BallTracker(max_distance=50.0, max_frames_missing=5)

# Update with detections
detection = {
    'detected': True,
    'centroid': [320, 240],
    'bbox': [300, 220, 340, 260],
    'confidence': 0.95
}

tracker.update(detection)

# Get trajectory
trajectory = tracker.get_trajectory()
print(trajectory)  # List of [x, y] coordinates
```

### Video Processing

```python
from src.utils import VideoProcessor, ResultsExporter

# Read video
frames, fps, frame_count = VideoProcessor.read_video('video.mp4')

# Process frames...

# Write results
ResultsExporter.export_to_csv('output.csv', tracking_results)
VideoProcessor.write_video('output.mp4', processed_frames, fps)
```

## 📈 Output Files

### CSV Format
```
frame,x,y,detected,confidence,interpolated
0,320,240,True,0.95,False
1,325,245,True,0.92,False
2,330,250,False,0.0,True
```

### Video Output
Processed videos with:
- Ball centroid marked with circle
- Trajectory line connecting detections
- Confidence scores displayed (optional)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_detector.py

# Run with coverage
python -m pytest tests/ --cov=src
```



## 📝 Dataset Information

This project uses two datasets:

1. **Cricket Ball YOLO Dataset** (Kaggle)
   - Pre-annotated cricket ball images
   - YOLO format annotations

2. **Bat-Ball Tracking Dataset** (GitHub)
   - Real match video frames
   - Processed to extract ball annotations only



## 🎯 Model Training Results

### YOLO11n Performance
- **Precision**: 0.988
- **Recall**: 0.967
- **mAP@50**: 0.986
- **mAP@50-95**: 0.877


