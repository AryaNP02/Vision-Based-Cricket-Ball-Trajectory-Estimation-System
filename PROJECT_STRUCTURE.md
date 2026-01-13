# Cricket Ball Tracking - Project Overview

## 📊 Complete Directory Structure with Descriptions

```
Cricket-Ball-Tracking-Refactored/
│
├── 📂 src/                          # ⭐ Main source code (modular & reusable)
│   ├── __init__.py                  # Package initialization
│   │
│   ├── 📂 detection/                # Ball detection module
│   │   ├── __init__.py
│   │   └── detector.py              # BallDetector class - YOLO wrapper
│   │                                 # - detect(frame) → detection dict
│   │                                 # - detect_batch(frames) → list of detections
│   │
│   ├── 📂 tracking/                 # Ball tracking module
│   │   ├── __init__.py
│   │   └── tracker.py               # BallTracker class - centroid tracking
│   │                                 # - update(detection) → tracking state
│   │                                 # - get_trajectory() → list of positions
│   │                                 # - reset() → clear history
│   │
│   └── 📂 utils/                    # Utility functions
│       ├── __init__.py
│       └── helpers.py               # VideoProcessor & ResultsExporter
│                                     # - read/write video files
│                                     # - draw annotations
│                                     # - export CSV results
│
├── 📂 data/                         # Data management
│   ├── raw/                         # 🎬 Raw input videos (place videos here)
│   │   └── .gitkeep
│   │
│   ├── processed/                   # 📁 Organized dataset (after preprocessing)
│   │   ├── train/
│   │   │   ├── images/              # Training images
│   │   │   └── labels/              # YOLO format labels (.txt files)
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   │
│   └── annotations/                 # 📝 Additional annotation files
│       └── .gitkeep
│
├── 📂 models/                       # Model management
│   ├── pretrained/                  # 🤖 Pre-trained model weights
│   │   └── yolo11n.pt               # (auto-downloaded on first use)
│   │
│   └── checkpoints/                 # 💾 Trained model checkpoints
│       ├── yolo11n_cricket_ball/
│       │   ├── weights/
│       │   │   ├── best.pt          # Best model weights
│       │   │   └── last.pt          # Last epoch weights
│       │   ├── args.yaml            # Training arguments
│       │   └── results.csv          # Training metrics
│       └── .gitkeep
│
├── 📂 scripts/                      # Executable scripts
│   ├── train.py                     # 🎓 Training script
│   │                                 # Usage: python scripts/train.py
│   │                                 # - Loads config from YAML
│   │                                 # - Trains YOLO model
│   │                                 # - Saves checkpoints
│   │
│   ├── inference.py                 # 🔍 Inference & tracking script
│   │                                 # Usage: python scripts/inference.py --video input.mp4
│   │                                 # - Detects ball in each frame
│   │                                 # - Tracks ball across frames
│   │                                 # - Exports CSV & video
│   │
│   └── preprocess.py                # ⚙️ Data preprocessing script
│                                     # - Organizes dataset into splits
│                                     # - Converts annotations
│
├── 📂 config/                       # Configuration files
│   ├── training.yaml                # 🎓 Training hyperparameters
│   │                                 # - model architecture
│   │                                 # - epochs, batch size, learning rate
│   │                                 # - augmentation settings
│   │
│   ├── inference.yaml               # 🔍 Inference configuration
│   │                                 # - model weights path
│   │                                 # - confidence threshold
│   │                                 # - tracking parameters
│   │                                 # - output paths
│   │
│   └── dataset.yaml                 # 📊 Dataset configuration
│                                     # - data paths
│                                     # - number of classes
│                                     # - class names
│
├── 📂 experiments/                  # Experiment tracking & results
│   ├── logs/                        # 📋 Training/inference logs
│   │   ├── train_*.log              # Training logs
│   │   └── .gitkeep
│   │
│   ├── results/                     # 📊 Inference results
│   │   ├── videos/                  # Output videos with annotations
│   │   │   ├── video1_tracking.mp4
│   │   │   └── video2_tracking.mp4
│   │   ├── csv/                     # CSV tracking data
│   │   │   ├── video1_tracking.csv
│   │   │   └── video2_tracking.csv
│   │   └── confidence/              # Confidence scores per frame
│   │       ├── video1_confidence.csv
│   │       └── video2_confidence.csv
│   │
│   └── metrics/                     # 📈 Evaluation metrics
│       └── .gitkeep
│
├── 📂 docs/                         # Documentation
│   ├── README.md                    # This file (project overview)
│   ├── SETUP.md                     # 🔧 Setup & installation guide
│   ├── API.md                       # 📖 Complete API reference
│   ├── USAGE.md                     # 💡 Usage guide & examples
│   └── images/                      # Documentation images
│
├── 📂 tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_detector.py             # Tests for BallDetector
│   ├── test_tracker.py              # Tests for BallTracker
│   └── test_utils.py                # Tests for utility functions
│
├── 📂 notebooks/                    # Jupyter notebooks
│   ├── analysis.ipynb               # Data analysis & visualization
│   └── .gitkeep
│
├── requirements.txt                 # Python dependencies list
├── setup.py                         # Package setup script
├── .gitignore                       # Git ignore rules
├── README.md                        # Main readme
└── MIGRATION.md                     # Migration guide from old structure
```

---

## 🎯 Quick Navigation

### For Training
```
Training Files:
- scripts/train.py              ← Run this
- config/training.yaml          ← Configure this
- data/processed/               ← Put data here
- models/checkpoints/           ← Results go here
```

### For Inference
```
Inference Files:
- scripts/inference.py          ← Run this
- config/inference.yaml         ← Configure this
- data/raw/                     ← Put videos here
- experiments/results/          ← Results go here
```

### For API Usage
```
Core Classes:
- src/detection/detector.py     ← BallDetector
- src/tracking/tracker.py       ← BallTracker
- src/utils/helpers.py          ← Helper functions
- docs/API.md                   ← Full API reference
```

---

## 📊 File Purpose Reference

| File | Purpose | Used For |
|------|---------|----------|
| `train.py` | Execute model training | `python scripts/train.py` |
| `inference.py` | Run detection & tracking | `python scripts/inference.py` |
| `preprocess.py` | Organize dataset | `python scripts/preprocess.py` |
| `training.yaml` | Training settings | Hyperparameter tuning |
| `inference.yaml` | Inference settings | Output format, thresholds |
| `dataset.yaml` | Dataset paths | Points to data splits |
| `detector.py` | Ball detection logic | Core ML component |
| `tracker.py` | Ball tracking logic | Temporal analysis |
| `helpers.py` | I/O & visualization | Video & CSV operations |
| `test_*.py` | Unit tests | Verify code correctness |

---

## 🔄 Data Flow

### Training Pipeline
```
data/raw/videos
    ↓
scripts/preprocess.py
    ↓
data/processed/train/val/test/
    ↓
scripts/train.py
    ↓
models/checkpoints/best.pt
    ↓
experiments/logs/train.log
```

### Inference Pipeline
```
data/raw/video.mp4
    ↓
scripts/inference.py
    ↓
src/detection/detector.py (detect ball)
    ↓
src/tracking/tracker.py (track across frames)
    ↓
src/utils/helpers.py (draw & export)
    ↓
experiments/results/
├── video_tracking.csv
├── video_tracking.mp4
└── video_confidence.csv
```

---

## 🚀 Getting Started in 3 Steps

### Step 1: Setup
```bash
cd Cricket-Ball-Tracking-Refactored
pip install -r requirements.txt
```

### Step 2: Prepare Data
```bash
# Place videos in data/raw/
cp your_videos/*.mp4 data/raw/

# Or train with existing dataset
# (copy from old project)
```

### Step 3: Run
```bash
# Training
python scripts/train.py

# Inference
python scripts/inference.py
```

Results will be in `experiments/results/`

---

## 📚 Documentation Map

```
New to project?
├─→ START HERE: README.md (this file)
├─→ THEN: docs/SETUP.md (installation)
├─→ NEXT: docs/USAGE.md (how to use)
└─→ REFERENCE: docs/API.md (code details)

Want to use code?
├─→ Import: from src.detection import BallDetector
├─→ Reference: docs/API.md
└─→ Examples: docs/USAGE.md

Want to configure?
├─→ Training: edit config/training.yaml
├─→ Inference: edit config/inference.yaml
└─→ Dataset: edit config/dataset.yaml

Want to extend?
├─→ Add to: src/detection/ or src/tracking/
├─→ Test: tests/test_*.py
└─→ Document: docs/API.md

Having issues?
├─→ Check: docs/SETUP.md (setup issues)
├─→ Check: docs/USAGE.md (usage issues)
└─→ Check: tests/ (code tests)
```

---

## ✨ Key Features of New Structure

✅ **Modular Design**
- Each component is independent
- Easy to test and modify
- Can be used separately

✅ **Professional Standards**
- Follows Python conventions
- Industry best practices
- Ready for production

✅ **Comprehensive Documentation**
- Setup guide
- API reference
- Usage examples

✅ **Configuration Management**
- YAML-based configs
- Version controllable
- Easy to experiment

✅ **Results Organization**
- Separate folders for different outputs
- Consistent naming
- Easy to track experiments

✅ **Testing Framework**
- Unit tests included
- Easy to extend
- Ensures code quality

✅ **Scalable**
- Can add new modules
- Can handle multiple datasets
- Ready for deployment

---



Happy tracking! 🎯
