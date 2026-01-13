# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA 11.8+ (for GPU acceleration, optional but recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Cricket-Ball-Tracking-Refactored
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n cricket-ball python=3.10
conda activate cricket-ball
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "from src.detection import BallDetector; print('Installation successful!')"
```

## Directory Setup

The project expects the following directories to exist:

```bash
# Create data directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed/{train,val,test}
mkdir -p data/annotations
mkdir -p models/pretrained
mkdir -p models/checkpoints
mkdir -p experiments/{logs,results,metrics}
```

## GPU Setup (Optional)

### For NVIDIA GPU:

```bash
# Install CUDA-enabled PyTorch (if not already included in requirements.txt)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For CPU-only:

If you don't have a GPU, the installation will default to CPU usage. This will be slower but still functional.

## Configuration Files

Copy or review the configuration files in `config/`:

- `config/training.yaml` - Adjust hyperparameters if needed
- `config/inference.yaml` - Set input/output paths
- `config/dataset.yaml` - Update dataset paths

## Pre-trained Models

Download pre-trained YOLO11n weights:

```bash
# This will be automatically downloaded during first training/inference
# Or manually download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
```

## Troubleshooting

### ImportError: No module named 'ultralytics'

```bash
pip install ultralytics
```

### CUDA out of memory

Reduce batch size in `config/training.yaml`:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Video codec issues

Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (using conda)
conda install -c conda-forge ffmpeg
```

## Next Steps

1. Place your video files in `data/raw/`
2. Configure settings in `config/` files
3. Run training: `python scripts/train.py`
4. Run inference: `python scripts/inference.py`

For detailed usage, see [USAGE.md](USAGE.md)
