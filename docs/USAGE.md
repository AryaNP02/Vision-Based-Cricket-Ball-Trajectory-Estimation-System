# Usage Guide

## Quick Start

### 1. Prepare Your Data

Place cricket videos in `data/raw/`:
```bash
data/raw/
├── video1.mp4
├── video2.mp4
└── video3.avi
```

### 2. Run Inference

```bash
python scripts/inference.py
```

This will:
- Detect cricket ball in each frame
- Track ball across frames
- Generate output video with annotations
- Export CSV with per-frame coordinates
- Save confidence scores

### 3. Check Results

Results are organized by confidence threshold:
```
experiments/
├── results/
│   ├── videos/          # Annotated output videos
│   ├── csv/             # Tracking coordinates
│   └── confidence/      # Confidence scores
└── logs/                # Training/inference logs
```

---

## Training

### Basic Training

```bash
python scripts/train.py
```

### Custom Training

Edit `config/training.yaml` then run:

```bash
python scripts/train.py --config config/training.yaml
```

#### Key Training Parameters:

```yaml
training:
  epochs: 50           # Number of training cycles
  batch_size: 16       # Samples per batch
  image_size: 640      # Input resolution

optimization:
  lr0: 0.01           # Initial learning rate
  momentum: 0.937     # SGD momentum

augmentation:
  mosaic: true        # Image mosaic augmentation
  fliplr: 0.5         # Horizontal flip probability
  hsv_v: 0.4          # HSV value augmentation
```

#### Monitoring Training:

Training logs are saved to `experiments/logs/`

Check metrics:
```bash
tail -f experiments/logs/training.log
```

View results:
```bash
python -c "import pandas as pd; df = pd.read_csv('models/checkpoints/yolo11n_cricket_ball/results.csv'); print(df[['epoch', 'metrics/precision(B)', 'metrics/recall(B)']].tail())"
```

---

## Inference Options

### Process All Videos

```bash
python scripts/inference.py
```

### Process Specific Video

```bash
python scripts/inference.py --video path/to/video.mp4
```

### Custom Configuration

```bash
python scripts/inference.py --config my_config.yaml
```

### Different Confidence Thresholds

Edit `config/inference.yaml`:
```yaml
model:
  conf_threshold: 0.25  # Lower threshold = more detections
```

Options:
- `0.1` - Very sensitive, many false positives
- `0.35` - Balanced (default)
- `0.5` - Conservative, fewer false positives
- `0.9` - Very strict, only high confidence detections

---

## Output Formats

### CSV Output

Track coordinates for each frame:

```csv
frame,x,y,detected,confidence,interpolated
0,320,240,1,0.95,0
1,325,245,1,0.92,0
2,330,250,1,0.88,0
```

**Fields:**
- `frame`: Frame number (0-indexed)
- `x`, `y`: Ball centroid coordinates
- `detected`: 1 if detected by model, 0 if interpolated
- `confidence`: Detection confidence (0-1)
- `interpolated`: 1 if position was extrapolated

### Video Output

Processed video with:
- Ball centroid marked with **amber circle**
- Trajectory line showing ball path
- Frame-by-frame annotations

### Confidence Summary

Confidence scores for each frame:

```csv
frame,confidence,detected
0,0.95,1
1,0.92,1
2,0.0,0
```

---

## Data Preprocessing

Prepare custom dataset:

```bash
python scripts/preprocess.py \
    --source path/to/raw/data \
    --output data/processed \
    --train-ratio 0.7 \
    --val-ratio 0.15
```

This creates:
- `data/processed/train/` - 70% of data
- `data/processed/val/` - 15% of data
- `data/processed/test/` - 15% of data

---

## Python API Usage

### Basic Detection

```python
from src.detection import BallDetector
import cv2

# Load detector
detector = BallDetector('models/checkpoints/best.pt', conf_threshold=0.35)

# Load frame
frame = cv2.imread('frame.jpg')

# Detect
result = detector.detect(frame)

if result['detected']:
    x, y = result['centroid']
    conf = result['confidence']
    print(f"Ball at ({x}, {y}) with confidence {conf:.2%}")
else:
    print("No ball detected")
```

### Tracking

```python
from src.detection import BallDetector
from src.tracking import BallTracker
from src.utils import VideoProcessor
import cv2

# Setup
detector = BallDetector('models/checkpoints/best.pt')
tracker = BallTracker(max_distance=50)
frames, fps, _ = VideoProcessor.read_video('video.mp4')

# Process
results = []
for frame in frames:
    detection = detector.detect(frame)
    state = tracker.update(detection)
    results.append(state)

# Get trajectory
trajectory = tracker.get_trajectory()
print(f"Detected {len(trajectory)} frames with ball")
```

### Batch Processing

```python
from src.detection import BallDetector

detector = BallDetector('models/checkpoints/best.pt')

# Process multiple frames efficiently
detections = detector.detect_batch(frames)

for i, det in enumerate(detections):
    if det['detected']:
        print(f"Frame {i}: Ball at {det['centroid']}")
```

### Custom Visualization

```python
from src.utils import VideoProcessor
import cv2

frame = cv2.imread('frame.jpg')

# Draw centroid
frame = VideoProcessor.draw_centroid(frame, (320, 240), color=(0, 165, 255))

# Draw trajectory
trajectory = [[100, 200], [110, 210], [120, 220]]
frame = VideoProcessor.draw_trajectory(frame, trajectory)

cv2.imshow('Annotated', frame)
cv2.waitKey(0)
```

### Export Results

```python
from src.utils import ResultsExporter

# Export tracking CSV
ResultsExporter.export_to_csv('results.csv', tracking_results)

# Export confidence summary
ResultsExporter.export_confidence_summary('confidence.csv', detections)
```

---

## Advanced Configuration

### Multi-GPU Training

```yaml
training:
  device: [0, 1, 2]  # List of GPU IDs
  batch_size: 48     # Adjust for total VRAM
```

### Custom Augmentation

```yaml
augmentation:
  mosaic: true
  flipud: 0.5
  fliplr: 0.5
  blur: 0.1
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  perspective: 0.0
```

### Custom Tracking Parameters

```yaml
tracking:
  max_distance: 75      # Higher = more tolerance for movement
  max_frames_missing: 10 # Higher = interpolate more frames
```

---

## Troubleshooting

### No detections in output

**Solution 1: Lower confidence threshold**
```yaml
model:
  conf_threshold: 0.25  # Lower threshold
```

**Solution 2: Check model weights**
```bash
ls -lh models/checkpoints/best.pt
```

**Solution 3: Verify video quality**
- Ensure video has sufficient lighting
- Check that ball is visible in frames
- Try with test frame: `python -c "import cv2; f = cv2.imread('frame.jpg'); print(f.shape)"`

### Slow processing

**Solution 1: Use GPU**
```yaml
training:
  device: 0  # GPU instead of -1 (CPU)
```

**Solution 2: Reduce image size**
```yaml
training:
  image_size: 416  # Instead of 640
```

**Solution 3: Lower resolution input**
- Pre-process video to lower resolution
- Crop to region of interest

### Memory errors

**Solution 1: Reduce batch size**
```yaml
training:
  batch_size: 8  # Instead of 16
```

**Solution 2: Reduce image size**
```yaml
training:
  image_size: 416  # Instead of 640
```

**Solution 3: Enable mixed precision**
```yaml
training:
  mixed_precision: true
```

### Video codec errors

**Solution: Install FFmpeg**
```bash
# Ubuntu
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

---

## Performance Optimization

### For Speed (Real-time Processing)

```yaml
model:
  conf_threshold: 0.5        # Skip low confidence
training:
  image_size: 416            # Smaller input
  device: 0                  # GPU
```

### For Accuracy (Best Results)

```yaml
model:
  conf_threshold: 0.25       # Accept more detections
training:
  image_size: 640            # Larger input
  epochs: 100                # More training
```

---

## Integration Examples

### Integrate with Existing Pipeline

```python
def process_cricket_video(video_path, output_path):
    from src.detection import BallDetector
    from src.tracking import BallTracker
    from src.utils import VideoProcessor, ResultsExporter
    
    detector = BallDetector('models/checkpoints/best.pt')
    tracker = BallTracker()
    
    frames, fps, _ = VideoProcessor.read_video(video_path)
    results = []
    out_frames = []
    
    for frame in frames:
        detection = detector.detect(frame)
        tracker.update(detection)
        
        frame_out = frame.copy()
        frame_out = VideoProcessor.draw_centroid(frame_out, tracker.get_current_state()['centroid'])
        out_frames.append(frame_out)
        
        results.append(tracker.get_current_state())
    
    VideoProcessor.write_video(output_path, out_frames, fps)
    ResultsExporter.export_to_csv(output_path.replace('.mp4', '.csv'), results)

# Usage
process_cricket_video('input.mp4', 'output.mp4')
```

---

