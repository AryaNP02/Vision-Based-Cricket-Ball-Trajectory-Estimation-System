# API Documentation

## Core Classes and Functions

### Detection Module

#### `BallDetector`

Main class for detecting cricket balls in frames.

**Constructor:**
```python
BallDetector(model_path: str, conf_threshold: float = 0.35)
```

**Parameters:**
- `model_path` (str): Path to YOLO model weights file
- `conf_threshold` (float): Confidence threshold for detections (0.0-1.0)

**Methods:**

##### `detect(frame: np.ndarray) -> Dict`

Detect cricket ball in a single frame.

**Parameters:**
- `frame` (np.ndarray): Input frame in BGR format (height, width, 3)

**Returns:**
- Dictionary with keys:
  - `detected` (bool): Whether ball was detected
  - `bbox` (list): Bounding box [x1, y1, x2, y2] or None
  - `centroid` (list): Ball center [x, y] or None
  - `confidence` (float): Detection confidence score (0.0-1.0)

**Example:**
```python
detector = BallDetector('models/best.pt')
result = detector.detect(frame)
if result['detected']:
    print(f"Ball at: {result['centroid']}, confidence: {result['confidence']}")
```

##### `detect_batch(frames: List[np.ndarray]) -> List[Dict]`

Detect ball in multiple frames.

**Parameters:**
- `frames` (List[np.ndarray]): List of frames

**Returns:**
- List of detection results (same format as `detect()`)

---

### Tracking Module

#### `BallTracker`

Tracks cricket ball across video frames.

**Constructor:**
```python
BallTracker(max_distance: float = 50.0, max_frames_missing: int = 5)
```

**Parameters:**
- `max_distance` (float): Maximum pixel distance to match centroids between frames
- `max_frames_missing` (int): Maximum consecutive frames without detection before losing track

**Methods:**

##### `update(detection: Dict) -> Dict`

Update tracker with new detection.

**Parameters:**
- `detection` (Dict): Detection result from `BallDetector.detect()`

**Returns:**
- Dictionary with keys:
  - `tracked` (bool): Whether ball is being tracked
  - `centroid` (list): Current ball position [x, y]
  - `bbox` (list): Bounding box or None
  - `confidence` (float): Detection confidence
  - `interpolated` (bool): Whether position was interpolated

**Example:**
```python
tracker = BallTracker()
for frame in frames:
    detection = detector.detect(frame)
    track_state = tracker.update(detection)
    print(f"Ball at frame {tracker.frame_count}: {track_state['centroid']}")
```

##### `get_trajectory() -> List[List[int]]`

Get complete ball trajectory up to current frame.

**Returns:**
- List of [x, y] coordinates representing the ball's path

**Example:**
```python
trajectory = tracker.get_trajectory()
# trajectory = [[100, 200], [105, 205], [110, 210], ...]
```

##### `get_current_state() -> Dict`

Get current tracking state.

**Returns:**
- Dictionary with same format as `update()` return value

##### `reset()`

Reset tracker for new video.

**Parameters:** None

**Returns:** None

---

### Utils Module

#### `VideoProcessor`

Utilities for video I/O and frame manipulation.

**Static Methods:**

##### `read_video(video_path: str) -> Tuple`

Read video file and extract frames.

**Parameters:**
- `video_path` (str): Path to video file

**Returns:**
- Tuple of:
  - `frames` (List[np.ndarray]): List of frames
  - `fps` (float): Frames per second
  - `frame_count` (int): Total number of frames

**Example:**
```python
frames, fps, count = VideoProcessor.read_video('video.mp4')
print(f"Video has {count} frames at {fps} FPS")
```

##### `write_video(output_path: str, frames: List[np.ndarray], fps: float, codec: str = 'mp4v')`

Write frames to video file.

**Parameters:**
- `output_path` (str): Output video file path
- `frames` (List[np.ndarray]): Frames to write (all same shape)
- `fps` (float): Frames per second
- `codec` (str): Video codec ('mp4v', 'MJPG', etc.)

**Returns:** None

**Example:**
```python
VideoProcessor.write_video('output.mp4', processed_frames, 30.0)
```

##### `draw_centroid(frame: np.ndarray, centroid: tuple, color: tuple = (0, 165, 255), radius: int = 5) -> np.ndarray`

Draw ball centroid on frame.

**Parameters:**
- `frame` (np.ndarray): Input frame
- `centroid` (tuple): (x, y) coordinate
- `color` (tuple): BGR color (default: Amber)
- `radius` (int): Circle radius in pixels

**Returns:**
- Modified frame with centroid drawn

##### `draw_trajectory(frame: np.ndarray, trajectory: List[List[int]], color: tuple = (0, 165, 255)) -> np.ndarray`

Draw ball trajectory line on frame.

**Parameters:**
- `frame` (np.ndarray): Input frame
- `trajectory` (List[List[int]]): List of [x, y] points
- `color` (tuple): BGR color

**Returns:**
- Modified frame with trajectory drawn

---

#### `ResultsExporter`

Export tracking results to files.

**Static Methods:**

##### `export_to_csv(output_path: str, tracking_results: List[Dict])`

Export tracking results to CSV file.

**Parameters:**
- `output_path` (str): Output CSV file path
- `tracking_results` (List[Dict]): Tracking results for each frame

**CSV Format:**
```
frame,x,y,detected,confidence,interpolated
0,320,240,1,0.95,0
1,325,245,1,0.92,0
2,330,250,1,0.88,0
```

**Example:**
```python
ResultsExporter.export_to_csv('results.csv', tracking_results)
```

##### `export_confidence_summary(output_path: str, detections: List[Dict])`

Export detection confidence scores to CSV.

**Parameters:**
- `output_path` (str): Output CSV file path
- `detections` (List[Dict]): Detection results for each frame

**CSV Format:**
```
frame,confidence,detected
0,0.95,1
1,0.92,1
2,0.0,0
```

---

## Example: Complete Pipeline

```python
from src.detection import BallDetector
from src.tracking import BallTracker
from src.utils import VideoProcessor, ResultsExporter
import cv2

# Initialize components
detector = BallDetector('models/checkpoints/best.pt', conf_threshold=0.35)
tracker = BallTracker(max_distance=50.0, max_frames_missing=5)

# Load video
frames, fps, frame_count = VideoProcessor.read_video('cricket_video.mp4')

# Process frames
tracking_results = []
processed_frames = []

for frame in frames:
    # Detect ball
    detection = detector.detect(frame)
    
    # Track ball
    track_state = tracker.update(detection)
    tracking_results.append(track_state)
    
    # Draw annotations
    frame_copy = frame.copy()
    frame_copy = VideoProcessor.draw_trajectory(
        frame_copy, 
        tracker.get_trajectory()
    )
    frame_copy = VideoProcessor.draw_centroid(
        frame_copy, 
        track_state['centroid']
    )
    processed_frames.append(frame_copy)

# Export results
ResultsExporter.export_to_csv('results.csv', tracking_results)
VideoProcessor.write_video('output.mp4', processed_frames, fps)

print("Processing complete!")
```

---

## Configuration Parameters

See `config/` directory for YAML configuration files.

### Training Config Parameters

```yaml
model.epochs          # Number of training epochs (50 recommended)
model.batch_size      # Batch size for training (16 recommended)
model.image_size      # Input image resolution (640x640 recommended)
model.learning_rate   # Learning rate
augmentation.mosaic   # Enable mosaic augmentation (recommended)
```

### Inference Config Parameters

```yaml
model.conf_threshold  # Confidence threshold (0.35 recommended)
tracking.max_distance # Max centroid distance between frames (50 recommended)
tracking.max_frames_missing  # Max frames to interpolate (5 recommended)
output.draw_trajectory       # Draw trajectory line on output
output.save_csv             # Export results to CSV
output.save_video           # Export processed video
```

---

## Error Handling

All classes raise informative exceptions:

```python
try:
    detector = BallDetector('nonexistent.pt')
except FileNotFoundError:
    print("Model file not found")

try:
    frame = cv2.imread('nonexistent.jpg')
    result = detector.detect(frame)
except RuntimeError as e:
    print(f"Detection failed: {e}")
```

---

## Performance Tips

1. **Use GPU**: Ensure CUDA is properly installed for ~10x speedup
2. **Batch Processing**: Use `detect_batch()` for faster processing
3. **Lower Confidence Threshold**: Detect more balls but with more false positives
4. **Reduce Image Size**: Process smaller frames for speed vs accuracy tradeoff

---

## Common Issues

**Q: ImportError: No module named 'ultralytics'**
A: Install with `pip install ultralytics`

**Q: CUDA out of memory**
A: Reduce batch size or frame resolution

**Q: Video write errors**
A: Ensure FFmpeg is installed and video codec is supported

For more help, see [SETUP.md](SETUP.md)
