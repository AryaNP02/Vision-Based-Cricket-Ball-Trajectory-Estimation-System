"""Ball detection using YOLO model"""
from typing import Dict, List, Tuple, Optional
import numpy as np


class BallDetector:
    """
    Cricket ball detector using YOLO model.
    
    This class handles loading a YOLO model and performing
    inference on video frames to detect cricket balls.
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.35):
        """
        Initialize the ball detector.
        
        Args:
            model_path: Path to the YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model from path"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect cricket ball in a frame.
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            Dictionary containing detection results with keys:
            - 'detected': bool indicating if ball was found
            - 'bbox': bounding box [x1, y1, x2, y2]
            - 'centroid': [x, y] coordinates
            - 'confidence': confidence score
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = {
            'detected': False,
            'bbox': None,
            'centroid': None,
            'confidence': 0.0
        }
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get the detection with highest confidence
            boxes = results[0].boxes
            best_idx = np.argmax(boxes.conf.cpu().numpy())
            
            box = boxes[best_idx]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            
            detections['detected'] = True
            detections['bbox'] = [x1, y1, x2, y2]
            detections['centroid'] = [centroid_x, centroid_y]
            detections['confidence'] = confidence
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Detect cricket ball in multiple frames.
        
        Args:
            frames: List of frames
            
        Returns:
            List of detection results
        """
        return [self.detect(frame) for frame in frames]
