"""Ball tracking across video frames"""
from typing import Dict, List, Tuple, Optional
import numpy as np


class BallTracker:
    """
    Tracker for cricket ball across video frames.
    
    Uses centroid tracking to maintain ball identity across frames
    and interpolate missing detections.
    """
    
    def __init__(self, max_distance: float = 50.0, max_frames_missing: int = 5):
        """
        Initialize the ball tracker.
        
        Args:
            max_distance: Maximum distance to match centroids between frames
            max_frames_missing: Maximum frames to interpolate before losing track
        """
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.ball_history = []
        self.frame_count = 0
    
    def update(self, detection: Dict) -> Dict:
        """
        Update tracker with new detection.
        
        Args:
            detection: Detection result from BallDetector
            
        Returns:
            Updated tracking information
        """
        self.frame_count += 1
        
        if detection['detected']:
            self.ball_history.append({
                'frame': self.frame_count,
                'centroid': detection['centroid'],
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'interpolated': False
            })
        else:
            # Try to interpolate
            if len(self.ball_history) > 0:
                last_detection = self.ball_history[-1]
                if self.frame_count - last_detection['frame'] <= self.max_frames_missing:
                    # Extrapolate position
                    self.ball_history.append({
                        'frame': self.frame_count,
                        'centroid': last_detection['centroid'],
                        'bbox': last_detection['bbox'],
                        'confidence': 0.0,
                        'interpolated': True
                    })
        
        return self.get_current_state()
    
    def get_current_state(self) -> Dict:
        """Get current tracking state"""
        if len(self.ball_history) == 0:
            return {'tracked': False, 'centroid': None}
        
        last_entry = self.ball_history[-1]
        return {
            'tracked': True,
            'centroid': last_entry['centroid'],
            'bbox': last_entry['bbox'],
            'confidence': last_entry['confidence'],
            'interpolated': last_entry['interpolated']
        }
    
    def get_trajectory(self) -> List[List[int]]:
        """Get complete ball trajectory"""
        return [entry['centroid'] for entry in self.ball_history]
    
    def reset(self):
        """Reset tracker for new video"""
        self.ball_history = []
        self.frame_count = 0
