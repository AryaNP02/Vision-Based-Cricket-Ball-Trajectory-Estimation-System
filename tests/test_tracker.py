"""Unit tests for ball tracker"""
import unittest
from src.tracking import BallTracker


class TestBallTracker(unittest.TestCase):
    """Test cases for BallTracker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = BallTracker(max_distance=50.0, max_frames_missing=5)
    
    def test_tracker_initialization(self):
        """Test tracker can be initialized"""
        self.assertIsNotNone(self.tracker)
        self.assertEqual(self.tracker.frame_count, 0)
        self.assertEqual(len(self.tracker.ball_history), 0)
    
    def test_update_with_detection(self):
        """Test tracker update with detection"""
        detection = {
            'detected': True,
            'centroid': [100, 200],
            'bbox': [80, 180, 120, 220],
            'confidence': 0.95
        }
        
        result = self.tracker.update(detection)
        
        self.assertTrue(result['tracked'])
        self.assertEqual(result['centroid'], [100, 200])
        self.assertEqual(len(self.tracker.ball_history), 1)
    
    def test_trajectory_extraction(self):
        """Test trajectory extraction"""
        detections = [
            {'detected': True, 'centroid': [100, 200], 'bbox': None, 'confidence': 0.95},
            {'detected': True, 'centroid': [110, 210], 'bbox': None, 'confidence': 0.92},
            {'detected': True, 'centroid': [120, 220], 'bbox': None, 'confidence': 0.90},
        ]
        
        for det in detections:
            self.tracker.update(det)
        
        trajectory = self.tracker.get_trajectory()
        self.assertEqual(len(trajectory), 3)
        self.assertEqual(trajectory[0], [100, 200])
        self.assertEqual(trajectory[-1], [120, 220])
    
    def test_tracker_reset(self):
        """Test tracker reset"""
        detection = {'detected': True, 'centroid': [100, 200], 'bbox': None, 'confidence': 0.95}
        self.tracker.update(detection)
        
        self.assertEqual(len(self.tracker.ball_history), 1)
        
        self.tracker.reset()
        self.assertEqual(len(self.tracker.ball_history), 0)
        self.assertEqual(self.tracker.frame_count, 0)


if __name__ == '__main__':
    unittest.main()
