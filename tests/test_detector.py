"""Unit tests for ball detector"""
import unittest
import numpy as np
from src.detection import BallDetector


class TestBallDetector(unittest.TestCase):
    """Test cases for BallDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # This would require a valid model path
        self.model_path = "models/checkpoints/best.pt"
    
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        try:
            detector = BallDetector(self.model_path)
            self.assertIsNotNone(detector.model)
        except FileNotFoundError:
            self.skipTest("Model weights not found")
    
    def test_detect_empty_frame(self):
        """Test detection on empty frame"""
        try:
            detector = BallDetector(self.model_path)
            # Create a blank frame
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            result = detector.detect(frame)
            
            self.assertIn('detected', result)
            self.assertIn('centroid', result)
            self.assertIsInstance(result['detected'], bool)
        except FileNotFoundError:
            self.skipTest("Model weights not found")


if __name__ == '__main__':
    unittest.main()
