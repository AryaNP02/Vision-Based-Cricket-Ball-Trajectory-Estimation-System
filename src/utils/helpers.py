"""Helper utilities for video processing and results export"""
import csv
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np


class VideoProcessor:
    """Utilities for video processing and frame extraction"""
    
    @staticmethod
    def read_video(video_path: str) -> tuple:
        """
        Read video and extract frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames, fps, frame_count)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames, fps, frame_count
    
    @staticmethod
    def write_video(output_path: str, frames: List[np.ndarray], fps: float, codec: str = 'mp4v'):
        """
        Write frames to video file.
        
        Args:
            output_path: Output video path
            frames: List of frames
            fps: Frames per second
            codec: Video codec (default: mp4v)
        """
        if len(frames) == 0:
            raise ValueError("No frames to write")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    @staticmethod
    def draw_centroid(frame: np.ndarray, centroid: tuple, color: tuple = (0, 165, 255), radius: int = 5) -> np.ndarray:
        """Draw centroid on frame"""
        if centroid is not None:
            cv2.circle(frame, tuple(centroid), radius, color, 2)
        return frame
    
    @staticmethod
    def draw_trajectory(frame: np.ndarray, trajectory: List[List[int]], color: tuple = (0, 165, 255)) -> np.ndarray:
        """Draw trajectory on frame"""
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                pt1 = tuple(trajectory[i])
                pt2 = tuple(trajectory[i + 1])
                cv2.line(frame, pt1, pt2, color, 2)
        return frame


class ResultsExporter:
    """Export tracking results to CSV"""
    
    @staticmethod
    def export_to_csv(output_path: str, tracking_results: List[Dict]):
        """
        Export tracking results to CSV.
        
        Args:
            output_path: Path to output CSV file
            tracking_results: List of tracking results per frame
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'x', 'y', 'detected', 'confidence', 'interpolated']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for frame_idx, result in enumerate(tracking_results):
                if result['centroid'] is not None:
                    writer.writerow({
                        'frame': frame_idx,
                        'x': result['centroid'][0],
                        'y': result['centroid'][1],
                        'detected': not result.get('interpolated', False),
                        'confidence': result.get('confidence', 0.0),
                        'interpolated': result.get('interpolated', False)
                    })
    
    @staticmethod
    def export_confidence_summary(output_path: str, detections: List[Dict]):
        """
        Export confidence scores for each detection.
        
        Args:
            output_path: Path to output CSV file
            detections: List of detection results
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'confidence', 'detected']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for frame_idx, det in enumerate(detections):
                writer.writerow({
                    'frame': frame_idx,
                    'confidence': det.get('confidence', 0.0),
                    'detected': det.get('detected', False)
                })
