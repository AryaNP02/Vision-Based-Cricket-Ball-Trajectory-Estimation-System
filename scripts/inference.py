#!/usr/bin/env python3
"""
Inference script for cricket ball detection and tracking.

Detects and tracks cricket ball in video files.
"""

import argparse
import yaml
from pathlib import Path
import cv2
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import BallDetector
from src.tracking import BallTracker
from src.utils import VideoProcessor, ResultsExporter


def load_config(config_path: str) -> dict:
    """Load inference configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_video(video_path: str, detector: BallDetector, tracker: BallTracker, config: dict) -> tuple:
    """
    Process a single video file.
    
    Args:
        video_path: Path to input video
        detector: BallDetector instance
        tracker: BallTracker instance
        config: Configuration dictionary
        
    Returns:
        Tuple of (tracking_results, frames, fps)
    """
    print(f"Processing: {video_path}")
    
    # Read video
    frames, fps, frame_count = VideoProcessor.read_video(video_path)
    print(f"  Frames: {frame_count}, FPS: {fps}")
    
    # Process each frame
    tracking_results = []
    processed_frames = []
    detections = []
    
    for i, frame in enumerate(frames):
        # Detect ball
        detection = detector.detect(frame)
        detections.append(detection)
        
        # Update tracker
        tracker.update(detection)
        track_state = tracker.get_current_state()
        tracking_results.append(track_state)
        
        # Draw annotations
        frame_copy = frame.copy()
        if config['output']['draw_trajectory']:
            trajectory = tracker.get_trajectory()
            frame_copy = VideoProcessor.draw_trajectory(
                frame_copy, 
                trajectory,
                tuple(config['output']['trajectory_color'])
            )
        
        frame_copy = VideoProcessor.draw_centroid(
            frame_copy,
            track_state['centroid'],
            tuple(config['output']['centroid_color']),
            config['output']['centroid_radius']
        )
        
        processed_frames.append(frame_copy)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{frame_count} frames")
    
    return tracking_results, processed_frames, fps, detections


def run_inference(config_path: str = 'config/inference.yaml', video_file: str = None):
    """
    Run inference on video(s).
    
    Args:
        config_path: Path to inference configuration file
        video_file: Specific video file to process (optional)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize detector and tracker
    model_weights = config['model']['weights']
    conf_threshold = config['model']['conf_threshold']
    
    detector = BallDetector(model_weights, conf_threshold)
    
    # Get video files to process
    if video_file:
        video_files = [video_file]
    else:
        input_dir = Path(config['paths']['input_videos'])
        video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))
    
    # Create output directories
    Path(config['paths']['output_videos']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['output_csv']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['confidence_output']).mkdir(parents=True, exist_ok=True)
    
    # Process each video
    for video_path in video_files:
        # Reset tracker
        tracker = BallTracker(
            max_distance=config['tracking']['max_distance'],
            max_frames_missing=config['tracking']['max_frames_missing']
        )
        
        # Process video
        tracking_results, processed_frames, fps, detections = process_video(
            str(video_path), detector, tracker, config
        )
        
        # Save results
        video_stem = video_path.stem
        
        if config['output']['save_video']:
            output_video = Path(config['paths']['output_videos']) / f"{video_stem}_tracking.mp4"
            VideoProcessor.write_video(
                str(output_video),
                processed_frames,
                fps,
                config['output']['video_codec']
            )
            print(f"  Saved video: {output_video}")
        
        if config['output']['save_csv']:
            output_csv = Path(config['paths']['output_csv']) / f"{video_stem}_tracking.csv"
            ResultsExporter.export_to_csv(str(output_csv), tracking_results)
            print(f"  Saved CSV: {output_csv}")
            
            # Also save confidence summary
            conf_csv = Path(config['paths']['confidence_output']) / f"{video_stem}_confidence.csv"
            ResultsExporter.export_confidence_summary(str(conf_csv), detections)
            print(f"  Saved confidence: {conf_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on cricket ball video')
    parser.add_argument('--config', type=str, default='config/inference.yaml',
                       help='Path to inference config file')
    parser.add_argument('--video', type=str, default=None,
                       help='Specific video file to process')
    
    args = parser.parse_args()
    run_inference(args.config, args.video)
