#!/usr/bin/env python3
"""
Training script for cricket ball detection model.

Trains a YOLO model on the cricket ball dataset.
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(config_path: str = 'config/training.yaml'):
    """
    Train YOLO model for cricket ball detection.
    
    Args:
        config_path: Path to training configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Initialize model
    model_name = config['model']['pretrained_weights']
    model = YOLO(model_name)
    
    # Prepare training arguments
    train_args = {
        'data': config['dataset']['config'],
        'epochs': config['training']['epochs'],
        'imgsz': config['training']['image_size'],
        'batch': config['training']['batch_size'],
        'patience': config['training']['patience'],
        'device': config['training']['device'],
        'cache': config['training']['cache'],
        'name': config['output']['experiment_name'],
        'project': config['output']['save_dir'],
        'save': True,
        'exist_ok': True,
    }
    
    # Add augmentation settings
    if config['augmentation']['mosaic']:
        train_args['mosaic'] = True
    
    # Train model
    print("Starting training...")
    results = model.train(**train_args)
    
    print(f"Training completed. Results saved to {config['output']['save_dir']}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train cricket ball detection model')
    parser.add_argument('--config', type=str, default='config/training.yaml',
                       help='Path to training config file')
    
    args = parser.parse_args()
    train(args.config)
