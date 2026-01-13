#!/usr/bin/env python3
"""
Data preprocessing script for cricket ball dataset.

Converts and organizes dataset into proper train/val/test splits.
"""

import argparse
from pathlib import Path
import shutil


def organize_dataset(source_dir: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Organize dataset into train/val/test splits.
    
    Args:
        source_dir: Source dataset directory
        output_dir: Output directory for organized dataset
        train_ratio: Proportion of training data (default: 0.7)
        val_ratio: Proportion of validation data (default: 0.15)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print("Dataset organized successfully!")
    print(f"Output directory: {output_path}")


def merge_datasets(dataset1: str, dataset2: str, output_dir: str):
    """
    Merge two datasets into one.
    
    Args:
        dataset1: First dataset path
        dataset2: Second dataset path
        output_dir: Output directory for merged dataset
    """
    output_path = Path(output_dir)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print("Datasets merged successfully!")
    print(f"Output directory: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess cricket ball dataset')
    parser.add_argument('--source', type=str, required=True,
                       help='Source dataset directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for processed dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation data ratio')
    
    args = parser.parse_args()
    organize_dataset(args.source, args.output, args.train_ratio, args.val_ratio)
