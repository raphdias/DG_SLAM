#!/usr/bin/env python3
"""
Add controlled noise to semantic masks to test robustness
"""
import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import argparse

def add_noise_to_mask(mask, noise_percent):
    """
    Add random noise to a binary mask
    noise_percent: percentage of pixels to flip (0-100)
    """
    h, w = mask.shape
    total_pixels = h * w
    pixels_to_flip = int(total_pixels * noise_percent / 100)
    
    # Create noisy mask
    noisy_mask = mask.copy()
    
    # Random pixel coordinates to flip
    flip_coords = np.random.choice(total_pixels, pixels_to_flip, replace=False)
    
    for coord in flip_coords:
        y = coord // w
        x = coord % w
        # Flip the pixel (0->255 or 255->0)
        noisy_mask[y, x] = 255 - noisy_mask[y, x]
    
    return noisy_mask

def process_dataset(input_dir, output_dir, noise_percent):
    """
    Add noise to all masks in a directory
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all mask files
    mask_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    
    print(f"Processing {len(mask_files)} masks with {noise_percent}% noise")
    
    for mask_file in mask_files:
        # Load mask
        mask_path = os.path.join(input_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Add noise
        noisy_mask = add_noise_to_mask(mask, noise_percent)
        
        # Save noisy mask
        output_path = os.path.join(output_dir, mask_file)
        cv2.imwrite(output_path, noisy_mask)
    
    print(f"Saved noisy masks to {output_dir}")
    
    # Calculate and print statistics
    original_mask = cv2.imread(os.path.join(input_dir, mask_files[0]), cv2.IMREAD_GRAYSCALE)
    noisy_mask = cv2.imread(os.path.join(output_dir, mask_files[0]), cv2.IMREAD_GRAYSCALE)
    
    diff = np.sum(original_mask != noisy_mask)
    total = original_mask.shape[0] * original_mask.shape[1]
    actual_noise = (diff / total) * 100
    
    print(f"Actual noise added: {actual_noise:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=int, default=20, help='Noise percentage (0-100)')
    parser.add_argument('--input', type=str, 
                       default='data/TUM/rgbd_dataset_freiburg3_walking_xyz/seg_mask',
                       help='Input mask directory')
    args = parser.parse_args()
    
    # Create output directory name
    output_dir = f"{args.input}_noise{args.noise}"
    
    process_dataset(args.input, output_dir, args.noise)