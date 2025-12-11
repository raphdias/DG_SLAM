#!/usr/bin/env python3
"""
Generate dilated semantic masks to simulate varying dynamic object densities.
This tests the hypothesis that performance degrades non-linearly with increased dynamic content.

Usage:
    python generate_density_masks.py --density 25
    python generate_density_masks.py --density 50
    python generate_density_masks.py --density 75
    
Or generate all at once:
    python generate_density_masks.py --all
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def get_mask_coverage(mask):
    """Calculate percentage of frame covered by mask (non-zero pixels)"""
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_pixels = np.sum(mask > 0)
    return (mask_pixels / total_pixels) * 100


def expand_mask_to_percentage(mask, target_percentage):
    """
    Dilate mask until it covers target_percentage of the image.
    
    Args:
        mask: Binary mask (0 or 255)
        target_percentage: Target coverage as percentage (e.g., 25 for 25%)
    
    Returns:
        Dilated mask covering approximately target_percentage of frame
    """
    h, w = mask.shape
    total_pixels = h * w
    target_pixels = int(total_pixels * target_percentage / 100)
    
    current_mask = mask.copy()
    kernel = np.ones((5, 5), np.uint8)
    
    # If mask is empty, create a seed in the center
    if np.sum(current_mask > 0) == 0:
        # Create a small seed region in center
        cy, cx = h // 2, w // 2
        current_mask[cy-10:cy+10, cx-10:cx+10] = 255
    
    iteration = 0
    max_iterations = 500  # Safety limit
    
    while np.sum(current_mask > 0) < target_pixels and iteration < max_iterations:
        current_mask = cv2.dilate(current_mask, kernel, iterations=1)
        iteration += 1
        
        # Stop if we've nearly filled the frame
        if np.sum(current_mask > 0) >= total_pixels * 0.95:
            break
    
    return current_mask


def process_masks_for_density(input_dir, output_dir, target_density):
    """
    Process all masks in input_dir, dilate to target_density, save to output_dir
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mask_files = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
    
    if not mask_files:
        print(f"❌ No mask files found in {input_dir}")
        return None
    
    print(f"📊 Processing {len(mask_files)} masks for {target_density}% density")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    original_coverages = []
    final_coverages = []
    
    for mask_file in mask_files:
        # Load original mask
        mask_path = input_path / mask_file
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"⚠️ Failed to read {mask_file}")
            continue
        
        original_coverage = get_mask_coverage(mask)
        original_coverages.append(original_coverage)
        
        # Expand to target density
        expanded_mask = expand_mask_to_percentage(mask, target_density)
        
        final_coverage = get_mask_coverage(expanded_mask)
        final_coverages.append(final_coverage)
        
        # Save expanded mask
        output_file = output_path / mask_file
        cv2.imwrite(str(output_file), expanded_mask)
    
    # Print statistics
    avg_original = np.mean(original_coverages)
    avg_final = np.mean(final_coverages)
    
    print(f"\n✅ Processing complete!")
    print(f"   Original average coverage: {avg_original:.1f}%")
    print(f"   Final average coverage: {avg_final:.1f}%")
    print(f"   Target was: {target_density}%")
    
    return avg_final


def create_visualization(input_dir, output_dir, sample_idx=0):
    """Create side-by-side visualization of original vs dilated mask"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    mask_files = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
    
    if sample_idx >= len(mask_files):
        return
    
    original = cv2.imread(str(input_path / mask_files[sample_idx]), cv2.IMREAD_GRAYSCALE)
    dilated = cv2.imread(str(output_path / mask_files[sample_idx]), cv2.IMREAD_GRAYSCALE)
    
    # Create colored visualization
    vis_original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    vis_original[:, :, 0] = 0  # Remove blue
    vis_original[:, :, 1] = 0  # Remove green, keep red
    
    vis_dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    vis_dilated[:, :, 0] = 0
    vis_dilated[:, :, 1] = 0
    
    comparison = np.hstack([vis_original, vis_dilated])
    
    vis_dir = output_path.parent / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    density = output_path.name.split('density')[-1]
    cv2.imwrite(str(vis_dir / f'density_{density}_comparison.png'), comparison)
    print(f"📸 Saved visualization to {vis_dir / f'density_{density}_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Generate dilated masks for density experiments')
    parser.add_argument('--density', type=int, choices=[25, 50, 75],
                       help='Target density percentage')
    parser.add_argument('--all', action='store_true',
                       help='Generate all density levels (25, 50, 75)')
    parser.add_argument('--input', type=str,
                       default='data/TUM/rgbd_dataset_freiburg3_walking_xyz/seg_mask',
                       help='Input mask directory (clean YOLO masks)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization images')
    args = parser.parse_args()
    
    if not args.density and not args.all:
        print("❌ Please specify --density or --all")
        parser.print_help()
        return
    
    densities = [25, 50, 75] if args.all else [args.density]
    
    print("="*60)
    print("🚀 Dynamic Object Density Mask Generator")
    print("="*60)
    
    for density in densities:
        print(f"\n{'='*60}")
        print(f"Processing {density}% density")
        print('='*60)
        
        output_dir = f"{args.input}_density{density}"
        avg_coverage = process_masks_for_density(args.input, output_dir, density)
        
        if args.visualize and avg_coverage:
            create_visualization(args.input, output_dir)
    
    print("\n" + "="*60)
    print("🎉 All density masks generated!")
    print("="*60)
    print("\nNext steps - run tests with:")
    for density in densities:
        print(f"  python test_with_semantics.py --mask_dir seg_mask_density{density}")


if __name__ == "__main__":
    main()