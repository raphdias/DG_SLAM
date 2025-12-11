#!/usr/bin/env python3
"""
Generate semantic masks for DG-SLAM using YOLOv8-seg
This provides high-quality person segmentation masks for motion filtering
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from ultralytics import YOLO
import sys

def generate_masks_for_sequence(sequence_path, model=None, visualize_samples=False):
    """Generate person segmentation masks for a sequence"""
    
    if model is None:
        print("Loading YOLOv8x-seg model (downloading ~137MB on first run)...")
        model = YOLO('yolov8x-seg.pt')
        print("✓ Model loaded")
    
    sequence_path = Path(sequence_path)
    rgb_dir = sequence_path / 'rgb'
    output_dir = sequence_path / 'seg_mask'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all RGB images
    rgb_files = sorted(list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg')))
    
    if not rgb_files:
        print(f"❌ No images found in {rgb_dir}")
        return None
    
    print(f"📸 Processing {len(rgb_files)} images from {sequence_path.name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Process all images
    mask_count = 0
    for idx, rgb_file in enumerate(tqdm(rgb_files, desc="Generating masks")):
        # Read image
        img = cv2.imread(str(rgb_file))
        if img is None:
            print(f"⚠️ Failed to read {rgb_file}")
            continue
        h, w = img.shape[:2]
        
        # Run YOLO inference (verbose=False for cleaner output)
        results = model(str(rgb_file), verbose=False)
        
        # Create binary mask for persons (class 0 in COCO)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for r in results:
            if r.masks is not None and r.boxes is not None:
                for i, cls in enumerate(r.boxes.cls):
                    if int(cls) == 0:  # Person class in COCO
                        # Get segmentation mask for this person
                        if i < len(r.masks.data):
                            person_mask = r.masks.data[i].cpu().numpy()
                            # Resize mask to original image size
                            person_mask = cv2.resize(person_mask.astype(np.float32), (w, h))
                            # Convert to binary (255 for person, 0 for background)
                            mask = np.maximum(mask, (person_mask > 0.5).astype(np.uint8) * 255)
                            mask_count += 1
        
        # Optional: Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Save mask with same filename as RGB image
        mask_file = output_dir / rgb_file.name
        cv2.imwrite(str(mask_file), mask)
        
        # Optionally visualize first few results
        if visualize_samples and idx < 3:
            vis_dir = sequence_path / 'visualization'
            vis_dir.mkdir(exist_ok=True)
            # Create side-by-side comparison
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:,:,0] = 0  # Remove blue channel for red mask
            comparison = np.hstack([img, mask_colored])
            cv2.imwrite(str(vis_dir / f'comparison_{idx:03d}.png'), comparison)
    
    print(f"✅ Successfully saved {len(rgb_files)} masks to {output_dir}")
    print(f"   Found persons in {mask_count} detections")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Generate semantic masks for DG-SLAM using YOLOv8')
    parser.add_argument('--data-root', type=str, 
                       default='/home/ubuntu/projects/DG-SLAM/data',
                       help='Root data directory')
    parser.add_argument('--dataset', type=str, default='TUM', 
                       choices=['TUM', 'BONN'],
                       help='Dataset type')
    parser.add_argument('--sequence', type=str, 
                       default='rgbd_dataset_freiburg3_walking_xyz',
                       help='Sequence name (or "all" for all sequences)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization of first few frames')
    args = parser.parse_args()
    
    # Load model once for efficiency
    print("🚀 Initializing YOLOv8x-seg model...")
    model = YOLO('yolov8x-seg.pt')
    print("✓ Model ready")
    
    if args.sequence == 'all':
        # Process all sequences in the dataset
        dataset_path = Path(args.data_root) / args.dataset
        sequences = [d for d in dataset_path.iterdir() if d.is_dir() and (d / 'rgb').exists()]
        print(f"Found {len(sequences)} sequences to process")
        
        for seq_path in sequences:
            print(f"\n{'='*60}")
            print(f"Processing {seq_path.name}")
            print('='*60)
            generate_masks_for_sequence(seq_path, model, args.visualize)
    else:
        # Process single sequence
        sequence_path = Path(args.data_root) / args.dataset / args.sequence
        if not sequence_path.exists():
            print(f"❌ Error: Sequence path {sequence_path} does not exist")
            sys.exit(1)
        
        generate_masks_for_sequence(sequence_path, model, args.visualize)
    
    print("\n" + "="*60)
    print("🎉 All done! Semantic masks generated successfully.")
    print("="*60)

if __name__ == "__main__":
    main()