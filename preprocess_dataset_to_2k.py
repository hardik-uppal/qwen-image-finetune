#!/usr/bin/env python3
"""
Preprocess entire dataset by resizing all images to 2K max dimension.
This eliminates memory issues and allows training on 100% of the data.
"""
import pandas as pd
import PIL.Image
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format='%(message)s')

MAX_DIMENSION = 2000
OUTPUT_BASE = "/skynas/01/hardik/qwen-lora/dataset-ryan"
NUM_WORKERS = 32  # Use more workers for faster processing


def create_unique_filename(original_path):
    """
    Create a unique filename that won't collide with others.
    Uses hash of the full path + original filename for uniqueness and readability.
    """
    # Get the original filename and extension
    path_obj = Path(original_path)
    stem = path_obj.stem
    ext = path_obj.suffix
    
    # Create a short hash from the full path (8 chars is enough for uniqueness)
    path_hash = hashlib.md5(str(original_path).encode()).hexdigest()[:8]
    
    # Combine: hash_originalname.ext
    unique_name = f"{path_hash}_{stem}{ext}"
    
    return unique_name


def resize_and_save_image(args):
    """Resize a single image and save to new location."""
    src_path, dst_path, max_dim = args
    
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Skip if already exists
        if os.path.exists(dst_path):
            return {'success': True, 'src': src_path, 'dst': dst_path, 'skipped': True}
        
        # Load image
        with PIL.Image.open(src_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get current size
            w, h = img.size
            max_current = max(w, h)
            
            # Resize if needed
            if max_current > max_dim:
                scale = max_dim / max_current
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), PIL.Image.LANCZOS)
            
            # Save with good quality
            img.save(dst_path, 'JPEG', quality=95, optimize=True)
        
        return {'success': True, 'src': src_path, 'dst': dst_path, 'skipped': False}
    
    except Exception as e:
        return {'success': False, 'src': src_path, 'error': str(e)}


def preprocess_dataset(csv_path, output_base, max_dimension=2000, num_workers=16):
    """Preprocess entire dataset by resizing all images."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded CSV with {len(df)} rows")
    
    # Get all control columns (path_control, path_control_1, path_control_2, etc.)
    control_columns = [col for col in df.columns if col.startswith('path_control')]
    logging.info(f"Found control columns: {control_columns}")
    
    # Collect all image paths to process
    tasks = []
    new_rows = []
    
    for idx, row in df.iterrows():
        # Create new row for output CSV
        new_row = row.copy()
        
        # Process target image
        if pd.notna(row['path_target']):
            src_target = row['path_target']
            # Create unique filename to avoid collisions
            unique_name = create_unique_filename(src_target)
            dst_target = os.path.join(output_base, 'targets', unique_name)
            tasks.append((src_target, dst_target, max_dimension))
            new_row['path_target'] = dst_target
        
        # Process all control images (path_control, path_control_1, etc.)
        for control_col in control_columns:
            if control_col in row and pd.notna(row[control_col]):
                src_control = row[control_col]
                # Create unique filename to avoid collisions
                unique_name = create_unique_filename(src_control)
                # Use column name to organize (controls, controls_1, controls_2, etc.)
                subdir = control_col.replace('path_', '')  # 'control', 'control_1', etc.
                dst_control = os.path.join(output_base, subdir, unique_name)
                tasks.append((src_control, dst_control, max_dimension))
                new_row[control_col] = dst_control
        
        new_rows.append(new_row)
    
    logging.info(f"\nTotal images to process: {len(tasks)}")
    logging.info(f"Output directory: {output_base}")
    logging.info(f"Max dimension: {max_dimension}px")
    logging.info(f"Workers: {num_workers}\n")
    
    # Create output directories
    os.makedirs(os.path.join(output_base, 'targets'), exist_ok=True)
    for control_col in control_columns:
        subdir = control_col.replace('path_', '')
        os.makedirs(os.path.join(output_base, subdir), exist_ok=True)
    
    # Process images in parallel
    logging.info("Processing images with multiprocessing...")
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(resize_and_save_image, tasks, chunksize=50),
            total=len(tasks),
            desc="Resizing images",
            unit="img"
        ))
    
    # Collect statistics
    success_count = sum(1 for r in results if r['success'])
    skipped_count = sum(1 for r in results if r.get('skipped', False))
    error_count = sum(1 for r in results if not r['success'])
    errors = [r for r in results if not r['success']]
    
    logging.info(f"\n{'='*70}")
    logging.info("PREPROCESSING COMPLETE")
    logging.info(f"{'='*70}")
    logging.info(f"  Total images: {len(tasks)}")
    logging.info(f"  Successfully processed: {success_count}")
    logging.info(f"  Skipped (already exist): {skipped_count}")
    logging.info(f"  Errors: {error_count}")
    
    if errors:
        logging.info(f"\nFirst 5 errors:")
        for err in errors[:5]:
            logging.info(f"  {err['src']}: {err['error']}")
    
    # Save new CSV
    output_csv = csv_path.replace('.csv', '_2k.csv')
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_csv, index=False)
    logging.info(f"\nNew CSV saved to: {output_csv}")
    logging.info(f"Total rows: {len(new_df)}")
    
    return output_csv


if __name__ == "__main__":
    # Process training dataset
    train_csv = "workspace/metadata-v12-recaptioned-long/train_filtered.csv"
    logging.info("="*70)
    logging.info("PREPROCESSING TRAINING DATASET")
    logging.info("="*70)
    output_train_csv = preprocess_dataset(
        train_csv, 
        OUTPUT_BASE, 
        MAX_DIMENSION, 
        NUM_WORKERS
    )
    
    # Process validation dataset
    val_csv = "workspace/metadata-v12-recaptioned-long/val_filtered.csv"
    logging.info("\n" + "="*70)
    logging.info("PREPROCESSING VALIDATION DATASET")
    logging.info("="*70)
    output_val_csv = preprocess_dataset(
        val_csv, 
        OUTPUT_BASE, 
        MAX_DIMENSION, 
        NUM_WORKERS
    )
    
    logging.info("\n" + "="*70)
    logging.info("ALL DONE!")
    logging.info("="*70)
    logging.info(f"\nUpdate your config to use:")
    logging.info(f"  Training: {output_train_csv}")
    logging.info(f"  Validation: {output_val_csv}")

