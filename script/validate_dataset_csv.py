#!/usr/bin/env python3
"""
Validate dataset CSV for common issues that cause training anomalies.
"""

import pandas as pd
import logging
from pathlib import Path
import argparse
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_csv(csv_path: str):
    """Validate CSV dataset for common issues."""
    print(f"\n{'='*60}")
    print(f"Validating CSV: {csv_path}")
    print(f"{'='*60}\n")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        return False
    
    # Check required columns
    required_cols = ['path_target', 'prompt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        print(f"  Available columns: {df.columns.tolist()}")
        return False
    print(f"✓ Required columns present: {required_cols}")
    
    # Check for NaN
    nan_count = df[required_cols].isna().sum()
    if nan_count.any():
        print(f"⚠ NaN values found:")
        for col, count in nan_count.items():
            if count > 0:
                print(f"  - {col}: {count} NaN values")
                # Show first few rows with NaN
                nan_rows = df[df[col].isna()].index.tolist()[:3]
                print(f"    Example rows: {nan_rows}")
    else:
        print(f"✓ No NaN values in required columns")
    
    # Check control columns
    control_cols = [col for col in df.columns if 'path_control' in col]
    if len(control_cols) == 0:
        print(f"✗ No control columns found (expected path_control or path_control_1, etc.)")
        return False
    print(f"✓ Found {len(control_cols)} control columns: {control_cols}")
    
    # Check for empty control images in each row
    rows_without_controls = 0
    for idx, row in df.iterrows():
        has_control = any(pd.notna(row.get(col)) for col in control_cols)
        if not has_control:
            rows_without_controls += 1
            if rows_without_controls <= 5:
                print(f"  Row {idx}: No control images")
    
    if rows_without_controls > 0:
        print(f"⚠ {rows_without_controls} rows have no control images")
    else:
        print(f"✓ All rows have at least one control image")
    
    # Validate file paths
    print(f"\nValidating file paths...")
    missing_targets = 0
    missing_controls = 0
    corrupt_images = 0
    
    for idx, row in df.iterrows():
        # Check target
        if pd.notna(row['path_target']):
            target_path = Path(row['path_target'])
            if not target_path.exists():
                missing_targets += 1
                if missing_targets <= 5:  # Only print first 5
                    print(f"  Row {idx}: Missing target - {row['path_target']}")
            else:
                # Try to open image
                try:
                    img = Image.open(target_path)
                    img.verify()
                except Exception as e:
                    corrupt_images += 1
                    if corrupt_images <= 5:
                        print(f"  Row {idx}: Corrupt image - {row['path_target']}: {e}")
        
        # Check controls
        for col in control_cols:
            if col in row and pd.notna(row[col]):
                control_path = Path(row[col])
                if not control_path.exists():
                    missing_controls += 1
                    if missing_controls <= 5:
                        print(f"  Row {idx}: Missing control - {row[col]}")
    
    print(f"\nFile Validation Results:")
    print(f"  Missing targets: {missing_targets}")
    print(f"  Missing controls: {missing_controls}")
    print(f"  Corrupt images: {corrupt_images}")
    
    if missing_targets + missing_controls + corrupt_images == 0:
        print(f"✓ All files exist and are readable")
    else:
        print(f"⚠ Found {missing_targets + missing_controls + corrupt_images} file issues")
    
    # Check mask column if it exists
    if 'path_mask' in df.columns:
        mask_count = df['path_mask'].notna().sum()
        print(f"\nMask Information:")
        print(f"  Rows with masks: {mask_count} / {len(df)} ({100*mask_count/len(df):.1f}%)")
        
        missing_masks = 0
        for idx, row in df.iterrows():
            if pd.notna(row['path_mask']):
                if not Path(row['path_mask']).exists():
                    missing_masks += 1
        if missing_masks > 0:
            print(f"  ⚠ Missing mask files: {missing_masks}")
    
    # Check image statistics
    print(f"\nChecking image statistics (sampling first 50 valid images)...")
    widths, heights, aspects = [], [], []
    
    for idx, row in df.head(100).iterrows():  # Check first 100 to get 50 valid
        if pd.notna(row['path_target']) and Path(row['path_target']).exists():
            try:
                img = Image.open(row['path_target'])
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w/h)
                if len(widths) >= 50:
                    break
            except:
                pass
    
    if len(widths) > 0:
        print(f"  Sampled {len(widths)} images")
        print(f"  Width range: {min(widths)} - {max(widths)} (mean: {np.mean(widths):.0f}, std: {np.std(widths):.0f})")
        print(f"  Height range: {min(heights)} - {max(heights)} (mean: {np.mean(heights):.0f}, std: {np.std(heights):.0f})")
        print(f"  Aspect ratio range: {min(aspects):.2f} - {max(aspects):.2f} (mean: {np.mean(aspects):.2f})")
        
        # Check for extreme variations
        if max(widths) / (min(widths) + 1) > 3 or max(heights) / (min(heights) + 1) > 3:
            print(f"  ⚠ LARGE variation in image sizes detected - may cause batch inconsistency!")
            print(f"    Consider resizing all images to similar dimensions")
        
        if np.std(aspects) > 0.5:
            print(f"  ⚠ High variation in aspect ratios - may affect training")
    else:
        print(f"  ✗ Could not sample any valid images")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Validation Summary")
    print(f"{'='*60}")
    
    issues = []
    if nan_count.any():
        issues.append(f"{nan_count.sum()} NaN values")
    if missing_targets > 0:
        issues.append(f"{missing_targets} missing target images")
    if missing_controls > 0:
        issues.append(f"{missing_controls} missing control images")
    if corrupt_images > 0:
        issues.append(f"{corrupt_images} corrupt images")
    if rows_without_controls > 0:
        issues.append(f"{rows_without_controls} rows without controls")
    
    if issues:
        print(f"⚠ Found {len(issues)} types of issues:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\n💡 Tip: Use skip_on_error: true in dataset config to skip problematic samples")
    else:
        print(f"✓ No critical issues found! Dataset looks good.")
    
    print(f"{'='*60}\n")
    
    return len(issues) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset CSV for training")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more details")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = validate_csv(args.csv_path)
    exit(0 if success else 1)

