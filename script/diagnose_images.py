#!/usr/bin/env python3
"""
Diagnostic script to find problematic images that cause hangs during loading.
"""
import json
import sys
from pathlib import Path
from PIL import Image
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_image_load(image_path: str, timeout: float = 3.0) -> tuple[str, bool, str]:
    """
    Test loading a single image with timeout.
    Returns (image_path, success, error_message)
    """
    try:
        start = time.time()
        image = Image.open(image_path)
        image.verify()
        image = Image.open(image_path).convert("RGB")
        load_time = time.time() - start
        
        # Check if it took too long
        if load_time > 2.0:
            return (image_path, False, f"SLOW: {load_time:.2f}s")
        
        return (image_path, True, "")
    except Exception as e:
        return (image_path, False, f"{type(e).__name__}: {str(e)}")


def diagnose_jsonl(jsonl_path: str, max_samples: int = None, batch_size: int = 100):
    """Diagnose images in a JSONL file."""
    logger.info(f"Diagnosing images from: {jsonl_path}")
    
    # Load all image paths
    image_paths = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            sample = json.loads(line)
            image_paths.append(sample["image_path"])
    
    logger.info(f"Found {len(image_paths)} images to test")
    
    problematic_images = []
    
    # Test images in batches with timeout
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(image_paths) + batch_size - 1) // batch_size
            
            logger.info(f"Testing batch {batch_num}/{total_batches} ({len(batch)} images)...")
            
            # Submit all tasks in batch
            futures = {executor.submit(test_image_load, img_path): img_path for img_path in batch}
            
            # Collect results with timeout
            for future in futures:
                img_path = futures[future]
                try:
                    # Wait max 5 seconds per image
                    result = future.result(timeout=5.0)
                    if not result[1]:  # If not successful
                        logger.error(f"❌ PROBLEM: {result[0]}")
                        logger.error(f"   Error: {result[2]}")
                        problematic_images.append(result)
                except TimeoutError:
                    logger.error(f"⏱️ TIMEOUT (5s): {img_path}")
                    problematic_images.append((img_path, False, "TIMEOUT: Hung for 5+ seconds"))
                except Exception as e:
                    logger.error(f"❌ EXCEPTION loading {img_path}: {e}")
                    problematic_images.append((img_path, False, f"Exception: {e}"))
            
            # Progress update
            tested = min(i + batch_size, len(image_paths))
            logger.info(f"Progress: {tested}/{len(image_paths)} ({100*tested/len(image_paths):.1f}%)")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info(f"DIAGNOSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Total images tested: {len(image_paths)}")
    logger.info(f"Problematic images: {len(problematic_images)}")
    
    if problematic_images:
        logger.info("\n" + "="*60)
        logger.info("PROBLEMATIC IMAGES:")
        logger.info("="*60)
        for img_path, success, error in problematic_images:
            logger.info(f"\n{img_path}")
            logger.info(f"  → {error}")
        
        # Save to file
        problem_file = Path("problematic_images.txt")
        with open(problem_file, 'w') as f:
            for img_path, success, error in problematic_images:
                f.write(f"{img_path}\t{error}\n")
        logger.info(f"\n✅ Saved problematic images to: {problem_file}")
    else:
        logger.info("\n✅ All images loaded successfully!")
    
    return problematic_images


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_images.py <jsonl_path> [max_samples]")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    problematic = diagnose_jsonl(jsonl_path, max_samples=max_samples)
    
    sys.exit(0 if not problematic else 1)



