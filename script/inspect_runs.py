#!/usr/bin/env python3
"""
Inspect and report on training run metadata.

This script scans a directory for training runs and displays:
- Run name and timestamp
- Wandb run URL (if available)
- Output directory
- Config used
- Training parameters
- Checkpoint status (complete/incomplete)

Usage:
    # Scan a base directory for all runs
    python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora
    
    # Scan and filter by run name pattern
    python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora --filter experiment_v
    
    # Show only incomplete runs (missing adapter files)
    python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora --incomplete-only
    
    # Export to JSON
    python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora --output runs_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Terminal colors for better readability
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def check_run_completeness(run_dir: Path) -> Dict[str, any]:
    """
    Check if a training run is complete by looking for expected files.
    
    Returns:
        Dict with completion status and missing files
    """
    expected_files = {
        "adapter_config.json": "LoRA adapter config",
        "adapter_model.safetensors": "LoRA adapter weights (safetensors)",
        "adapter_model.bin": "LoRA adapter weights (bin)",  # Alternative format
        "training_config.yaml": "Training configuration",
        "run_metadata.json": "Run metadata with wandb info",
    }
    
    found_files = {}
    missing_files = []
    
    for file, description in expected_files.items():
        file_path = run_dir / file
        if file_path.exists():
            found_files[file] = {
                "description": description,
                "size_mb": file_path.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            }
        else:
            # For adapter weights, check if either format exists
            if file in ["adapter_model.safetensors", "adapter_model.bin"]:
                alt_file = "adapter_model.bin" if file == "adapter_model.safetensors" else "adapter_model.safetensors"
                if (run_dir / alt_file).exists():
                    continue  # One format is enough
            missing_files.append(file)
    
    # Check for checkpoints
    checkpoints = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-") and (run_dir / d).is_dir()]
    
    # Determine completeness
    has_adapter = (run_dir / "adapter_model.safetensors").exists() or (run_dir / "adapter_model.bin").exists()
    has_config = (run_dir / "adapter_config.json").exists()
    
    if has_adapter and has_config:
        status = "complete"
    elif len(checkpoints) > 0:
        status = "training_in_progress"
    else:
        status = "incomplete"
    
    return {
        "status": status,
        "found_files": found_files,
        "missing_files": missing_files,
        "checkpoints": sorted(checkpoints),
        "num_checkpoints": len(checkpoints),
    }


def load_run_metadata(run_dir: Path) -> Optional[Dict]:
    """Load run metadata from run_metadata.json if it exists."""
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"{Colors.WARNING}Warning: Could not parse {metadata_path}: {e}{Colors.ENDC}")
    return None


def scan_directory(base_dir: Path, filter_pattern: Optional[str] = None, incomplete_only: bool = False) -> List[Dict]:
    """
    Scan a directory for training runs and collect metadata.
    
    Args:
        base_dir: Base directory containing training runs
        filter_pattern: Optional pattern to filter run names
        incomplete_only: If True, only return incomplete runs
    
    Returns:
        List of run information dicts
    """
    runs = []
    
    if not base_dir.exists():
        print(f"{Colors.FAIL}Error: Directory does not exist: {base_dir}{Colors.ENDC}")
        return runs
    
    # Check if this directory itself is a run (has run_metadata.json or checkpoints)
    potential_runs = []
    
    # First, check direct subdirectories
    for item in os.listdir(base_dir):
        item_path = base_dir / item
        if item_path.is_dir():
            potential_runs.append(item_path)
    
    # If no subdirectories with metadata, maybe base_dir itself is a run
    if len(potential_runs) == 0 or (base_dir / "run_metadata.json").exists():
        potential_runs = [base_dir]
    
    for run_dir in potential_runs:
        # Skip hidden directories and non-run directories
        if run_dir.name.startswith('.'):
            continue
        
        # Check if this looks like a training run
        has_metadata = (run_dir / "run_metadata.json").exists()
        has_checkpoints = any(d.startswith("checkpoint-") for d in os.listdir(run_dir) if (run_dir / d).is_dir())
        has_adapter = (run_dir / "adapter_config.json").exists()
        
        if not (has_metadata or has_checkpoints or has_adapter):
            continue  # Not a run directory
        
        # Apply filter
        if filter_pattern and filter_pattern not in run_dir.name:
            continue
        
        # Load metadata
        metadata = load_run_metadata(run_dir)
        completeness = check_run_completeness(run_dir)
        
        # Apply incomplete filter
        if incomplete_only and completeness["status"] == "complete":
            continue
        
        run_info = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "metadata": metadata,
            "completeness": completeness,
        }
        
        runs.append(run_info)
    
    return runs


def format_status(status: str) -> str:
    """Format status with colors."""
    if status == "complete":
        return f"{Colors.OKGREEN}✓ Complete{Colors.ENDC}"
    elif status == "training_in_progress":
        return f"{Colors.OKCYAN}⋯ In Progress{Colors.ENDC}"
    else:
        return f"{Colors.WARNING}✗ Incomplete{Colors.ENDC}"


def print_runs_report(runs: List[Dict], verbose: bool = False):
    """Print a formatted report of training runs."""
    if not runs:
        print(f"{Colors.WARNING}No training runs found.{Colors.ENDC}")
        return
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}Training Runs Report{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    print(f"Found {Colors.BOLD}{len(runs)}{Colors.ENDC} training run(s)\n")
    
    for i, run in enumerate(runs, 1):
        print(f"{Colors.BOLD}[{i}] {run['run_name']}{Colors.ENDC}")
        print(f"    Status: {format_status(run['completeness']['status'])}")
        print(f"    Directory: {run['run_dir']}")
        
        # Metadata info
        if run['metadata']:
            metadata = run['metadata']
            if 'created_at' in metadata:
                print(f"    Created: {metadata['created_at']}")
            
            if 'wandb' in metadata:
                wandb_info = metadata['wandb']
                print(f"    {Colors.OKCYAN}Wandb Run:{Colors.ENDC}")
                print(f"      • Name: {wandb_info.get('run_name', 'N/A')}")
                print(f"      • ID: {wandb_info.get('run_id', 'N/A')}")
                print(f"      • URL: {wandb_info.get('run_url', 'N/A')}")
                print(f"      • Project: {wandb_info.get('entity', 'N/A')}/{wandb_info.get('project', 'N/A')}")
            
            if 'config_summary' in metadata and verbose:
                config = metadata['config_summary']
                print(f"    Config:")
                for key, value in config.items():
                    if value is not None:
                        print(f"      • {key}: {value}")
        else:
            print(f"    {Colors.WARNING}⚠ No metadata file found{Colors.ENDC}")
        
        # Completeness info
        comp = run['completeness']
        if comp['num_checkpoints'] > 0:
            print(f"    Checkpoints: {comp['num_checkpoints']} found")
            if verbose:
                for ckpt in comp['checkpoints'][:3]:  # Show first 3
                    print(f"      • {ckpt}")
                if len(comp['checkpoints']) > 3:
                    print(f"      • ... and {len(comp['checkpoints']) - 3} more")
        
        if comp['missing_files']:
            print(f"    {Colors.WARNING}Missing files:{Colors.ENDC}")
            for file in comp['missing_files']:
                print(f"      • {file}")
        
        if verbose and comp['found_files']:
            print(f"    Found files:")
            for file, info in comp['found_files'].items():
                print(f"      • {file} ({info['size_mb']:.2f} MB)")
        
        print()  # Blank line between runs
    
    # Summary statistics
    complete = sum(1 for r in runs if r['completeness']['status'] == 'complete')
    in_progress = sum(1 for r in runs if r['completeness']['status'] == 'training_in_progress')
    incomplete = sum(1 for r in runs if r['completeness']['status'] == 'incomplete')
    
    print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"  Complete: {Colors.OKGREEN}{complete}{Colors.ENDC}")
    print(f"  In Progress: {Colors.OKCYAN}{in_progress}{Colors.ENDC}")
    print(f"  Incomplete: {Colors.WARNING}{incomplete}{Colors.ENDC}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect training run directories and metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing training runs"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter runs by name pattern"
    )
    parser.add_argument(
        "--incomplete-only",
        action="store_true",
        help="Show only incomplete runs"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Export report to JSON file"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    runs = scan_directory(base_dir, filter_pattern=args.filter, incomplete_only=args.incomplete_only)
    
    # Print report
    print_runs_report(runs, verbose=args.verbose)
    
    # Export to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"{Colors.OKGREEN}✓ Report exported to: {output_path}{Colors.ENDC}")


if __name__ == "__main__":
    main()

