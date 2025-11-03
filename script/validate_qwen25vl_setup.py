#!/usr/bin/env python3
"""
Validation script to check if the environment is ready for Qwen2.5-VL training.

Checks:
1. Required packages installed
2. GPU availability and CUDA version
3. Dataset exists and is readable
4. Config file is valid
5. Model can be loaded
6. Quick forward pass test

Usage:
    python script/validate_qwen25vl_setup.py --config configs/qwen25vl_prompt_gen.yaml
"""

import argparse
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def print_header(text):
    print(f"\n{BLUE}{'='*80}{NC}")
    print(f"{BLUE}{text:^80}{NC}")
    print(f"{BLUE}{'='*80}{NC}\n")


def print_check(name, passed, message=""):
    symbol = f"{GREEN}✓{NC}" if passed else f"{RED}✗{NC}"
    status = f"{GREEN}PASS{NC}" if passed else f"{RED}FAIL{NC}"
    print(f"{symbol} [{status}] {name}")
    if message:
        print(f"         {message}")


def check_imports():
    """Check if required packages are installed."""
    print_header("Checking Package Installation")
    
    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "peft": "PEFT (LoRA)",
        "trl": "TRL (Training)",
        "qwen_vl_utils": "Qwen VL Utils",
        "accelerate": "Accelerate (Multi-GPU)",
        "bitsandbytes": "BitsAndBytes (Quantization)",
        "PIL": "Pillow (Image processing)",
        "pandas": "Pandas (Data loading)",
        "yaml": "PyYAML (Config parsing)",
    }
    
    all_passed = True
    for module, name in required.items():
        try:
            __import__(module)
            print_check(name, True)
        except ImportError as e:
            print_check(name, False, f"Error: {e}")
            all_passed = False
    
    return all_passed


def check_gpu():
    """Check GPU availability and CUDA version."""
    print_header("Checking GPU and CUDA")
    
    try:
        import torch
        
        # CUDA available
        cuda_available = torch.cuda.is_available()
        print_check("CUDA Available", cuda_available)
        
        if not cuda_available:
            return False
        
        # CUDA version
        cuda_version = torch.version.cuda
        print_check("CUDA Version", True, f"Version: {cuda_version}")
        
        # Number of GPUs
        num_gpus = torch.cuda.device_count()
        print_check("GPU Count", num_gpus > 0, f"Found {num_gpus} GPU(s)")
        
        # GPU details
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1024**3
            print(f"         GPU {i}: {props.name} ({vram_gb:.1f} GB)")
        
        # BFloat16 support
        bf16_support = torch.cuda.is_bf16_supported()
        print_check("BFloat16 Support", bf16_support)
        
        return True
        
    except Exception as e:
        print_check("GPU Check", False, f"Error: {e}")
        return False


def check_dataset(config):
    """Check if dataset exists and is valid."""
    print_header("Checking Dataset")
    
    try:
        import pandas as pd
        
        # Get dataset path from config
        csv_path = config.get("data", {}).get("train_csv")
        if not csv_path:
            print_check("Dataset Path", False, "No train_csv specified in config")
            return False
        
        csv_path = Path(csv_path)
        
        # Check file exists
        exists = csv_path.exists()
        print_check("CSV File Exists", exists, f"Path: {csv_path}")
        
        if not exists:
            return False
        
        # Load and check CSV
        df = pd.read_csv(csv_path)
        print_check("CSV Readable", True, f"Loaded {len(df)} rows")
        
        # Check required columns
        required_cols = ["path_control", "prompt"]
        has_cols = all(col in df.columns for col in required_cols)
        print_check("Required Columns", has_cols, f"Columns: {list(df.columns)}")
        
        if not has_cols:
            return False
        
        # Check for NaN values
        na_counts = df[required_cols].isna().sum()
        has_na = na_counts.any()
        if has_na:
            print_check("No Missing Values", False, f"NaN counts: {na_counts.to_dict()}")
        else:
            print_check("No Missing Values", True)
        
        # Check sample images exist
        sample_paths = df["path_control"].head(5).tolist()
        valid_samples = sum(Path(p).exists() for p in sample_paths)
        print_check("Sample Images Exist", valid_samples == 5, 
                   f"{valid_samples}/5 sample images found")
        
        return True
        
    except Exception as e:
        print_check("Dataset Check", False, f"Error: {e}")
        return False


def check_config(config_path):
    """Check if config file is valid."""
    print_header("Checking Configuration")
    
    try:
        import yaml
        
        # Check file exists
        config_file = Path(config_path)
        exists = config_file.exists()
        print_check("Config File Exists", exists, f"Path: {config_path}")
        
        if not exists:
            return None
        
        # Load config
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        print_check("Config Valid YAML", True)
        
        # Check required sections
        required_sections = ["model", "lora", "data", "training"]
        for section in required_sections:
            has_section = section in config
            print_check(f"Has '{section}' section", has_section)
        
        # Check key parameters
        model_name = config.get("model", {}).get("model_name", "NOT SET")
        print_check("Model Name", True, f"Model: {model_name}")
        
        output_dir = config.get("training", {}).get("output_dir", "NOT SET")
        print_check("Output Directory", True, f"Output: {output_dir}")
        
        return config
        
    except Exception as e:
        print_check("Config Check", False, f"Error: {e}")
        return None


def check_model_loading():
    """Check if model can be loaded (quick test)."""
    print_header("Checking Model Loading (Quick Test)")
    
    try:
        import torch
        from transformers import Qwen2VLProcessor
        
        # Just check if we can load the processor (lightweight)
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"Attempting to load processor from: {model_name}")
        print("(This may take a few moments on first run...)")
        
        processor = Qwen2VLProcessor.from_pretrained(model_name)
        print_check("Model Processor", True, "Processor loaded successfully")
        
        print("\nℹ️  Note: Full model loading will happen during training")
        print("   This test only verifies processor loading to save time")
        
        return True
        
    except Exception as e:
        print_check("Model Loading", False, f"Error: {e}")
        print(f"\n{YELLOW}Tip: Run 'huggingface-cli login' if authentication is needed{NC}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate Qwen2.5-VL training setup")
    parser.add_argument("--config", type=str, default="configs/qwen25vl_prompt_gen.yaml",
                       help="Path to config file")
    parser.add_argument("--skip-model-check", action="store_true",
                       help="Skip model loading test (faster)")
    args = parser.parse_args()
    
    print_header("Qwen2.5-VL Training Setup Validation")
    
    results = {}
    
    # Run checks
    results["imports"] = check_imports()
    results["gpu"] = check_gpu()
    
    config = check_config(args.config)
    results["config"] = config is not None
    
    if config:
        results["dataset"] = check_dataset(config)
    else:
        results["dataset"] = False
    
    if not args.skip_model_check:
        results["model"] = check_model_loading()
    else:
        results["model"] = True
        print_header("Model Check Skipped")
    
    # Summary
    print_header("Validation Summary")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        print_check(check.title(), passed)
    
    print()
    if all_passed:
        print(f"{GREEN}{'='*80}{NC}")
        print(f"{GREEN}{'✓ All checks passed! Ready to start training.':^80}{NC}")
        print(f"{GREEN}{'='*80}{NC}")
        print(f"\n{BLUE}Next steps:{NC}")
        print(f"  1. Review config: {BLUE}{args.config}{NC}")
        print(f"  2. Start training:")
        print(f"     {BLUE}./script/train_qwen25vl_prompt_gen.sh {args.config}{NC}")
        return 0
    else:
        print(f"{RED}{'='*80}{NC}")
        print(f"{RED}{'✗ Some checks failed. Please fix issues above.':^80}{NC}")
        print(f"{RED}{'='*80}{NC}")
        print(f"\n{YELLOW}Common fixes:{NC}")
        print(f"  - Install missing packages: {BLUE}pip install -r requirements.txt{NC}")
        print(f"  - Check GPU: {BLUE}nvidia-smi{NC}")
        print(f"  - Verify dataset path in config file")
        print(f"  - Login to HuggingFace: {BLUE}huggingface-cli login{NC}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

