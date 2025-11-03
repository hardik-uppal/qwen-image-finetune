# Run Tracking and Naming System - Implementation Summary

## Problem Solved

Previously, you had:
- Output directories like `/skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora`, then `-v2`, `-v3`, etc.
- No way to correlate these with wandb runs
- Incomplete runs missing `adapter.json` and other files
- No easy way to track which experiment is which

## Solution Implemented

### ✅ 1. Consistent Run Naming System

**What**: Single run name used for both output directory AND wandb run name

**How to use**:
```bash
# Option 1: Auto-generated (config_name_timestamp)
python script/train_qwen25vl_prompt_generation.py --config configs/qwen25vl_prompt_gen.yaml

# Option 2: Custom name (RECOMMENDED)
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml \
    --run-name experiment_v1_high_lr

# Option 3: In config file
# Add to configs/qwen25vl_prompt_gen.yaml:
#   training:
#     run_name: my_experiment
```

**Result**: Creates organized structure:
```
/skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora/
├── experiment_v1_high_lr/          # Your custom run name
│   ├── run_metadata.json           # Links to wandb!
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── checkpoint-500/
├── experiment_v2_low_lr/
│   └── ...
└── qwen25vl_prompt_gen_20251031_143022/  # Auto-generated
    └── ...
```

### ✅ 2. Wandb Run Metadata

**What**: Automatic `run_metadata.json` file in each checkpoint directory

**Content**:
```json
{
  "run_name": "experiment_v1_high_lr",
  "output_dir": "/skynas/.../experiment_v1_high_lr",
  "created_at": "2025-10-31T14:30:22",
  "wandb": {
    "run_id": "abc123def456",
    "run_name": "experiment_v1_high_lr",
    "run_url": "https://wandb.ai/your-entity/project/runs/abc123",
    "project": "qwen25vl-prompt-gen",
    "entity": "your-entity"
  },
  "config_summary": {
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "lora_r": 32,
    "learning_rate": 2e-05
  }
}
```

**Benefit**: Direct link from checkpoint → wandb run URL!

### ✅ 3. Inspection Utility

**What**: New script to scan and report on all training runs

**Usage**:
```bash
# Basic scan
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora

# Filter by name
python script/inspect_runs.py /path/to/runs --filter experiment_v1

# Show only incomplete runs
python script/inspect_runs.py /path/to/runs --incomplete-only

# Verbose details
python script/inspect_runs.py /path/to/runs --verbose

# Export to JSON
python script/inspect_runs.py /path/to/runs --output report.json
```

**Output Example**:
```
================================================================================
Training Runs Report
================================================================================

Found 3 training run(s)

[1] experiment_v1_high_lr
    Status: ✓ Complete
    Directory: /skynas/.../experiment_v1_high_lr
    Created: 2025-10-31T14:30:22
    Wandb Run:
      • Name: experiment_v1_high_lr
      • URL: https://wandb.ai/your-entity/project/runs/abc123
      • Project: your-entity/qwen25vl-prompt-gen
    Checkpoints: 5 found

[2] experiment_v2_training
    Status: ⋯ In Progress
    Checkpoints: 2 found
    Missing files:
      • adapter_config.json
      • adapter_model.safetensors

Summary:
  Complete: 1
  In Progress: 1
  Incomplete: 0
```

**Features**:
- Shows completion status (complete/in-progress/incomplete)
- Direct wandb URLs
- Identifies missing files
- Filter and search capabilities
- Colored output for easy reading

## Files Modified

1. **script/train_qwen25vl_prompt_generation.py**
   - Added `--run-name` argument
   - Auto-generates run names if not provided
   - Creates hierarchical output directory structure
   - Saves run metadata with wandb info

2. **configs/qwen25vl_prompt_gen.yaml**
   - Added optional `run_name` parameter with documentation

3. **configs/qwen25vl_prompt_gen_improved.yaml**
   - Added optional `run_name` parameter with documentation

## Files Created

1. **script/inspect_runs.py**
   - Utility to scan and report on training runs
   - Checks completion status
   - Displays wandb links
   - Supports filtering and export

2. **docs/run_tracking.md**
   - Complete documentation of the new system
   - Usage examples
   - Best practices
   - Troubleshooting guide

3. **docs/qwen25vl_prompt_generation_trl.md** (updated)
   - Added Run Tracking section
   - Quick reference for new features

## Workflow Examples

### Starting a New Training Run

```bash
# Good: Use semantic name
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml \
    --run-name baseline_r32_lr2e5

# Also good: Auto-generated timestamp
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml
```

### Finding Checkpoint from Wandb Run

1. Open wandb run (e.g., see great results for "experiment_v1")
2. Note the run name from wandb
3. Run inspector:
```bash
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --filter experiment_v1
```
4. Get exact checkpoint path from output

### Finding Wandb Run from Checkpoint

```bash
# Option 1: Read metadata directly
cat /path/to/checkpoint/run_metadata.json | grep run_url

# Option 2: Use inspector
python script/inspect_runs.py /path/to/base_dir --filter <run_name>
```

### Cleaning Up Incomplete Runs

```bash
# Find incomplete runs
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --incomplete-only

# Review output, then manually delete unwanted directories
```

## Migration Guide

For existing runs without metadata:

1. **Option A**: Keep as-is (they still work)
2. **Option B**: Manually add metadata:
```bash
cat > /path/to/old_run/run_metadata.json << 'EOF'
{
  "run_name": "legacy_run_oct30",
  "output_dir": "/path/to/old_run",
  "notes": "Migrated from old system",
  "wandb": {
    "run_url": "https://wandb.ai/.../runs/<old_run_id>"
  }
}
EOF
```

## Benefits Summary

✅ **No more confusion**: Direct link between checkpoints and wandb runs  
✅ **Better organization**: Clear directory structure with semantic names  
✅ **Easy inspection**: Scan all runs, check status, find wandb links  
✅ **Team-friendly**: Shareable, understandable run names  
✅ **Automatic**: Metadata saved automatically during training  
✅ **Backward compatible**: Old runs still work, new system optional  

## Next Steps

1. **For new training runs**: Use `--run-name` with descriptive names
2. **Check existing runs**: Run `python script/inspect_runs.py <your_base_dir>`
3. **Read full docs**: See `docs/run_tracking.md` for complete guide

## Questions?

- **Q**: Do I need to use run names?
  **A**: No, but recommended. Auto-generated names work fine too.

- **Q**: What about my old checkpoints?
  **A**: They still work! The system is backward compatible.

- **Q**: Can I change a run name after training?
  **A**: Yes, just rename the directory and update `run_metadata.json`.

- **Q**: What if metadata file is missing?
  **A**: Training works without it, but you won't have the wandb link. Create it manually if needed.

## Implementation Details

- **Run name priority**: CLI arg > config file > auto-generated
- **Auto-generated format**: `{config_name}_{timestamp}`
- **Metadata saved after**: `trainer.train()` starts (wandb is initialized)
- **Inspector checks**: `adapter_config.json`, `adapter_model.safetensors/bin`, checkpoints

---

**Implemented**: October 31, 2025  
**Affects**: Qwen2.5-VL prompt generation training workflow  
**Backward Compatible**: Yes ✅

