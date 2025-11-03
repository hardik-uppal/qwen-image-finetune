# Run Tracking and Naming System

This document explains the **consistent run naming and tracking system** for Qwen2.5-VL prompt generation training.

## Problem Statement

Previously, it was difficult to correlate:
- Output directories on disk (e.g., `/skynas/.../qwen25vl-prompt-gen-lora-v2`)
- Wandb run names (auto-generated random names)
- Training configurations used
- Checkpoint completeness status

This made it challenging to:
- Find which checkpoint corresponds to which wandb run
- Resume specific training runs
- Track experiment results across multiple runs
- Clean up incomplete or failed training runs

## Solution Overview

The new system provides:

1. **Consistent Run Naming**: Single run name used for both output directory and wandb
2. **Automatic Metadata Saving**: Links checkpoint directories to wandb runs
3. **Directory Structure**: Organized hierarchy for easy navigation
4. **Inspection Tools**: Utilities to scan and report on existing runs

## Directory Structure

### New Structure (Recommended)

```
/skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora/
├── qwen25vl_prompt_gen_20251031_143022/    # Auto-generated run name
│   ├── run_metadata.json                    # Wandb link + config info
│   ├── training_config.yaml                 # Full training config
│   ├── adapter_config.json                  # LoRA adapter config
│   ├── adapter_model.safetensors            # LoRA weights
│   ├── checkpoint-500/                      # Intermediate checkpoints
│   │   └── ...
│   └── checkpoint-1000/
│       └── ...
├── experiment_v1_high_lr/                   # Custom run name
│   ├── run_metadata.json
│   └── ...
└── experiment_v2_low_lr/
    ├── run_metadata.json
    └── ...
```

### Old Structure (Still Supported)

```
/skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora/
├── checkpoint-500/
├── checkpoint-1000/
├── adapter_config.json
└── ...

/skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora-v2/
├── checkpoint-500/
└── ...
```

## Usage

### 1. Training with Run Names

#### Option A: Auto-generated Run Name (Default)

```bash
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml
```

**Result**: Creates `qwen25vl_prompt_gen_20251031_143022/` (config name + timestamp)

#### Option B: Custom Run Name via CLI (Recommended)

```bash
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml \
    --run-name experiment_v1_high_lr_r64
```

**Result**: Creates `experiment_v1_high_lr_r64/`

This is **recommended** for experiments because:
- You can use semantic names (e.g., `baseline`, `high_lr`, `r128`)
- Easy to remember and reference
- Better for team collaboration

#### Option C: Run Name in Config

Edit `configs/qwen25vl_prompt_gen.yaml`:

```yaml
training:
  output_dir: /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora
  run_name: experiment_baseline  # Add this line
```

Then run:

```bash
python script/train_qwen25vl_prompt_generation.py \
    --config configs/qwen25vl_prompt_gen.yaml
```

**Result**: Creates `experiment_baseline/`

### 2. Run Metadata File

After training starts, `run_metadata.json` is automatically created:

```json
{
  "run_name": "experiment_v1_high_lr",
  "output_dir": "/skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora/experiment_v1_high_lr",
  "config_file": "configs/qwen25vl_prompt_gen.yaml",
  "created_at": "2025-10-31T14:30:22.123456",
  "hostname": "dgx-server-01",
  "config_summary": {
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "lora_r": 32,
    "lora_alpha": 16,
    "num_epochs": 10,
    "learning_rate": 2e-05,
    "batch_size": 2
  },
  "wandb": {
    "run_id": "abc123def456",
    "run_name": "experiment_v1_high_lr",
    "run_url": "https://wandb.ai/your-entity/qwen25vl-prompt-gen/runs/abc123def456",
    "project": "qwen25vl-prompt-gen",
    "entity": "your-entity"
  }
}
```

**Key Features**:
- Direct link to wandb run URL
- Config summary for quick reference
- Timestamp and hostname for debugging
- Complete traceability

### 3. Inspecting Existing Runs

Use the `inspect_runs.py` utility to scan and report on training runs:

#### Basic Usage

```bash
# Scan a directory for all runs
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora
```

**Output**:
```
================================================================================
Training Runs Report
================================================================================

Found 3 training run(s)

[1] experiment_v1_high_lr
    Status: ✓ Complete
    Directory: /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora/experiment_v1_high_lr
    Created: 2025-10-31T14:30:22.123456
    Wandb Run:
      • Name: experiment_v1_high_lr
      • ID: abc123def456
      • URL: https://wandb.ai/your-entity/qwen25vl-prompt-gen/runs/abc123def456
      • Project: your-entity/qwen25vl-prompt-gen
    Checkpoints: 5 found

[2] experiment_v2_low_lr
    Status: ⋯ In Progress
    Directory: /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora/experiment_v2_low_lr
    Created: 2025-10-31T15:00:00.000000
    Wandb Run:
      • Name: experiment_v2_low_lr
      • URL: https://wandb.ai/your-entity/qwen25vl-prompt-gen/runs/xyz789
    Checkpoints: 2 found
    Missing files:
      • adapter_config.json
      • adapter_model.safetensors

[3] qwen25vl_prompt_gen_20251030_120000
    Status: ✗ Incomplete
    Directory: /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora/qwen25vl_prompt_gen_20251030_120000
    ⚠ No metadata file found
    Missing files:
      • adapter_config.json
      • adapter_model.safetensors

Summary:
  Complete: 1
  In Progress: 1
  Incomplete: 1
```

#### Advanced Options

```bash
# Show only incomplete runs
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --incomplete-only

# Filter by run name pattern
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --filter experiment_v

# Verbose output (show all files and checkpoints)
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --verbose

# Export to JSON for programmatic access
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --output runs_report.json
```

### 4. Finding Checkpoint by Wandb Run

**Scenario**: You found a good run in wandb, now you want to find the checkpoint.

1. Open wandb run page (e.g., `https://wandb.ai/your-entity/project/runs/abc123`)
2. Note the run name (e.g., `experiment_v1_high_lr`)
3. Run the inspector:

```bash
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --filter experiment_v1_high_lr
```

The output will show you the exact directory path and checkpoint status.

### 5. Finding Wandb Run by Checkpoint

**Scenario**: You have a checkpoint directory, now you want to find the wandb run.

```bash
# Option 1: Check the metadata file directly
cat /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora/experiment_v1_high_lr/run_metadata.json

# Option 2: Use the inspector
python script/inspect_runs.py /skynas/01/hardik/qwen-lora/qwen25vl-prompt-gen-lora \
    --filter experiment_v1_high_lr --verbose
```

The wandb URL will be displayed directly.

## Run Completion Status

The inspector automatically determines run status:

### ✓ Complete
- Has `adapter_config.json`
- Has `adapter_model.safetensors` OR `adapter_model.bin`
- Training finished successfully

### ⋯ In Progress
- Has checkpoints (e.g., `checkpoint-500/`)
- Missing final adapter files
- Training may still be running or was interrupted

### ✗ Incomplete
- Missing adapter files
- No or few checkpoints
- Training likely failed early

## Best Practices

### For Individual Experiments

Use descriptive run names:

```bash
# Good examples
--run-name baseline_r32_lr2e5
--run-name experiment_high_dropout_0.1
--run-name ablation_no_warmup
--run-name prod_v1_20k_steps

# Bad examples (hard to remember)
--run-name test1
--run-name abc
--run-name run_20231031
```

### For Systematic Experiments

Create a naming convention:

```bash
# Format: <experiment_id>_<variable>_<value>
--run-name exp001_lr_1e5
--run-name exp001_lr_2e5
--run-name exp001_lr_5e5

--run-name exp002_r_16
--run-name exp002_r_32
--run-name exp002_r_64
```

### For Production Training

Include version and purpose:

```bash
--run-name prod_v1_full_dataset
--run-name prod_v2_improved_prompts
--run-name prod_v3_longer_training
```

## Migration from Old System

If you have existing runs without metadata:

1. **Check wandb history**: Look at past wandb runs and note their timestamps
2. **Match by timestamp**: Compare wandb run start time with directory modification time
3. **Manually create metadata** (optional):

```bash
# Create run_metadata.json manually
cat > /path/to/old_run/run_metadata.json << 'EOF'
{
  "run_name": "legacy_run_oct30",
  "output_dir": "/path/to/old_run",
  "created_at": "2025-10-30T12:00:00",
  "notes": "Migrated from old system",
  "wandb": {
    "run_url": "https://wandb.ai/your-entity/project/runs/old_run_id"
  }
}
EOF
```

## Troubleshooting

### Run metadata file not created

**Problem**: `run_metadata.json` is missing after training.

**Causes**:
- Training crashed before metadata was saved
- Wandb not initialized properly

**Solution**:
- Metadata is saved after `trainer.train()` starts
- Check wandb logs for initialization errors
- Ensure `report_to: wandb` in config

### Multiple runs with same name

**Problem**: Accidentally used same run name twice.

**Solution**:
- The output directory handler will prompt you to:
  - Resume from checkpoint
  - Overwrite (delete old)
  - Create new version (appends `-v2`)
- In `--non-interactive` mode, automatically creates versioned directory

### Can't find wandb run

**Problem**: Metadata exists but wandb URL is broken.

**Causes**:
- Wandb project was renamed/deleted
- Wandb account changed

**Solution**:
- Search wandb by run_id (in metadata file)
- Check other wandb projects
- Use wandb CLI: `wandb runs list your-entity/project`

## Technical Details

### Run Name Priority

The run name is determined in this order:

1. CLI argument: `--run-name`
2. Config file: `training.run_name`
3. Auto-generated: `{config_name}_{timestamp}`

### Wandb Integration

- Run name set via `TrainingArguments(run_name=...)`
- Wandb run is initialized by HuggingFace Trainer
- Metadata saved after `trainer.train()` starts
- Requires `report_to: wandb` in training config

### File Checks

The inspector checks for these files:

**Required for complete run**:
- `adapter_config.json` (LoRA config)
- `adapter_model.safetensors` OR `adapter_model.bin` (weights)

**Optional but recommended**:
- `training_config.yaml` (full config)
- `run_metadata.json` (wandb link)
- `checkpoint-*/` (intermediate checkpoints)

## Summary

The new run tracking system provides:

✅ **Consistent naming** between directories and wandb runs  
✅ **Automatic metadata** linking checkpoints to wandb  
✅ **Easy inspection** of all training runs  
✅ **Clear status** (complete/in-progress/incomplete)  
✅ **Better organization** with hierarchical directories  
✅ **Team-friendly** with shareable run names  

Use `--run-name` for experiments and `inspect_runs.py` to manage your training runs effectively!

