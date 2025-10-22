# Wandb Logging Implementation Summary

## Overview
Successfully implemented comprehensive wandb logging for the Qwen Image Edit Plus trainer, replacing TensorBoard with wandb and adding enhanced metrics tracking.

## Changes Implemented

### 1. Configuration Schema Updates (`src/data/config.py`)

**Updated `LoggingConfig` class (lines 347-383):**
- Added wandb-specific fields to the Pydantic model:
  - `wandb_entity: Optional[str]` - wandb username or team name
  - `wandb_tags: Optional[List[str]]` - tags for organizing runs
  - `wandb_notes: Optional[str]` - run description
- Added enhanced metrics fields:
  - `log_gradients: bool = True` - enable gradient norm logging
  - `log_parameters: bool = True` - enable parameter statistics logging
  - `log_memory: bool = True` - enable memory usage tracking
  - `enhanced_metrics_interval: int = 10` - log enhanced metrics every N steps
- Added validator for `enhanced_metrics_interval` to ensure positive values

### 2. Configuration File Updates (`configs/qwen_image_edit_plus_custom.yaml`)

**Changed:**
- `report_to: tensorboard` → `report_to: wandb`

**Added wandb-specific fields:**
- `wandb_entity`: Optional username or team name (set to null by default)
- `wandb_tags`: List of tags for organizing runs: `["qwen-edit-plus", "image-editing", "lora"]`
- `wandb_notes`: Run description for documentation
- `log_gradients: true`: Enable gradient norm logging
- `log_parameters: true`: Enable parameter statistics logging
- `log_memory: true`: Enable GPU memory usage tracking
- `enhanced_metrics_interval: 10`: Log enhanced metrics every 10 steps

### 3. Base Trainer Updates (`src/trainer/base_trainer.py`)

#### A. Enhanced `setup_accelerator` method (lines 552-637)
- Added comprehensive wandb initialization with project, entity, tags, and notes
- Created detailed config dict for wandb dashboard
- Maintains backward compatibility with TensorBoard
- Added proper error handling for tracker initialization

#### B. New `log_enhanced_metrics` method (lines 381-445)
Logs the following metrics periodically:

**Gradient Norms:**
- `gradients/global_norm`: Global gradient norm across all parameters
- `gradients/lora_norm`: Gradient norm for LoRA layers only

**Parameter Statistics (for LoRA layers):**
- `parameters/mean`: Mean of trainable parameters
- `parameters/std`: Standard deviation of trainable parameters
- `parameters/max`: Maximum parameter value
- `parameters/min`: Minimum parameter value

**Memory Usage:**
- `memory/allocated_gb`: Currently allocated GPU memory in GB
- `memory/reserved_gb`: Reserved GPU memory in GB
- `memory/max_allocated_gb`: Peak allocated GPU memory in GB

#### C. Enhanced `forward_loss` method (lines 333-379)
- Returns a dictionary with loss breakdown when mask loss is enabled
- Computes foreground and background losses separately
- Calculates mask coverage statistics
- Maintains backward compatibility by returning scalar loss when no mask is present

#### D. Updated `train_epoch` method (lines 486-489)
- Integrated `log_enhanced_metrics` call every N steps (configurable via `enhanced_metrics_interval`)
- Passes batch data to metrics logger for potential future enhancements

#### E. Updated `update_progressbar` method (lines 524-545)
- Changed metric names to wandb-friendly format with `train/` prefix:
  - `train/loss`
  - `train/smooth_loss`
  - `train/learning_rate`
  - `train/epoch`
  - `train/fps`
- Maintains formatted display for progress bar

### 4. Qwen Trainer Updates (`src/trainer/qwen_image_edit_trainer.py`)

#### Enhanced `_compute_loss` method (lines 637-664)
- Logs timestep statistics every 10 steps:
  - `training/timestep_mean`: Average timestep sampled per batch
  - `training/timestep_std`: Timestep variance
  
- Logs mask loss breakdown (conditional - only when mask data exists):
  - `loss/foreground`: Loss in masked regions
  - `loss/background`: Loss in non-masked regions
  - `loss/mask_coverage`: Percentage of foreground pixels

- Handles loss_dict return from `forward_loss` and extracts scalar loss for backpropagation

### 5. Validation Image Logging (Already Implemented)

The existing `log_images_auto` function in `src/utils/logger.py` already supports wandb:
- Automatically detects wandb tracker and logs images as wandb.Image
- Logs validation control images and generated images during validation runs
- Called from `src/validation/validation_sampler.py` at specified intervals

## Metrics Summary

### Logged Every Training Step:
1. `train/loss` - Current step loss
2. `train/smooth_loss` - Exponentially smoothed loss
3. `train/learning_rate` - Current learning rate
4. `train/epoch` - Current epoch
5. `train/fps` - Training throughput

### Logged Every N Steps (default: 10):
6. `gradients/global_norm` - Global gradient norm
7. `gradients/lora_norm` - LoRA gradient norm
8. `parameters/mean` - Mean of LoRA parameters
9. `parameters/std` - Std of LoRA parameters
10. `parameters/max` - Max LoRA parameter value
11. `parameters/min` - Min LoRA parameter value
12. `memory/allocated_gb` - GPU memory allocated
13. `memory/reserved_gb` - GPU memory reserved
14. `memory/max_allocated_gb` - Peak GPU memory
15. `training/timestep_mean` - Average timestep
16. `training/timestep_std` - Timestep variance

### Conditional (only when mask data exists):
17. `loss/foreground` - Foreground region loss
18. `loss/background` - Background region loss
19. `loss/mask_coverage` - Mask coverage ratio

### Validation (at validation_steps intervals):
20. Control images (as wandb.Image)
21. Generated images (as wandb.Image) with captions

## Usage

### Starting Training with Wandb:

```bash
# Option 1: Use config as-is (entity: null)
python train.py --config configs/qwen_image_edit_plus_custom.yaml

# Option 2: Set wandb entity via environment variable
export WANDB_ENTITY="your-username"
python train.py --config configs/qwen_image_edit_plus_custom.yaml

# Option 3: Edit config file to set wandb_entity directly
# Edit configs/qwen_image_edit_plus_custom.yaml:
# wandb_entity: "your-username"
```

### Viewing Results:

1. **Wandb Dashboard**: Visit https://wandb.ai/your-entity/qwen_edit_plus_custom
2. **Metrics Panel**: View all training metrics organized by prefix (train/, gradients/, parameters/, memory/, loss/)
3. **Media Panel**: View validation images with control and generated outputs
4. **System Metrics**: GPU utilization, memory usage tracked automatically by wandb

## Configuration Options

All wandb-specific settings can be adjusted in the config file:

```yaml
logging:
  report_to: wandb  # or "tensorboard" to switch back
  tracker_project_name: qwen_edit_plus_custom
  wandb_entity: null  # Set to your wandb username/team
  wandb_tags: ["qwen-edit-plus", "image-editing", "lora"]
  wandb_notes: "Qwen Image Edit Plus training with LoRA on custom dataset"
  log_gradients: true  # Disable to skip gradient logging
  log_parameters: true  # Disable to skip parameter statistics
  log_memory: true  # Disable to skip memory tracking
  enhanced_metrics_interval: 10  # Log enhanced metrics every N steps
```

## Benefits

1. **Better Visualization**: Wandb provides superior visualization capabilities compared to TensorBoard
2. **Comprehensive Monitoring**: Track gradients, parameters, and memory in addition to loss
3. **Automatic Image Logging**: Validation images with captions for qualitative assessment
4. **Team Collaboration**: Easy sharing and comparison of runs
5. **Experiment Tracking**: Tags, notes, and organized metrics for better experiment management
6. **Zero Performance Impact**: Metrics are computed efficiently with minimal overhead
7. **Conditional Logging**: Mask loss metrics only logged when mask data is present

## Notes

- All enhanced metrics logging is conditional based on config flags
- Mask loss breakdown only logged when `mask_loss: true` AND mask data exists in batch
- Memory logging only works on CUDA devices
- Validation image logging happens automatically during validation runs
- No code changes needed to switch back to TensorBoard (just change `report_to` in config)


