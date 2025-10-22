# Qwen-Image-Edit-Plus Custom Fine-Tune Guide

This runbook captures the end-to-end workflow for fine-tuning **Qwen Image Edit Plus** on the `metadata-v12-recaptioned-long` dataset using the repository's LoRA pipeline.

## 1. Dataset Preparation
- **Source metadata**: `workspace/metadata-v12-recaptioned-long/metadata.jsonl`.
- **Prompt regeneration**: Use `vLLM` with `Qwen2.5-VL` to compare each `hdr_file_name` (control image) against `output_file_name` (target image) and produce an edit description. Persist prompts alongside the metadata row.
  - Automation: once the vLLM server is running, execute `python script/generate_qwen25vl_prompts.py --base-url http://<host>:<port>/v1` (defaults read `/workspace/metadata-v12-recaptioned-long/train/metadata.jsonl` and `/workspace/metadata-v12-recaptioned-long/test/metadata.jsonl`; the latter becomes validation).
- **CSV conversion**: Convert the enriched JSONL into `train.csv` (and optionally `val.csv`) with columns:
  - `path_target`: absolute or repo-relative path to `output_file_name`.
  - `path_control`: path to `hdr_file_name`.
  - `prompt`: regenerated edit caption.
  - Additional controls or masks can be added as `path_control_1`, `path_mask`, etc.
- **Sanity checks**:
  - Sample a few rows to confirm text/control/target alignment.
  - Run `python -m src.main --config configs/qwen_image_edit_plus_custom.yaml --dry-run` once CSVs exist to ensure the loader sees the files (requires minor code tweak if `--dry-run` not available).

> **Clarification needed**: confirm final CSV filenames/locations so they match the paths configured in `configs/qwen_image_edit_plus_custom.yaml` (currently `/workspace/metadata-v12-recaptioned-long/train.csv` placeholder).

## 2. Cache Generation (Optional but Recommended)
- Update `cache.devices` in the config if you want different GPU assignments.
- Command example (after CSV creation):
  ```bash
  accelerate launch -m src.main --config configs/qwen_image_edit_plus_custom.yaml --cache
  ```
- Ensure there is sufficient disk space under `output_dir` for cached latents/embeddings.

## 3. Training Configuration Highlights
- Config file: `configs/qwen_image_edit_plus_custom.yaml`
  - LoRA rank increased to `32` to leverage the larger dataset.
  - Mixed precision set to `bf16`; per-GPU batch size 2 with gradient accumulation 4 for 8 GPUs (effective global batch size 64).
  - Scheduler: cosine with 500 warmup steps; total steps 20k (adjust once dataset stats are finalized).
- Review `fit_device` and `cache.devices` mapping to match the planned GPU layout (edit the config if distributing across different cards).
- Set `logging.output_dir` to a persistent location with enough space for checkpoints.
- If you want validation sampling during training, populate `logging.sampling.validation_data` or provide a validation CSV path.

## 4. Launch Training
1. Create/activate the environment (`setup.sh` or manual install).
2. Set visible GPUs, e.g.
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ```
3. Kick off training with Accelerate:
   ```bash
   accelerate launch --num_processes 8 --mixed_precision bf16 -m src.main --config configs/qwen_image_edit_plus_custom.yaml
   ```
4. Monitor TensorBoard at `logging.output_dir` for loss curves and checkpoint status.

## 5. Evaluation & Handoff
- After training, use the same config with `--predict` (if supported) or the notebook under `tests/trainer/test_qwen_image_edit_plus.ipynb` as a template to generate comparison grids.
- Archive:
  - Final LoRA weights (`logging.output_dir/checkpoint-XXXX`).
  - Updated metadata CSVs and prompt generation script.
  - Training logs (loss curves, metrics, sample outputs).
- Document any anomalies or failure cases for future retrains.

## 6. Next Steps / Open Items
1. **Confirm** the actual `pretrained_model_name_or_path` you intend to use (full-precision vs. 4-bit base).
2. **Verify** CSV paths and add validation split details if required.
3. **Decide** on evaluation metrics (CLIP similarity, human review cadence) and fold them into the pipeline.
