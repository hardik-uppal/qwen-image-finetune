import logging
import torch, torchvision
from accelerate.logging import get_logger


def load_logger(name, log_level="INFO"):
    logger = get_logger(name, log_level=log_level)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )
    return logger


def log_images_auto(accelerator, tag, images, step, caption=None, nrow=4, max_images=16):
    """images: [B,C,H,W] in [-1,1]"""
    if not accelerator.is_main_process:
        return

    # 预处理：裁样、归一化到[0,1]、拼网格 (CHW)
    t = images.detach().float()[:max_images]
    t = (t + 1) / 2
    t = t.clamp(0, 1)
    grid = torchvision.utils.make_grid(t.cpu(), nrow=nrow, padding=2)  # [C,H,W]

    logged = False

    # 1) Weights & Biases
    try:
        run = accelerator.get_tracker("wandb", unwrap=True)  # 若未启用会抛异常
        if run is not None:
            import wandb
            npimg = grid.permute(1, 2, 0).numpy()            # HWC
            run.log({tag: wandb.Image(npimg, caption=caption)}, step=step)
            logged = True
            logging.info(f"Logged image '{tag}' to wandb at step {step}")
    except Exception as e:
        logging.warning(f"Failed to log image to wandb: {e}")

    # 2) TensorBoard
    try:
        tb = accelerator.get_tracker("tensorboard")          # 返回包装器
        if hasattr(tb, "writer"):
            tb.writer.add_image(tag, grid, step, dataformats="CHW")
            logged = True
    except Exception:
        pass

    # 3) 没有可用富媒体 tracker：至少打一个标量占位
    if not logged:
        accelerator.log({f"{tag}/num_images": int(t.shape[0])}, step=step)


def log_comparison_images(
    accelerator, 
    tag, 
    control_images, 
    generated_images, 
    step, 
    target_images=None,
    caption=None, 
    padding=4
):
    """
    Log control, target (optional), and generated images side by side for easy comparison.
    
    Args:
        accelerator: Accelerator instance
        tag: Logging tag
        control_images: Tensor [B,C,H,W] in [-1,1]
        generated_images: Tensor [B,C,H,W] in [-1,1]
        target_images: Optional tensor [B,C,H,W] in [-1,1]
        step: Current training step
        caption: Optional caption for the image
        padding: Padding between images
    """
    if not accelerator.is_main_process:
        return
    
    # Convert all images to [0,1] range
    control = (control_images.detach().float() + 1) / 2
    control = control.clamp(0, 1).cpu()
    
    generated = (generated_images.detach().float() + 1) / 2
    generated = generated.clamp(0, 1).cpu()
    
    # Prepare list of image tensors to concatenate
    images_to_concat = [control]
    
    if target_images is not None:
        target = (target_images.detach().float() + 1) / 2
        target = target.clamp(0, 1).cpu()
        images_to_concat.append(target)
    
    images_to_concat.append(generated)
    
    # Concatenate horizontally for each sample in batch
    batch_size = control.shape[0]
    concatenated_rows = []
    
    for i in range(batch_size):
        # Get images for this sample
        sample_images = [img[i] for img in images_to_concat]
        
        # Resize all images to same height (use control image height as reference)
        _, h, w = sample_images[0].shape
        resized_images = []
        
        for img in sample_images:
            _, img_h, img_w = img.shape
            if img_h != h or img_w != w:
                # Resize to match control image dimensions
                img_resized = torch.nn.functional.interpolate(
                    img.unsqueeze(0), 
                    size=(h, w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                resized_images.append(img_resized)
            else:
                resized_images.append(img)
        
        # Add padding between images
        padded_images = []
        for j, img in enumerate(resized_images):
            if j > 0:  # Add white padding before image (except first)
                pad_strip = torch.ones(img.shape[0], img.shape[1], padding)
                padded_images.append(pad_strip)
            padded_images.append(img)
        
        # Concatenate horizontally (along width dimension)
        concatenated = torch.cat(padded_images, dim=2)  # [C, H, W_total]
        concatenated_rows.append(concatenated)
    
    # Stack all samples vertically if batch > 1
    if len(concatenated_rows) > 1:
        # Add horizontal separator between rows
        final_rows = []
        for i, row in enumerate(concatenated_rows):
            if i > 0:
                separator = torch.ones(row.shape[0], padding, row.shape[2])
                final_rows.append(separator)
            final_rows.append(row)
        final_image = torch.cat(final_rows, dim=1)  # [C, H_total, W]
    else:
        final_image = concatenated_rows[0]
    
    logged = False
    
    # Log to W&B
    try:
        run = accelerator.get_tracker("wandb", unwrap=True)
        if run is not None:
            import wandb
            npimg = final_image.permute(1, 2, 0).numpy()  # CHW -> HWC
            run.log({tag: wandb.Image(npimg, caption=caption)}, step=step)
            logged = True
            logging.info(f"Logged comparison image '{tag}' to wandb at step {step}")
    except Exception as e:
        logging.warning(f"Failed to log comparison image to wandb: {e}")
    
    # Log to TensorBoard
    try:
        tb = accelerator.get_tracker("tensorboard")
        if hasattr(tb, "writer"):
            tb.writer.add_image(tag, final_image, step, dataformats="CHW")
            logged = True
    except Exception:
        pass
    
    if not logged:
        accelerator.log({f"{tag}/num_images": int(batch_size)}, step=step)


def log_text_auto(accelerator, tag, rows, step, max_rows=64):
    """
    rows: list[dict] 或 list[str]
    自动用 W&B 的 Table 或 TensorBoard 的 add_text
    """
    if not accelerator.is_main_process:
        return

    logged = False

    # 1) W&B 表格更直观
    try:
        run = accelerator.get_tracker("wandb", unwrap=True)
        if run is not None:
            import wandb
            rows_clip = rows[:max_rows]
            if isinstance(rows_clip[0], dict):
                cols = list(rows_clip[0].keys())
                table = wandb.Table(columns=cols)
                for r in rows_clip:
                    table.add_data(*[str(r[c]) for c in cols])
            else:
                table = wandb.Table(columns=["text"])
                for s in rows_clip:
                    table.add_data(str(s))
            run.log({tag: table}, step=step)
            logged = True
    except Exception:
        pass

    # 2) TensorBoard 文本
    try:
        tb = accelerator.get_tracker("tensorboard")
        if hasattr(tb, "writer"):
            rows_clip = rows[:max_rows]
            if isinstance(rows_clip[0], dict):
                text = "\n".join(
                    f"{i}. " + " | ".join(f"{k}: {v}" for k, v in r.items())
                    for i, r in enumerate(rows_clip)
                )
            else:
                text = "\n".join(f"{i}. {s}" for i, s in enumerate(rows_clip))
            tb.writer.add_text(tag, text, step)
            logged = True
    except Exception:
        pass

    if not logged:
        # 退化为打印到控制台（只主进程一次）
        accelerator.print(f"[{tag}] (no tracker) sample: {rows[:3]}")

# 用法示例（训练循环里）
# accelerator = Accelerator(log_with=["wandb","tensorboard"]); accelerator.init_trackers("my-run")
# if accelerator.is_main_process:
#     log_images_auto(accelerator, "samples/train", batch_images, step, caption="pred vs gt")
#     log_text_auto(accelerator,
#           "eval/text", [{"id": i, "pred": p, "gt": g} for i,(p,g) in enumerate(zip(preds,gts))],
#          step)
#)
