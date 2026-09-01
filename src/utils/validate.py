"""Patch-level and full-volume validation used during training."""
from __future__ import annotations

import torch
from tqdm import tqdm

from src.infer import (
    load_subject_for_inference,
    load_text_embeddings,
    predict_full_volume,
    volumes_to_tensors,
)
from src.utils.metrics import deep_supervision_loss, foreground_channel_dice
from src.utils.nifti_io import label_to_multichannel


@torch.no_grad()
def evaluate_patches(model, loader, criterion, deep_sup_weights, device, threshold=0.5):
    """
    Patch-level validation.

    Returns mean deep-supervision loss and mean foreground Dice over val
    patches (full-resolution decoder stage only).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    dice_sum = 0.0
    n_patches = 0

    for primary_vol, secondary_vol, text_emb, gt_mask in loader:
        primary_vol = primary_vol.to(device)
        secondary_vol = secondary_vol.to(device)
        text_emb = text_emb.to(device)
        gt_mask = gt_mask.to(device)

        predictions = model(primary_vol, secondary_vol, text_emb)
        total_loss += deep_supervision_loss(
            predictions, gt_mask, criterion, deep_sup_weights
        ).item()
        n_batches += 1

        logits = predictions[-1]
        for batch_idx in range(logits.shape[0]):
            dice_sum += foreground_channel_dice(
                logits[batch_idx], gt_mask[batch_idx], threshold=threshold
            )
            n_patches += 1

    return {
        "val_loss": total_loss / n_batches if n_batches > 0 else 0.0,
        "dice_patch": dice_sum / n_patches if n_patches > 0 else 0.0,
    }


@torch.no_grad()
def evaluate_volume_dice(model, config, subject_ids, device) -> float | None:
    """Mean foreground channel Dice over full val volumes (sliding window)."""
    inf_cfg = config.get("inference", {})
    patch_size = tuple(int(x) for x in config["data"]["patch"]["size"])
    threshold = float(inf_cfg.get("threshold", 0.5))
    text_embeddings = load_text_embeddings(config, map_location="cpu")

    scores = []
    for subject_id in tqdm(subject_ids, desc="Val volume Dice", leave=False):
        primary_np, secondary_np, labels_np, _ = load_subject_for_inference(config, subject_id)
        if labels_np is None:
            print(f"Warning: skipping {subject_id} (no GT mask)")
            continue

        primary, secondary = volumes_to_tensors(primary_np, secondary_np)
        logits = predict_full_volume(
            model,
            primary.to(device),
            secondary.to(device),
            text_embeddings.to(device),
            roi_size=patch_size,
            sw_batch_size=int(inf_cfg.get("sw_batch_size", 2)),
            overlap=float(inf_cfg.get("overlap", 0.5)),
            mode=str(inf_cfg.get("mode", "gaussian")),
            sigma_scale=float(inf_cfg.get("sigma_scale", 0.125)),
            progress=False,
        )
        n_prompts = int(logits.shape[1])
        gt_onehot = torch.from_numpy(label_to_multichannel(labels_np, n_prompts))
        scores.append(
            foreground_channel_dice(
                logits.cpu(), gt_onehot.unsqueeze(0), threshold=threshold
            )
        )
        del logits

    if not scores:
        return None
    return float(sum(scores) / len(scores))
