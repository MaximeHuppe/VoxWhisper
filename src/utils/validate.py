"""Patch-level and full-volume validation used during training."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from tqdm import tqdm

from src.infer import (
    load_subject_for_inference,
    load_text_embeddings,
    predict_full_volume,
    volumes_to_tensors,
)
from src.utils.metrics import (
    channel_dice_from_logits,
    deep_supervision_loss,
    named_foreground_dice,
)
from src.utils.nifti_io import label_to_multichannel


def _mean_channel_scores(score_lists: Sequence[Sequence[float]]) -> list[float]:
    n = len(score_lists)
    n_ch = len(score_lists[0])
    return [sum(row[i] for row in score_lists) / n for i in range(n_ch)]


@torch.no_grad()
def evaluate_patches(
    model,
    loader,
    criterion,
    deep_sup_weights,
    device,
    threshold=0.5,
    class_names: Optional[Sequence[str]] = None,
):
    """
    Patch-level validation.

    Returns mean deep-supervision loss, mean foreground Dice, and per-tract
    Dice over val patches (full-resolution decoder stage only).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    per_patch_scores: list[list[float]] = []

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
            per_patch_scores.append(
                channel_dice_from_logits(
                    logits[batch_idx], gt_mask[batch_idx], threshold=threshold
                )
            )

    if not per_patch_scores:
        return {"val_loss": 0.0, "dice_patch": 0.0, "dice_patch_classes": {}}

    mean_scores = _mean_channel_scores(per_patch_scores)
    foreground = mean_scores[1:] if len(mean_scores) > 1 else mean_scores
    return {
        "val_loss": total_loss / n_batches if n_batches > 0 else 0.0,
        "dice_patch": sum(foreground) / len(foreground) if foreground else 1.0,
        "dice_patch_classes": named_foreground_dice(mean_scores, class_names),
    }


@torch.no_grad()
def evaluate_volume_dice(
    model, config, subject_ids, device
) -> Optional[Dict[str, object]]:
    """Mean foreground channel Dice over full val volumes (sliding window).

    Returns ``None`` when no val subject has a GT mask.  Otherwise a dict
    with ``dice_volume`` (mean over tracts) and ``dice_volume_classes``.
    """
    inf_cfg = config.get("inference", {})
    patch_size = tuple(int(x) for x in config["data"]["patch"]["size"])
    threshold = float(inf_cfg.get("threshold", 0.5))
    text_embeddings = load_text_embeddings(config, map_location="cpu")
    class_names = config["data"].get("structure_names") or config["data"].get("prompts")

    per_subject: list[list[float]] = []
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
        per_subject.append(
            channel_dice_from_logits(
                logits.cpu(), gt_onehot.unsqueeze(0), threshold=threshold
            )
        )
        del logits

    if not per_subject:
        return None
    mean_scores = _mean_channel_scores(per_subject)
    foreground = mean_scores[1:] if len(mean_scores) > 1 else mean_scores
    return {
        "dice_volume": float(sum(foreground) / len(foreground)) if foreground else 1.0,
        "dice_volume_classes": named_foreground_dice(mean_scores, class_names),
    }
