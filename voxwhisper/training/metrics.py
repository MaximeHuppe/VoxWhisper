"""Dice score metrics for VoxWhisper patch-level and volume-level evaluation."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def per_class_dice(
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    n_classes: int,
    eps: float = 1e-5,
) -> List[float]:
    """Dice coefficient per class from integer label maps.

    An empty class (absent in both prediction and ground truth) scores 1.0,
    reflecting perfect agreement on absence.

    Parameters
    ----------
    pred_labels, gt_labels : integer tensors of any shape.
    n_classes              : total number of classes (including background).

    Returns
    -------
    list of float, length ``n_classes``, one score per class.
    """
    pred_labels = pred_labels.long()
    gt_labels = gt_labels.long()
    scores: List[float] = []
    for class_id in range(n_classes):
        pred_c = pred_labels == class_id
        gt_c = gt_labels == class_id
        intersection = (pred_c & gt_c).sum().float()
        denom = pred_c.sum() + gt_c.sum()
        if denom == 0:
            scores.append(1.0)
        else:
            scores.append(float((2.0 * intersection + eps) / (denom + eps)))
    return scores


def channel_dice_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-5,
    ignore_empty_targets: bool = False,
) -> List[float]:
    """Per-channel Dice after sigmoid thresholding (multi-label convention).

    Parameters
    ----------
    logits, target : ``[B, N_T, D, H, W]`` or ``[N_T, D, H, W]``
    threshold      : sigmoid activation threshold (default 0.5).
    ignore_empty_targets : if True, channels with no GT voxels return ``nan``
        so callers can exclude them from the mean (avoids FP mass on absent
        structures dragging patch Dice to ~0).

    Returns
    -------
    list of float, length N_T, one score per prompt channel.
    """
    if logits.ndim == 4:
        logits = logits.unsqueeze(0)
        target = target.unsqueeze(0)
    pred = torch.sigmoid(logits) > threshold
    gt = target > 0.5
    scores: List[float] = []
    for class_id in range(logits.shape[1]):
        pred_c = pred[:, class_id]
        gt_c = gt[:, class_id]
        gt_sum = gt_c.sum()
        if ignore_empty_targets and gt_sum == 0:
            scores.append(float("nan"))
            continue
        intersection = (pred_c & gt_c).sum().float()
        denom = pred_c.sum() + gt_sum
        scores.append(1.0 if denom == 0 else float((2.0 * intersection + eps) / (denom + eps)))
    return scores


def foreground_channel_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> float:
    """Mean Dice over non-background channels (channel 0 excluded).

    Empty-target channels are skipped.  Returns 1.0 when there are no
    present foreground channels (degenerate case).
    """
    scores = channel_dice_from_logits(
        logits, target, threshold=threshold, eps=eps, ignore_empty_targets=True
    )
    foreground = scores[1:] if len(scores) > 1 else scores
    present = [s for s in foreground if s == s]  # drop NaN
    return sum(present) / len(present) if present else 1.0


def named_foreground_dice(
    scores: Sequence[float],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Map per-channel Dice to foreground names (channel 0 dropped).

    When ``class_names`` is omitted, keys are ``c1``, ``c2``, …
    NaN scores (absent GT) are kept so callers can filter them.
    """
    foreground = list(scores[1:] if len(scores) > 1 else scores)
    if class_names is None:
        names = [f"c{i}" for i in range(1, len(foreground) + 1)]
    else:
        names = list(class_names)
        if len(names) == len(scores) and len(scores) > 1:
            names = names[1:]
        if len(names) != len(foreground):
            raise ValueError(
                f"class_names has {len(names)} entries, expected {len(foreground)} "
                "foreground names (or all channels including background)"
            )
    return {name: float(score) for name, score in zip(names, foreground)}
