"""Loss functions for VoxWhisper training.

Design notes
------------
``DiceBCELoss`` combines:
* **BCE** — ``binary_cross_entropy_with_logits`` over *all* channels.
* **Dice** — mean soft Dice over *foreground channels only* (indices 1…N_T−1),
  when ``exclude_background=True`` (default).

**Contract:** channel 0 must be the background class.  ``VoxDenseDataset``
always prepends background when sampling ``prompts_per_crop`` so training
batches obey the same layout as full-prompt validation.

The total loss is ``Dice + bce_weight × BCE``.  Applied at every decoder scale
via ``deep_supervision_loss`` with configurable per-scale weights
(typically [0.1, 0.3, 0.6]).
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _channel_flat(volume: torch.Tensor) -> torch.Tensor:
    """``[B, C, ...]`` → ``[C, B*...]`` keeping each channel contiguous."""
    n_channels = volume.shape[1]
    return volume.movedim(1, 0).reshape(n_channels, -1)


class DiceBCELoss(nn.Module):
    """Foreground-focused Dice + full-spectrum BCE.

    Parameters
    ----------
    bce_weight : float
        Multiplier on the BCE term.  Combined loss = Dice + bce_weight × BCE.
    eps : float
        Smoothing constant for the Dice denominator.
    exclude_background : bool
        If True (default), soft Dice ignores channel 0.  Callers must ensure
        channel 0 is background — never a sampled foreground structure.
    """

    def __init__(
        self,
        eps: float = 1e-5,
        bce_weight: float = 1.0,
        exclude_background: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.bce_weight = bce_weight
        self.exclude_background = exclude_background

    @classmethod
    def from_config(cls, config: dict) -> "DiceBCELoss":
        """Construct from ``training.bce_weight`` in the loaded YAML config."""
        train_cfg = config.get("training", {})
        bce_weight = float(train_cfg.get("bce_weight", 1.0))
        exclude_background = bool(train_cfg.get("exclude_background", True))
        return cls(bce_weight=bce_weight, exclude_background=exclude_background)

    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_logits : [B, N_T, D, H, W]  — raw (un-sigmoided) logits
        target_mask : [B, N_T, D, H, W]  — binary targets in {0, 1}
        """
        if pred_logits.shape != target_mask.shape:
            raise ValueError(
                f"pred/target shape mismatch: {tuple(pred_logits.shape)} vs "
                f"{tuple(target_mask.shape)}"
            )
        n_channels = pred_logits.shape[1]
        if n_channels < 1:
            raise ValueError("expected at least one prompt channel")

        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
        pred_probs = torch.sigmoid(pred_logits)

        if self.exclude_background and n_channels > 1:
            fg_pred = _channel_flat(pred_probs[:, 1:])
            fg_tgt = _channel_flat(target_mask[:, 1:])
            intersection = (fg_pred * fg_tgt).sum(dim=1)
            union = fg_pred.sum(dim=1) + fg_tgt.sum(dim=1)
            channel_dice = (2.0 * intersection + self.eps) / (union + self.eps)
            dice_loss = (1.0 - channel_dice).mean()
        else:
            # Single-channel or explicit "Dice over every channel".
            pred_flat = _channel_flat(pred_probs)
            tgt_flat = _channel_flat(target_mask)
            intersection = (pred_flat * tgt_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)
            channel_dice = (2.0 * intersection + self.eps) / (union + self.eps)
            dice_loss = (1.0 - channel_dice).mean()

        return dice_loss + self.bce_weight * bce_loss


def deep_supervision_loss(
    predictions: List[torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
    weights: Sequence[float],
) -> torch.Tensor:
    """Weighted sum of ``criterion`` across decoder stages.

    The target is downsampled to each stage's spatial resolution with trilinear
    interpolation, producing soft labels near boundaries at coarser scales.

    Parameters
    ----------
    predictions : list of [B, N_T, D_s, H_s, W_s] logits, one per decoder stage.
    target      : [B, N_T, D, H, W] full-resolution binary target.
    criterion   : loss function accepting (pred_logits, target).
    weights     : per-stage weights, must match len(predictions).
    """
    loss = predictions[0].new_zeros(())
    for pred, w in zip(predictions, weights):
        downsampled = F.interpolate(
            target, size=pred.shape[2:], mode="trilinear", align_corners=True,
        )
        loss = loss + w * criterion(pred, downsampled)
    return loss
