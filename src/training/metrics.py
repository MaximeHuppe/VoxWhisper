"""Loss functions and Dice metrics for VoxWhisper.

Loss design notes
-----------------
``DiceBCELoss`` combines two terms:

* **BCE** — ``binary_cross_entropy_with_logits`` over *all* channels
  (background included).  Background supervision through BCE prevents the
  model from freely expanding foreground predictions to cover the whole
  volume.  Optional ``pos_weight`` upweights the rare positive voxels on
  foreground channels so an all-empty prediction is expensive.

* **Dice** — mean soft Dice over *foreground channels only* (indices 1…N_T−1).
  Each channel is pooled over the batch and spatial axes (channel dim is
  moved to the front before flattening).  Including the background in Dice
  is unhelpful: background voxels dominate the volume, so the background
  Dice is trivially ~1 and contributes almost no gradient.

The total is their sum (equal weighting).  During training the loss is applied
at every decoder scale via ``deep_supervision_loss`` with configurable per-
scale weights (typically [0.1, 0.3, 0.6]).  The target at lower resolutions is
downsampled with trilinear interpolation, producing soft labels near boundaries
that act as implicit label smoothing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

PosWeightSpec = Union[float, Sequence[float], torch.Tensor]


def build_pos_weight(
    n_channels: int,
    spec: Optional[PosWeightSpec],
) -> Optional[torch.Tensor]:
    """Build a ``[N_T]`` BCE positive-class weight tensor.

    * ``None`` or ``1`` / ``1.0`` → unweighted (returns ``None``).
    * A scalar ``w`` → background stays 1, every foreground channel is ``w``.
    * A sequence / 1-D tensor of length ``n_channels`` → used as-is.
    """
    if spec is None:
        return None
    if isinstance(spec, torch.Tensor):
        if spec.ndim != 1 or spec.numel() != n_channels:
            raise ValueError(
                f"bce_pos_weight tensor must have shape [{n_channels}], got {tuple(spec.shape)}"
            )
        weight = spec.detach().to(dtype=torch.float32)
    elif isinstance(spec, (list, tuple)):
        if len(spec) != n_channels:
            raise ValueError(
                f"bce_pos_weight has {len(spec)} values, expected {n_channels} channels"
            )
        weight = torch.tensor([float(x) for x in spec], dtype=torch.float32)
    else:
        value = float(spec)
        if value <= 0:
            raise ValueError(f"bce_pos_weight must be > 0, got {value}")
        if value == 1.0:
            return None
        weight = torch.ones(n_channels, dtype=torch.float32)
        if n_channels > 1:
            weight[1:] = value
        return weight

    if torch.any(weight <= 0):
        raise ValueError("bce_pos_weight values must all be > 0")
    if torch.allclose(weight, torch.ones(n_channels, dtype=torch.float32)):
        return None
    return weight


def _channel_flat(volume: torch.Tensor) -> torch.Tensor:
    """``[B, C, ...]`` → ``[C, B*...]`` keeping each channel contiguous."""
    n_channels = volume.shape[1]
    return volume.movedim(1, 0).reshape(n_channels, -1)


########################################################
#               DICE BCE LOSS FUNCTION                 #
########################################################

class DiceBCELoss(nn.Module):
    """Foreground-focused Dice + full-spectrum BCE.

    Parameters
    ----------
    eps : float
        Smoothing constant added to numerator and denominator of the Dice
        coefficient to avoid division by zero on empty classes.
    pos_weight : Tensor or None
        Optional ``[N_T]`` positive-class weights for BCE.  Broadcasts over
        batch and spatial dims.  ``None`` is unweighted BCE.
    """

    def __init__(
        self,
        eps: float = 1e-5,
        pos_weight: Optional[torch.Tensor] = None,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:


        super().__init__()
        self.eps = eps
        if pos_weight is None:
            self.pos_weight: Optional[torch.Tensor] = None
        else:
            weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            if weight.ndim != 1:
                raise ValueError(
                    f"pos_weight must be 1-D [N_T], got shape {tuple(weight.shape)}"
                )
            self.register_buffer("pos_weight", weight)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    @classmethod
    def from_config(cls, config: dict, n_channels: int) -> "DiceBCELoss":
        spec = config.get("training", {}).get("bce_pos_weight", 1.0)
        bce_weight = config.get("training", {}).get("bce_weight", 1.0)
        dice_weight = config.get("training", {}).get("dice_weight", 1.0)
        return cls(pos_weight=build_pos_weight(n_channels, spec), bce_weight=bce_weight, dice_weight=dice_weight)

    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_logits : [B, N_T, D, H, W]  — raw (un-sigmoided) logits
        target_mask : [B, N_T, D, H, W]  — binary targets in {0, 1}

        Returns
        -------
        Scalar loss = BCE (all channels) + mean soft Dice (foreground only).
        """
        n_channels = pred_logits.shape[1]
        pos_weight = None
        if self.pos_weight is not None:
            if self.pos_weight.numel() != n_channels:
                raise ValueError(
                    f"pos_weight has {self.pos_weight.numel()} values, "
                    f"but prediction has {n_channels} channels"
                )
            pos_weight = self.pos_weight.to(
                device=pred_logits.device, dtype=pred_logits.dtype
            )
            # PyTorch matches pos_weight against the last dim unless we expand
            # it onto the channel axis: [C] → [1, C, 1, 1, ...].
            spatial_rank = pred_logits.ndim - 2
            pos_weight = pos_weight.view(1, -1, *([1] * spatial_rank))

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_mask, pos_weight=pos_weight
        )

        pred_probs = torch.sigmoid(pred_logits)
        n_foreground = n_channels - 1

        if n_foreground > 0:
            fg_pred = _channel_flat(pred_probs[:, 1:])
            fg_tgt = _channel_flat(target_mask[:, 1:])
            intersection = (fg_pred * fg_tgt).sum(dim=1)
            union = fg_pred.sum(dim=1) + fg_tgt.sum(dim=1)
            channel_dice = (2.0 * intersection + self.eps) / (union + self.eps)
            dice_loss = (1.0 - channel_dice).mean()
        else:
            pred_flat = pred_probs.reshape(-1)
            tgt_flat = target_mask.reshape(-1)
            intersection = (pred_flat * tgt_flat).sum()
            union = pred_flat.sum() + tgt_flat.sum()
            dice_loss = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def deep_supervision_loss(
    predictions: List[torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
    weights: Sequence[float],
) -> torch.Tensor:
    """Weighted sum of ``criterion`` across decoder stages.

    The target is downsampled to each stage's spatial resolution with trilinear
    interpolation.  Interpolating binary one-hot targets produces soft labels
    near boundaries, acting as implicit label smoothing at coarser scales.

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
            target,
            size=pred.shape[2:],
            mode="trilinear",
            align_corners=True,
        )
        loss = loss + w * criterion(pred, downsampled)
    return loss


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
) -> List[float]:
    """Per-channel Dice after sigmoid thresholding (multi-label convention).

    Parameters
    ----------
    logits, target : ``[B, N_T, D, H, W]`` or ``[N_T, D, H, W]`` (batch dim
                     is added automatically when missing).
    threshold      : sigmoid activation threshold (default 0.5).

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
        intersection = (pred_c & gt_c).sum().float()
        denom = pred_c.sum() + gt_c.sum()
        scores.append(1.0 if denom == 0 else float((2.0 * intersection + eps) / (denom + eps)))
    return scores


def foreground_channel_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> float:
    """Mean Dice over non-background channels (channel 0 excluded).

    Returns 1.0 when there are no foreground channels (degenerate case).
    """
    scores = channel_dice_from_logits(logits, target, threshold=threshold, eps=eps)
    foreground = scores[1:] if len(scores) > 1 else scores
    return sum(foreground) / len(foreground) if foreground else 1.0


def named_foreground_dice(
    scores: Sequence[float],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Map per-channel Dice to foreground names (channel 0 dropped).

    When ``class_names`` is omitted, keys are ``c1``, ``c2``, …
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
