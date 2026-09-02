"""DiceBCELoss grouping, BCE pos-weight, and named per-tract Dice."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.metrics import (
    DiceBCELoss,
    _channel_flat,
    build_pos_weight,
    named_foreground_dice,
)


def test_channel_flat_keeps_channels_contiguous():
    volume = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 1, 1, 4)
    flat = _channel_flat(volume)
    assert flat.shape == (3, 8)
    # Channel c must equal concat of that channel over the batch.
    for channel in range(3):
        expected = volume[:, channel].reshape(-1)
        torch.testing.assert_close(flat[channel], expected)


def test_soft_dice_pools_per_channel_across_batch():
    """Wrong ``reshape(C, -1)`` on ``[B, C, ...]`` mixes tracts; movedim does not."""
    logits = torch.full((2, 3, 1, 1, 4), -10.0)
    target = torch.zeros(2, 3, 1, 1, 4)

    logits[:, 0] = 10.0
    target[:, 0] = 1.0

    # Tract 1: 4 GT voxels in sample 0, prediction hits 2 of them.
    target[0, 1, 0, 0, :] = 1.0
    logits[0, 1, 0, 0, :2] = 10.0

    # Tract 2: 1 GT voxel in sample 1, predicted perfectly.
    target[1, 2, 0, 0, 0] = 1.0
    logits[1, 2, 0, 0, 0] = 10.0

    criterion = DiceBCELoss(eps=1e-5)
    loss = criterion(logits, target)

    pred = torch.sigmoid(logits)[:, 1:]
    tgt = target[:, 1:]
    intersection = (pred * tgt).sum(dim=(0, 2, 3, 4))
    union = pred.sum(dim=(0, 2, 3, 4)) + tgt.sum(dim=(0, 2, 3, 4))
    expected_dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    expected_dice_loss = (1.0 - expected_dice).mean()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    torch.testing.assert_close(loss, bce + expected_dice_loss, atol=1e-5, rtol=1e-5)
    assert expected_dice[0].item() == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert expected_dice[1].item() == pytest.approx(1.0, abs=1e-3)


def test_build_pos_weight_scalar_skips_background():
    weight = build_pos_weight(7, 20.0)
    assert weight is not None
    assert weight.tolist() == [1.0] + [20.0] * 6
    assert build_pos_weight(7, 1.0) is None
    assert build_pos_weight(7, None) is None


def test_build_pos_weight_per_channel_list():
    weight = build_pos_weight(3, [1.0, 5.0, 10.0])
    assert weight is not None
    assert weight.tolist() == [1.0, 5.0, 10.0]
    with pytest.raises(ValueError, match="expected 3"):
        build_pos_weight(3, [1.0, 5.0])


def test_pos_weight_makes_false_negatives_more_expensive():
    logits = torch.full((1, 3, 2, 2, 2), -8.0)
    target = torch.zeros(1, 3, 2, 2, 2)
    target[:, 0] = 1.0
    target[0, 1, 0, 0, 0] = 1.0

    unweighted = DiceBCELoss()(logits, target)
    weighted = DiceBCELoss(pos_weight=build_pos_weight(3, 20.0))(logits, target)
    assert weighted.item() > unweighted.item()


def test_dice_bce_from_config():
    loss = DiceBCELoss.from_config({"training": {"bce_pos_weight": 20}}, n_channels=3)
    assert loss.pos_weight is not None
    assert loss.pos_weight.tolist() == [1.0, 20.0, 20.0]
    plain = DiceBCELoss.from_config({}, n_channels=3)
    assert plain.pos_weight is None


def test_named_foreground_dice_drops_background():
    scores = [0.99, 0.8, 0.0, 0.4]
    named = named_foreground_dice(scores, ["background", "ATR_left", "ATR_right", "CG_left"])
    assert named == {"ATR_left": 0.8, "ATR_right": 0.0, "CG_left": 0.4}
    assert named_foreground_dice(scores) == {"c1": 0.8, "c2": 0.0, "c3": 0.4}


def test_load_config_injects_structure_names():
    from src.utils.config import load_config

    cfg = load_config()
    names = cfg["data"]["structure_names"]
    assert names[0] == "background"
    assert "ATR_left" in names
    assert len(names) == len(cfg["data"]["prompts"])
