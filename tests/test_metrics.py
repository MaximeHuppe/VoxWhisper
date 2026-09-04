"""Dice metrics and DiceBCELoss tests."""
from __future__ import annotations

import pytest
import torch

from voxwhisper.training.loss import DiceBCELoss, _channel_flat
from voxwhisper.training.metrics import named_foreground_dice


def test_channel_flat_keeps_channels_contiguous():
    volume = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 1, 1, 4)
    flat = _channel_flat(volume)
    assert flat.shape == (3, 8)
    for channel in range(3):
        expected = volume[:, channel].reshape(-1)
        torch.testing.assert_close(flat[channel], expected)


def test_soft_dice_pools_per_channel_across_batch():
    """Wrong reshape(C, -1) on [B, C, ...] mixes tracts; movedim does not."""
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
    torch.testing.assert_close(
        loss, expected_dice_loss + bce, atol=1e-5, rtol=1e-5
    )
    assert expected_dice[0].item() == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert expected_dice[1].item() == pytest.approx(1.0, abs=1e-3)


def test_dice_bce_from_config():
    loss = DiceBCELoss.from_config({"training": {"bce_weight": 0.5}})
    assert loss.bce_weight == pytest.approx(0.5)
    default = DiceBCELoss.from_config({})
    assert default.bce_weight == pytest.approx(1.0)


def test_named_foreground_dice_drops_background():
    scores = [0.99, 0.8, 0.0, 0.4]
    named = named_foreground_dice(scores, ["background", "left_thalamus", "right_thalamus", "brainstem"])
    assert named == {"left_thalamus": 0.8, "right_thalamus": 0.0, "brainstem": 0.4}
    assert named_foreground_dice(scores) == {"c1": 0.8, "c2": 0.0, "c3": 0.4}


def test_load_config_injects_structure_names():
    from voxwhisper.util.config import load_config
    cfg = load_config()
    names = cfg["data"]["structure_names"]
    assert names[0] == "background"
    assert "left_thalamus" in names
    assert "brainstem" in names
    assert len(names) == 33
    assert len(names) == len(cfg["data"]["prompts"])
    assert cfg["model"]["name"] == "VoxDense"
