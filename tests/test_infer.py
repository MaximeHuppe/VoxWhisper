"""Sliding-window inference: Gaussian blending back to the native volume grid."""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infer import SlidingWindowPredictor, predict_full_volume
from src.utils.metrics import per_class_dice


class _IdentityNet(nn.Module):
    """Mimic VoxWhisper: three-arg forward, list of deep-supervision tensors."""

    def forward(self, primary, secondary, text):
        n_prompts = text.shape[1]
        # Clone so the stitch buffer does not alias the input crop.
        full = primary.expand(-1, n_prompts, -1, -1, -1).clone()
        return [full, full, full]


class _RecordingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, primary, secondary, text):
        self.calls.append((primary.shape, secondary.shape, text.shape))
        n_prompts = text.shape[1]
        full = (primary + secondary).expand(-1, n_prompts, -1, -1, -1).clone()
        return [full * 0.25, full * 0.5, full]


def test_predictor_splits_concatenated_channels_and_returns_fullres():
    model = _RecordingNet()
    text = torch.zeros(3, 8)
    predictor = SlidingWindowPredictor(model, text)

    inputs = torch.randn(2, 2, 16, 16, 16)
    out = predictor(inputs)

    assert out.shape == (2, 3, 16, 16, 16)
    assert len(model.calls) == 1
    primary_shape, secondary_shape, text_shape = model.calls[0]
    assert primary_shape == (2, 1, 16, 16, 16)
    assert secondary_shape == (2, 1, 16, 16, 16)
    assert text_shape == (2, 3, 8)


def test_sliding_window_output_matches_native_volume_size():
    """Odd native size (not a multiple of roi) is still reconstructed fully."""
    model = _IdentityNet()
    text = torch.ones(2, 4)
    primary = torch.randn(1, 1, 40, 37, 41)
    secondary = torch.randn(1, 1, 40, 37, 41)

    logits = predict_full_volume(
        model,
        primary,
        secondary,
        text,
        roi_size=(32, 32, 32),
        sw_batch_size=2,
        overlap=0.5,
        mode="gaussian",
    )

    assert logits.shape == (1, 2, 40, 37, 41)


def test_gaussian_blend_reconstructs_agreed_voxels():
    """When every patch agrees, Gaussian stitching must recover the primary volume."""
    model = _IdentityNet()
    text = torch.ones(1, 4)
    primary = torch.randn(1, 1, 48, 48, 48)
    secondary = torch.zeros_like(primary)

    logits = predict_full_volume(
        model,
        primary,
        secondary,
        text,
        roi_size=(32, 32, 32),
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian",
    )

    assert logits.shape == (1, 1, 48, 48, 48)
    torch.testing.assert_close(logits, primary, rtol=1e-4, atol=1e-4)


def test_mismatched_primary_secondary_shapes_raise():
    model = _IdentityNet()
    text = torch.ones(1, 4)
    primary = torch.zeros(1, 1, 32, 32, 32)
    secondary = torch.zeros(1, 1, 16, 16, 16)
    try:
        predict_full_volume(model, primary, secondary, text, roi_size=(16, 16, 16))
    except ValueError as exc:
        assert "shape" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for primary/secondary mismatch")


def test_per_class_dice_perfect_and_empty():
    pred = torch.tensor([0, 1, 1, 2])
    gt = torch.tensor([0, 1, 1, 2])
    scores = per_class_dice(pred, gt, n_classes=3)
    assert scores == [1.0, 1.0, 1.0]

    empty_pred = torch.zeros(4, dtype=torch.long)
    empty_gt = torch.zeros(4, dtype=torch.long)
    scores_empty = per_class_dice(empty_pred, empty_gt, n_classes=2)
    assert scores_empty == [1.0, 1.0]


def test_predict_restores_training_flag():
    model = _IdentityNet()
    model.train()
    text = torch.ones(1, 4)
    primary = torch.zeros(1, 1, 32, 32, 32)
    secondary = torch.zeros(1, 1, 32, 32, 32)
    predict_full_volume(model, primary, secondary, text, roi_size=(32, 32, 32), overlap=0.0)
    assert model.training is True


def test_checkpoint_epoch_numeric_not_lexicographic():
    from pathlib import Path
    from pipeline.evaluate import _checkpoint_epoch

    assert _checkpoint_epoch(Path("vox_whisper_epoch_9.pt")) < _checkpoint_epoch(
        Path("vox_whisper_epoch_10.pt")
    )
