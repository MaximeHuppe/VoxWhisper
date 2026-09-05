"""Checkpoint monitor selection and TopK helpers."""
from __future__ import annotations

import pytest
import torch

from voxwhisper.training.checkpoint import (
    TopKCheckpoints,
    is_better,
    monitor_score,
    qualifies_for_topk,
)
from voxwhisper.training.metrics import foreground_channel_dice


def test_monitor_score_loss_is_minimized():
    ckpt = {"monitor": "loss"}
    score, higher, name = monitor_score({"val_loss": 0.4, "dice_patch": 0.8}, ckpt)
    assert score == 0.4
    assert higher is False
    assert name == "val_loss"


def test_monitor_score_patch_dice_is_maximized():
    ckpt = {"monitor": "dice"}
    score, higher, name = monitor_score({"val_loss": 0.4, "dice_patch": 0.8}, ckpt)
    assert score == 0.8
    assert higher is True
    assert name == "dice_patch"


def test_monitor_score_default_is_dice():
    score, higher, name = monitor_score({"val_loss": 0.4, "dice_patch": 0.8}, {})
    assert name == "dice_patch"
    assert higher is True


def test_is_better_first_score_always_wins():
    assert is_better(0.1, None, higher_is_better=False)
    assert is_better(0.1, None, higher_is_better=True)


def test_is_better_loss_and_dice_directions():
    assert is_better(0.2, 0.3, higher_is_better=False)
    assert not is_better(0.4, 0.3, higher_is_better=False)
    assert is_better(0.9, 0.8, higher_is_better=True)
    assert not is_better(0.7, 0.8, higher_is_better=True)


def test_qualifies_for_topk_fills_then_beats_worst():
    assert qualifies_for_topk(0.5, [], 3, higher_is_better=False)
    assert qualifies_for_topk(0.9, [0.3, 0.4], 3, higher_is_better=False)
    assert qualifies_for_topk(0.35, [0.3, 0.4, 0.5], 3, higher_is_better=False)
    assert not qualifies_for_topk(0.6, [0.3, 0.4, 0.5], 3, higher_is_better=False)
    assert qualifies_for_topk(0.85, [0.9, 0.8, 0.7], 3, higher_is_better=True)
    assert not qualifies_for_topk(0.65, [0.9, 0.8, 0.7], 3, higher_is_better=True)
    assert not qualifies_for_topk(None, [], 3, higher_is_better=True)


def test_topk_keeps_three_lowest_losses(tmp_path):
    topk = TopKCheckpoints(k=3, cache_dir=tmp_path)

    def save_fn(path):
        path.write_text(path.name)

    sequence = [(1, 0.50), (2, 0.40), (3, 0.90), (4, 0.30), (5, 0.45)]
    ranks = [
        topk.update(score, False, "val_loss", epoch, save_fn)
        for epoch, score in sequence
    ]
    assert ranks == [1, 1, 3, 1, 3]
    assert [e["epoch"] for e in topk.entries] == [4, 2, 5]
    assert [e["score"] for e in topk.entries] == [0.30, 0.40, 0.45]
    assert not (tmp_path / "vox_whisper_e001.pt").exists()
    assert not (tmp_path / "vox_whisper_e003.pt").exists()
    assert (tmp_path / "vox_whisper_top1.pt").resolve().name == "vox_whisper_e004.pt"
    assert (tmp_path / "vox_whisper_best.pt").resolve().name == "vox_whisper_e004.pt"


def test_topk_keeps_three_highest_dice(tmp_path):
    topk = TopKCheckpoints(k=3, cache_dir=tmp_path)

    def save_fn(path):
        path.write_text(path.name)

    for epoch, score in [(1, 0.70), (2, 0.80), (3, 0.60), (4, 0.90)]:
        topk.update(score, True, "dice_patch", epoch, save_fn)
    assert [e["epoch"] for e in topk.entries] == [4, 2, 1]
    assert not (tmp_path / "vox_whisper_e003.pt").exists()


def test_foreground_channel_dice_ignores_background():
    logits = torch.zeros(1, 3, 2, 2, 2)
    target = torch.zeros(1, 3, 2, 2, 2)
    logits[:, 0] = -10.0
    target[:, 0] = 1.0
    logits[:, 1:] = 10.0
    target[:, 1:] = 1.0
    assert foreground_channel_dice(logits, target) == pytest.approx(1.0)
