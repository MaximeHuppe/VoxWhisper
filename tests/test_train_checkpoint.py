"""Checkpoint monitor selection and patch Dice helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.checkpoint import (
    TopKCheckpoints,
    checkpoint_config,
    is_better,
    monitor_score,
    qualifies_for_topk,
    should_eval_volume,
    should_save_periodic,
)
from src.utils.metrics import foreground_channel_dice


def test_checkpoint_config_defaults():
    cfg = checkpoint_config({})
    assert cfg["monitor"] == "loss"
    assert cfg["dice_scope"] == "patch"
    assert cfg["every"] == 10
    assert cfg["volume_every"] == 10
    assert cfg["keep"] == 3
    assert cfg["keep_periodic"] is True


def test_checkpoint_config_legacy_checkpoint_every():
    cfg = checkpoint_config({"checkpoint_every": 5})
    assert cfg["every"] == 5
    assert cfg["volume_every"] == 5


def test_volume_every_defaults_to_every():
    cfg = checkpoint_config({"checkpoint": {"every": 20, "monitor": "dice"}})
    assert cfg["every"] == 20
    assert cfg["volume_every"] == 20


def test_checkpoint_config_rejects_bad_monitor():
    with pytest.raises(ValueError, match="monitor"):
        checkpoint_config({"checkpoint": {"monitor": "acc"}})


def test_checkpoint_config_rejects_bad_scope():
    with pytest.raises(ValueError, match="dice_scope"):
        checkpoint_config({"checkpoint": {"monitor": "dice", "dice_scope": "subject"}})


def test_volume_eval_is_periodic_and_1_based():
    assert not should_eval_volume(0, 10)
    assert should_eval_volume(9, 10)
    assert not should_eval_volume(10, 10)
    assert should_eval_volume(19, 10)


def test_should_save_periodic():
    ckpt = checkpoint_config({"checkpoint": {"every": 10, "keep_periodic": True}})
    assert not should_save_periodic(0, ckpt)
    assert should_save_periodic(9, ckpt)
    ckpt["keep_periodic"] = False
    assert not should_save_periodic(9, ckpt)


def test_monitor_score_loss_is_minimized():
    ckpt = checkpoint_config({"checkpoint": {"monitor": "loss"}})
    score, higher, name = monitor_score({"val_loss": 0.4, "dice_patch": 0.8}, ckpt)
    assert score == 0.4
    assert higher is False
    assert name == "val_loss"


def test_monitor_score_patch_dice_is_maximized():
    ckpt = checkpoint_config(
        {"checkpoint": {"monitor": "dice", "dice_scope": "patch"}}
    )
    score, higher, name = monitor_score({"val_loss": 0.4, "dice_patch": 0.8}, ckpt)
    assert score == 0.8
    assert higher is True
    assert name == "dice_patch"


def test_monitor_score_volume_missing_when_not_computed():
    ckpt = checkpoint_config(
        {"checkpoint": {"monitor": "dice", "dice_scope": "volume"}}
    )
    score, higher, name = monitor_score({"val_loss": 0.4, "dice_patch": 0.8}, ckpt)
    assert score is None
    assert higher is True
    assert name == "dice_volume"


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

    # epoch, loss
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
    # Perfect foreground match; background is all-wrong.
    logits[:, 0] = -10.0
    target[:, 0] = 1.0
    logits[:, 1:] = 10.0
    target[:, 1:] = 1.0
    assert foreground_channel_dice(logits, target) == pytest.approx(1.0)
