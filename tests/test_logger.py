"""TrainingLogger JSONL handling of nested per-tract Dice dicts."""
from __future__ import annotations

import json

from voxwhisper.training.logger import METRICS_FILENAME, TrainingLogger, _format_class_dice, _jsonify_metric


def test_jsonify_nested_class_dice():
    payload = _jsonify_metric({"ATR_left": 0.1234567, "ATR_right": 0.0})
    assert payload == {"ATR_left": 0.123457, "ATR_right": 0.0}


def test_format_class_dice():
    assert _format_class_dice({}) == "[]"
    assert _format_class_dice({"ATR_left": 0.821, "UF_right": 0.0}) == (
        "[ATR_left=0.821 UF_right=0.000]"
    )


def test_logger_writes_class_dice_dicts(tmp_path, capsys):
    logger = TrainingLogger(tmp_path, total_epochs=10)
    logger.log_epoch(
        1,
        {
            "train_loss": 1.0,
            "val_loss": 0.9,
            "dice_patch": 0.14,
            "dice_patch_classes": {"ATR_left": 0.84, "ATR_right": 0.0},
        },
        lr=1e-4,
        rank=1,
    )
    logger.close()

    path = tmp_path / METRICS_FILENAME
    assert path.exists()
    record = json.loads(path.read_text().strip())
    assert record["dice_patch"] == 0.14
    assert record["dice_patch_classes"]["ATR_left"] == 0.84

    stdout = capsys.readouterr().out
    assert "dice_patch 0.1400" in stdout
    assert "ATR_left=0.840" in stdout
    assert "[top-1]" in stdout

    resumed = TrainingLogger(tmp_path, total_epochs=10, resume=True)
    bests = resumed.best()
    resumed.close()
    assert "dice_patch" in bests
    assert "dice_patch_classes" not in bests


def test_logger_resume_appends(tmp_path):
    logger = TrainingLogger(tmp_path, total_epochs=5)
    logger.log_epoch(1, {"train_loss": 1.0}, lr=1e-4)
    logger.close()

    resumed = TrainingLogger(tmp_path, total_epochs=5, resume=True)
    resumed.log_epoch(2, {"train_loss": 0.5}, lr=1e-4)
    resumed.close()

    lines = (tmp_path / METRICS_FILENAME).read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["epoch"] == 1
    assert json.loads(lines[1])["epoch"] == 2
