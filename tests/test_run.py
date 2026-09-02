"""Tests for timestamped run directory helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.run import (
    create_or_resume_run,
    create_run_dir,
    dataset_name_from_config,
    find_checkpoint_for_eval,
    find_latest_run_dir,
    pick_checkpoint_in_run,
    predictions_dir,
    resolve_run_dir_for_eval,
    resolve_run_name,
    run_family_dir,
    validate_run_name,
)


def _cfg(tmp_path: Path, *, run_name: str = "baseline", processed: str = "processed_T1_FA") -> dict:
    return {
        "data": {
            "paths": {
                "processed": str(tmp_path / processed),
                "cache": str(tmp_path / "cache"),
                "runs": str(tmp_path / "runs"),
            },
            "modalities": {"primary": "t1", "secondary": "fa"},
        },
        "training": {"run_name": run_name, "seed": 42},
    }


def test_validate_run_name_rejects_unsafe():
    with pytest.raises(ValueError):
        validate_run_name("")
    with pytest.raises(ValueError):
        validate_run_name("foo/bar")
    with pytest.raises(ValueError):
        validate_run_name("../x")
    with pytest.raises(ValueError):
        validate_run_name("-leading")
    assert validate_run_name("baseline") == "baseline"
    assert validate_run_name("high_bce.v2") == "high_bce.v2"


def test_dataset_and_family_from_config(tmp_path):
    cfg = _cfg(tmp_path)
    assert dataset_name_from_config(cfg) == "processed_T1_FA"
    family = run_family_dir(cfg)
    assert family == tmp_path / "runs" / "processed_T1_FA" / "baseline"
    assert resolve_run_name(cfg, name_override="ablation") == "ablation"


def test_create_run_dir_writes_config_and_meta(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir = create_run_dir(
        cfg,
        config_path="config/tracts.yaml",
        seed=42,
        argv=["train.py", "--config", "config/tracts.yaml"],
        timestamp="20260902_130304",
    )
    assert run_dir.name == "20260902_130304"
    assert run_dir.parent.name == "baseline"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "meta.json").exists()

    snap = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert snap["training"]["run_name"] == "baseline"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["dataset"] == "processed_T1_FA"
    assert meta["run_name"] == "baseline"
    assert meta["seed"] == 42


def test_resume_picks_newest_timestamp(tmp_path):
    cfg = _cfg(tmp_path)
    older = create_run_dir(cfg, timestamp="20260901_100000")
    newer = create_run_dir(cfg, timestamp="20260902_120000")
    (older / "vox_whisper_latest.pt").write_bytes(b"old")
    (newer / "vox_whisper_latest.pt").write_bytes(b"new")

    resumed = create_or_resume_run(cfg, resume=True)
    assert resumed == newer
    assert find_latest_run_dir(run_family_dir(cfg)) == newer


def test_resume_without_prior_creates_new(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    run_dir = create_or_resume_run(cfg, resume=True)
    assert run_dir.is_dir()
    assert (run_dir / "config.yaml").exists()
    out = capsys.readouterr().out
    assert "no prior run" in out.lower() or "starting a new run" in out.lower()


def test_legacy_flat_family_resume(tmp_path):
    """Pre-layout folders with latest.pt at the family root still resume."""
    cfg = _cfg(tmp_path)
    family = run_family_dir(cfg)
    family.mkdir(parents=True)
    (family / "vox_whisper_latest.pt").write_bytes(b"legacy")
    resumed = create_or_resume_run(cfg, resume=True)
    assert resumed == family


def test_find_checkpoint_prefers_best(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir = create_run_dir(cfg, timestamp="20260902_150000")
    (run_dir / "vox_whisper_latest.pt").write_bytes(b"latest")
    (run_dir / "vox_whisper_best.pt").write_bytes(b"best")
    (run_dir / "vox_whisper_epoch_10.pt").write_bytes(b"ep")

    assert pick_checkpoint_in_run(run_dir).name == "vox_whisper_best.pt"
    ckpt = find_checkpoint_for_eval(cfg)
    assert ckpt.name == "vox_whisper_best.pt"

    explicit = find_checkpoint_for_eval(cfg, explicit=str(run_dir / "vox_whisper_latest.pt"))
    assert explicit.name == "vox_whisper_latest.pt"

    via_dir = find_checkpoint_for_eval(cfg, run_dir=run_dir)
    assert via_dir.name == "vox_whisper_best.pt"


def test_name_override_changes_family(tmp_path):
    cfg = _cfg(tmp_path, run_name="baseline")
    run_dir = create_run_dir(cfg, name_override="high_bce", timestamp="20260902_160000")
    assert "high_bce" in run_dir.parts
    assert "baseline" not in run_dir.parts


def test_predictions_dir(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir = create_run_dir(cfg, timestamp="20260902_170000")
    assert predictions_dir(run_dir, "test") == run_dir / "predictions" / "test"
    assert predictions_dir(run_dir, "val") == run_dir / "predictions" / "val"


def test_resolve_run_dir_for_eval_returns_latest(tmp_path):
    cfg = _cfg(tmp_path)
    older = create_run_dir(cfg, timestamp="20260901_100000")
    newer = create_run_dir(cfg, timestamp="20260902_120000")
    (older / "vox_whisper_latest.pt").write_bytes(b"old")
    (newer / "vox_whisper_latest.pt").write_bytes(b"new")

    assert resolve_run_dir_for_eval(cfg) == newer


def test_resolve_run_dir_for_eval_explicit(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir = create_run_dir(cfg, timestamp="20260902_150000")
    assert resolve_run_dir_for_eval(cfg, run_dir=run_dir) == run_dir


def test_resolve_run_dir_for_eval_none_when_no_runs(tmp_path):
    cfg = _cfg(tmp_path)
    assert resolve_run_dir_for_eval(cfg) is None
