"""Timestamped training run directories under ``runs/{dataset}/{run_name}/``.

Layout
------
Each ``train.py`` invocation writes to::

    {data.paths.runs}/{Path(data.paths.processed).name}/{run_name}/{YYYYMMDD_HHMMSS}/
        config.yaml
        meta.json
        metrics.jsonl
        vox_whisper_*.pt
        vox_whisper_topk.json
        predictions/{split}/
            {subject_id}/pred_labels.nii.gz
            dice_per_subject.csv
            dice_summary.csv
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from voxwhisper.config import ensure_dir, get_project_root, resolve_path

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
_LATEST_CKPT = "vox_whisper_latest.pt"
_PREFERRED_CKPTS = ("vox_whisper_best.pt", "vox_whisper_top1.pt")


# ---------------------------------------------------------------------------
# Name / path helpers
# ---------------------------------------------------------------------------

def validate_run_name(name: str) -> str:
    """Return a validated, stripped run name or raise ``ValueError``."""
    if not isinstance(name, str):
        raise ValueError("run_name must be a string")
    name = name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("run_name must be a non-empty, non-relative string")
    if "/" in name or "\\" in name:
        raise ValueError(f"run_name must not contain path separators: {name!r}")
    if not _RUN_NAME_RE.match(name):
        raise ValueError(
            f"run_name must be filesystem-safe "
            f"(letters, digits, '.', '_', '-'; start with alnum): {name!r}"
        )
    return name


def resolve_run_name(config: Mapping[str, Any]) -> str:
    """Return validated run name from ``training.run_name``."""
    raw = config.get("training", {}).get("run_name", "baseline")
    return validate_run_name(str(raw))


def dataset_name_from_config(config: Mapping[str, Any]) -> str:
    """Leaf of ``data.paths.processed`` used as the dataset folder."""
    name = Path(str(config["data"]["paths"]["processed"])).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Cannot derive dataset name from data.paths.processed={name!r}")
    return name


def runs_root(config: Mapping[str, Any]) -> Path:
    """Absolute path to ``data.paths.runs`` (defaults to ``runs/`` under project root)."""
    paths = config.get("data", {}).get("paths", {})
    if paths.get("runs"):
        return resolve_path(config, "data.paths.runs")
    return get_project_root() / "runs"


def run_family_dir(config: Mapping[str, Any]) -> Path:
    """``{runs}/{dataset}/{run_name}`` — parent of timestamped children."""
    return runs_root(config) / dataset_name_from_config(config) / resolve_run_name(config)


def predictions_dir(run_dir: Path, split: str = "test") -> Path:
    """``{run_dir}/predictions/{split}/`` — where eval NIfTIs and CSVs are written."""
    return run_dir / "predictions" / split


def _as_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else get_project_root() / path


# ---------------------------------------------------------------------------
# Timestamp children
# ---------------------------------------------------------------------------

def list_timestamp_runs(family_dir: Path) -> list[Path]:
    """Sorted timestamp child dirs under a family (oldest first)."""
    if not family_dir.is_dir():
        return []
    return sorted(
        (p for p in family_dir.iterdir() if p.is_dir() and _TIMESTAMP_RE.match(p.name)),
        key=lambda p: p.name,
    )


def find_latest_run_dir(family_dir: Path) -> Optional[Path]:
    """Newest timestamp child that contains ``vox_whisper_latest.pt``."""
    for path in reversed(list_timestamp_runs(family_dir)):
        if (path / _LATEST_CKPT).exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Snapshots written at run creation
# ---------------------------------------------------------------------------

def write_config_snapshot(run_dir: Path, config: Mapping[str, Any]) -> Path:
    """Dump the resolved config (including injected prompts) as ``config.yaml``."""
    path = run_dir / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, default_flow_style=False, sort_keys=False)
    return path


def write_meta(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    run_name: str,
    dataset: str,
    config_path: Optional[str | Path] = None,
    seed: Optional[int] = None,
    argv: Optional[Sequence[str]] = None,
    resume: bool = False,
) -> Path:
    """Write ``meta.json`` next to checkpoints."""
    meta: dict[str, Any] = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "dataset": dataset,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config_path": str(config_path) if config_path else None,
        "seed": seed,
        "argv": list(argv) if argv is not None else list(sys.argv),
        "resume": resume,
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "modalities": {"primary": "t1", "secondary": "fa"},
    }
    # Best-effort git info
    try:
        root = get_project_root()
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                              stderr=subprocess.DEVNULL, text=True).strip())
        meta["git_sha"] = sha
        meta["git_dirty"] = dirty
    except (OSError, subprocess.CalledProcessError):
        meta["git_sha"] = meta["git_dirty"] = None
    # Best-effort torch info
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        meta["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        meta.update({"torch_version": None, "cuda_available": False, "cuda_device_count": 0})

    path = run_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Run creation / resume
# ---------------------------------------------------------------------------

def create_run_dir(
    config: Mapping[str, Any],
    *,
    config_path: Optional[str | Path] = None,
    seed: Optional[int] = None,
    argv: Optional[Sequence[str]] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Create a fresh timestamped run directory and write config + meta snapshots."""
    run_name = resolve_run_name(config)
    dataset = dataset_name_from_config(config)
    stamp = timestamp or datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    if not _TIMESTAMP_RE.match(stamp):
        raise ValueError(f"timestamp must be YYYYMMDD_HHMMSS, got {stamp!r}")
    run_dir = ensure_dir(run_family_dir(config) / stamp)
    write_config_snapshot(run_dir, config)
    write_meta(run_dir, config=config, run_name=run_name, dataset=dataset,
               config_path=config_path, seed=seed, argv=argv, resume=False)
    return run_dir


def create_or_resume_run(
    config: Mapping[str, Any],
    *,
    resume: bool = False,
    run_dir: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    seed: Optional[int] = None,
    argv: Optional[Sequence[str]] = None,
) -> Path:
    """Resolve the run directory for a training launch.

    Priority
    --------
    1. Explicit ``run_dir`` — created if absent.
    2. ``--resume`` → newest timestamp child with latest.pt.
    3. Otherwise create a fresh timestamped directory.
    """
    if run_dir is not None:
        path = _as_project_path(run_dir)
        ensure_dir(path)
        if not (path / "config.yaml").exists():
            write_config_snapshot(path, config)
        if not (path / "meta.json").exists():
            write_meta(path, config=config,
                       run_name=resolve_run_name(config),
                       dataset=dataset_name_from_config(config),
                       config_path=config_path, seed=seed, argv=argv, resume=resume)
        return path

    family = run_family_dir(config)

    if resume:
        latest = find_latest_run_dir(family)
        if latest is not None:
            return latest
        print(f"Warning: --resume requested but no prior run found under {family} — starting a new run")

    return create_run_dir(config, config_path=config_path, seed=seed, argv=argv)


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def pick_checkpoint_in_run(run_dir: Path) -> Optional[Path]:
    """Return best / top1 / newest periodic / latest checkpoint inside a run directory."""
    for name in _PREFERRED_CKPTS:
        path = run_dir / name
        if path.exists():
            return path
    periodic = list(run_dir.glob("vox_whisper_epoch_*.pt"))
    if periodic:
        return max(periodic, key=_checkpoint_epoch)
    latest = run_dir / _LATEST_CKPT
    return latest if latest.exists() else None


def resolve_run_dir_for_eval(
    config: Mapping[str, Any],
    *,
    run_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    """Return the run directory for evaluation, or ``None`` for standalone --checkpoint usage."""
    if run_dir is not None:
        path = _as_project_path(run_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"--run-dir not found: {path}")
        return path
    family = run_family_dir(config)
    return find_latest_run_dir(family)


def find_checkpoint_for_eval(
    config: Mapping[str, Any],
    *,
    explicit: Optional[str] = None,
    run_dir: Optional[str | Path] = None,
) -> Path:
    """Resolve a checkpoint for evaluation.

    Order: explicit path → ``--run-dir`` → latest timestamp under family.
    """
    if explicit:
        path = _as_project_path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    eval_run = resolve_run_dir_for_eval(config, run_dir=run_dir)
    if eval_run is not None:
        ckpt = pick_checkpoint_in_run(eval_run)
        if ckpt is not None:
            return ckpt

    family = run_family_dir(config)
    raise FileNotFoundError(
        f"No checkpoint found under {family}. "
        "Train first or pass --checkpoint / --run-dir."
    )
