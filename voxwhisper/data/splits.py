"""Subject-level train/val/test split utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from voxwhisper.config import get_project_root, resolve_path


def list_processed_subjects(config: Mapping) -> list[str]:
    """Return sorted subject IDs with processed NIfTI (or legacy NPZ) data."""
    from voxwhisper.data.nifti_io import list_subject_ids

    processed_dir = resolve_path(config, "data.paths.processed")
    return list_subject_ids(processed_dir)


def create_splits(subjects: list[str], config: Mapping) -> dict[str, list[str]]:
    """Create a subject-level train/val/test split from config ratios."""
    splits_cfg = config["splits"]
    train_ratio = float(splits_cfg["train_ratio"])
    val_ratio = float(splits_cfg["val_ratio"])
    test_ratio = float(splits_cfg["test_ratio"])

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    rng = np.random.default_rng(int(splits_cfg["seed"]))
    shuffled = list(subjects)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Ensure test gets the remainder so counts sum to n
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def create_or_load_splits(config: Mapping) -> dict[str, list[str]]:
    """Load an existing split manifest, or create and save one if missing."""
    manifest_rel = config["splits"]["manifest"]
    manifest_path = Path(manifest_rel)
    if not manifest_path.is_absolute():
        manifest_path = get_project_root() / manifest_path

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    subjects = list_processed_subjects(config)
    if len(subjects) < 3:
        raise ValueError(
            f"Need at least 3 processed subjects to create splits, found {len(subjects)}"
        )

    splits = create_splits(subjects, config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote split manifest to {manifest_path}")
    print(
        f"  train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])}"
    )
    return splits
