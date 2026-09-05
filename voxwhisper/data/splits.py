"""Subject-level train/val/test split utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from voxwhisper.util.config import get_project_root, resolve_path


def _as_repo_path(rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else get_project_root() / path


def list_raw_subject_ids(config: Mapping) -> list[str]:
    """Subject folders under ``data.paths.raw`` (volume root, e.g. HCP)."""
    raw_dir = resolve_path(config, "data.paths.raw")
    if not raw_dir.exists():
        return []
    return sorted(d.name for d in raw_dir.iterdir() if d.is_dir())


def nerve_masks_root(config: Mapping) -> Path:
    """Directory where Phase-2 holdout mask folders live.

    Prefer ``data.nerve_masks.root`` (e.g. ``data/raw``); fall back to
    ``data.paths.raw`` when unset.
    """
    rel = config.get("data", {}).get("nerve_masks", {}).get("root")
    if rel:
        path = Path(str(rel))
        return path if path.is_absolute() else get_project_root() / path
    return resolve_path(config, "data.paths.raw")


def subject_has_nerve_masks(masks_dir: Path, subject_id: str, source: str) -> bool:
    """True when ``{masks_dir}/{sid}/{source}`` exists and contains a NIfTI."""
    folder = masks_dir / subject_id / source
    nested = masks_dir / subject_id / "T1w" / source
    if nested.is_dir():
        folder = nested
    if not folder.is_dir():
        return False
    return any(folder.glob("*.nii.gz")) or any(folder.glob("*.nii"))


def build_subject_split(config: Mapping) -> dict[str, list[str]]:
    """Partition volume subjects into ``pretrain`` vs ``nerve`` by mask presence.

    Subjects are listed from ``data.paths.raw`` (HCP volumes). Holdouts are
    those with a NIfTI under ``data.nerve_masks.root`` / ``source`` (typically
    ``data/raw`` / ``tract_masks_1.25``).
    """
    masks_dir = nerve_masks_root(config)
    source = str(
        config.get("data", {}).get("nerve_masks", {}).get("source", "tract_masks_1.25")
    )
    pretrain: list[str] = []
    nerve: list[str] = []
    for sid in list_raw_subject_ids(config):
        if subject_has_nerve_masks(masks_dir, sid, source):
            nerve.append(sid)
        else:
            pretrain.append(sid)
    return {"pretrain": pretrain, "nerve": nerve}


def create_or_load_subject_split(config: Mapping) -> dict[str, list[str]]:
    """Load or write ``splits.subject_split`` (pretrain vs held-out nerve subjects)."""
    rel = config.get("splits", {}).get("subject_split", "config/subject_split.json")
    path = _as_repo_path(str(rel))
    if path.exists():
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return {
            "pretrain": list(payload.get("pretrain", [])),
            "nerve": list(payload.get("nerve", [])),
        }

    payload = build_subject_split(config)
    if not payload["pretrain"] and not payload["nerve"]:
        return payload

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote subject split to {path}")
    print(f"  pretrain={len(payload['pretrain'])} nerve={len(payload['nerve'])}")
    return payload


def list_processed_subjects(config: Mapping) -> list[str]:
    """Return processed subject IDs restricted to this stage's cohort."""
    from voxwhisper.data.nifti_io import list_subject_ids
    from voxwhisper.util.stage import cohort_name

    processed_dir = resolve_path(config, "data.paths.processed")
    available = list_subject_ids(processed_dir)
    stage = create_or_load_subject_split(config)
    allowed = set(stage[cohort_name(config)])
    if not allowed:
        return available
    return [s for s in available if s in allowed]


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
    manifest_path = _as_repo_path(str(manifest_rel))

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    subjects = list_processed_subjects(config)
    if len(subjects) < 3:
        from voxwhisper.util.stage import cohort_name
        raise ValueError(
            f"Need at least 3 processed {cohort_name(config)} subjects to create splits, "
            f"found {len(subjects)}"
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
