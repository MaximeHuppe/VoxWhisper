"""Build integer label maps on the T1 grid.

Phase 1 (``data.volumes.wmparc``): nearest-neighbour resample wmparc, collapse
to the SynthSeg/OpenMind dense set, apply brainmask.

Phase 2 (``data.masks.source``): stack per-structure NIfTIs (nerve masks) onto
the T1 grid with nearest neighbour.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from tqdm import tqdm

from voxwhisper.util.config import PRIMARY_MODALITY, ensure_dir, load_structures, resolve_path
from voxwhisper.data.nifti_io import (
    mask_path,
    resolve_raw_volume_path,
    save_nifti,
    subject_processed_dir,
)
from voxwhisper.data.preprocess.freesurfer import collapse_wmparc
from voxwhisper.util.stage import mask_kind


def _resampled_brainmask(raw_dir: str, subject_id: str, config: dict, t1_img):
    brain_cfg = config.get("data", {}).get("volumes", {}).get("brainmask")
    if not brain_cfg:
        return None
    path = resolve_raw_volume_path(raw_dir, subject_id, brain_cfg["filename"])
    if path is None:
        return None
    resampled = resample_from_to(nib.load(str(path)), t1_img, order=0)
    return resampled.get_fdata() > 0


def _named_mask_dir(raw_dir: str, subject_id: str, source: str) -> Path:
    root = Path(raw_dir) / subject_id
    nested = root / "T1w" / source
    if nested.is_dir():
        return nested
    return root / source


def process_dense_mask(subject_id: str, raw_dir: str, output_dir, config: dict) -> None:
    """Resample wmparc onto T1, collapse to dense ids, apply brainmask."""
    # print(f"Processing dense labels: {subject_id}")

    t1_cfg = config["data"]["volumes"][PRIMARY_MODALITY]
    t1_path = resolve_raw_volume_path(raw_dir, subject_id, t1_cfg["filename"])
    if t1_path is None:
        print(f"Warning: T1 reference missing for {subject_id}. Skipping.")
        return

    wmparc_cfg = config["data"]["volumes"]["wmparc"]
    wmparc_path = resolve_raw_volume_path(raw_dir, subject_id, wmparc_cfg["filename"])
    if wmparc_path is None:
        print(f"Warning: wmparc missing for {subject_id}. Skipping.")
        return

    t1_img = nib.load(str(t1_path))
    wmparc_img = nib.load(str(wmparc_path))
    resampled = resample_from_to(wmparc_img, t1_img, order=0)
    collapsed = collapse_wmparc(resampled.get_fdata())

    if bool(config.get("preprocessing", {}).get("apply_brainmask", True)):
        brain = _resampled_brainmask(raw_dir, subject_id, config, t1_img)
        if brain is not None and brain.shape == collapsed.shape:
            collapsed = collapsed.copy()
            collapsed[~brain] = 0

    _save_label_map(collapsed, t1_img, output_dir, subject_id)


def process_named_masks(
    subject_id: str,
    raw_dir: str,
    output_dir,
    config: dict,
    structures: dict,
) -> None:
    """Stack per-structure NIfTIs (nerves / named masks) onto the T1 grid."""
    # print(f"Processing named masks: {subject_id}")

    t1_cfg = config["data"]["volumes"][PRIMARY_MODALITY]
    t1_path = resolve_raw_volume_path(raw_dir, subject_id, t1_cfg["filename"])
    if t1_path is None:
        print(f"Warning: T1 reference missing for {subject_id}. Skipping.")
        return

    t1_img = nib.load(str(t1_path))
    label_data = np.zeros(t1_img.shape[:3], dtype=np.uint8)
    source = str(config["data"]["masks"]["source"])
    mask_dir = _named_mask_dir(raw_dir, subject_id, source)

    for name, label_val in structures["foreground"]:
        src = mask_dir / f"{name}.nii.gz"
        if not src.exists():
            src = mask_dir / f"{name}.nii"
        if not src.exists():
            print(f"  Warning: missing {name}")
            continue

        resampled = resample_from_to(nib.load(str(src)), t1_img, order=0)
        binary = resampled.get_fdata() > 0
        overlap = binary & (label_data > 0)
        if overlap.any():
            print(f"  Warning: {int(overlap.sum())} overlapping voxels for {name}")
        label_data[binary] = int(label_val)

    if not np.any(label_data):
        print(f"Warning: no named masks found for {subject_id}. Skipping.")
        return

    _save_label_map(label_data, t1_img, output_dir, subject_id)


def _save_label_map(label_data, t1_img, output_dir, subject_id: str) -> None:
    ensure_dir(subject_processed_dir(output_dir, subject_id))
    out_file = mask_path(output_dir, subject_id)
    save_nifti(label_data, out_file, affine=t1_img.affine, dtype=np.uint8)
    uniq, counts = np.unique(label_data, return_counts=True)
    # print(
    #     f"  Saved: {out_file} shape={label_data.shape} "
    #     f"labels={dict(zip(uniq.astype(int).tolist(), counts.tolist()))}"
    # )


def process_mask(subject_id: str, raw_dir: str, output_dir, config: dict) -> None:
    """Process one subject's labels for the configured stage."""
    kind = mask_kind(config)
    if kind == "named":
        structures = load_structures(config)
        if not structures:
            raise ValueError("Named-mask stage requires data.masks.structures")
        process_named_masks(subject_id, raw_dir, output_dir, config, structures)
        return
    process_dense_mask(subject_id, raw_dir, output_dir, config)


def preprocess_masks(config: dict, subject_ids=None) -> None:
    """Preprocess integer masks for all (or a filtered list of) subjects."""
    raw_dir = resolve_path(config, "data.paths.raw")
    output_dir = resolve_path(config, "data.paths.processed")
    ensure_dir(output_dir)

    if not raw_dir.exists():
        print(f"Error: raw data directory not found: {raw_dir}")
        sys.exit(1)

    kind = mask_kind(config)
    if kind == "named" and not load_structures(config):
        print("Error: no structures configured in config.")
        sys.exit(1)

    subjects = sorted(
        d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))
    )
    if subject_ids is not None:
        allowed = set(subject_ids)
        subjects = [s for s in subjects if s in allowed]

    t1_filename = config["data"]["volumes"][PRIMARY_MODALITY]["filename"]
    with_t1 = [
        s for s in subjects
        if resolve_raw_volume_path(raw_dir, s, t1_filename) is not None
    ]
    skipped = len(subjects) - len(with_t1)
    if skipped:
        print(f"Skipping {skipped} subject(s) with no T1 — masks not computed")
    subjects = with_t1

    label = "named nerve masks" if kind == "named" else "dense wmparc masks"
    print(f"Processing {label} for {len(subjects)} subjects (NN onto T1)")
    for subject_id in tqdm(subjects, desc="Processing masks"):
        process_mask(subject_id, str(raw_dir), output_dir, config)
