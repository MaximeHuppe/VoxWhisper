"""Build integer label maps on the T1 grid from tract masks."""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from voxwhisper.config import (
    PRIMARY_MODALITY,
    ensure_dir,
    load_structures,
    resolve_path,
)
from voxwhisper.data.nifti_io import (
    mask_path,
    resolve_raw_volume_path,
    save_nifti,
    subject_processed_dir,
)


def find_mask_path(mask_dir: str, subject_id: str, filename_template: str):
    filename = filename_template.format(subject_id=subject_id)
    for candidate in (
        os.path.join(mask_dir, subject_id, filename),
        os.path.join(mask_dir, filename),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def process_mask(subject_id: str, raw_dir: str, output_dir, config: dict, structures: dict) -> None:
    """Process tract masks for a given subject."""
    print(f"Processing tract masks: {subject_id}")

    primary_cfg = config["data"]["volumes"][PRIMARY_MODALITY]
    primary_path = resolve_raw_volume_path(raw_dir, subject_id, primary_cfg["filename"])
    if primary_path is None:
        print(f"Warning: T1 reference missing for {subject_id}. Skipping.")
        return

    t1_img = nib.load(str(primary_path))
    label_data = np.zeros(t1_img.shape[:3], dtype=np.uint8)

    mask_dir = os.path.join(raw_dir, subject_id, config["data"]["masks"]["source"])

    for name, label_val in structures["foreground"]:
        src = os.path.join(mask_dir, f"{name}.nii.gz")
        if not os.path.exists(src):
            print(f"  Warning: missing {name}")
            continue

        resampled = resample_from_to(nib.load(src), t1_img, order=0)
        binary = resampled.get_fdata() > 0

        overlap = binary & (label_data > 0)
        if overlap.any():
            print(f"  Warning: {int(overlap.sum())} overlapping voxels for {name}")

        label_data[binary] = label_val

    if not np.any(label_data):
        print(f"Warning: no tract masks found for {subject_id}. Skipping.")
        return

    ensure_dir(subject_processed_dir(output_dir, subject_id))
    out_file = mask_path(output_dir, subject_id)
    save_nifti(label_data, out_file, affine=t1_img.affine, dtype=np.uint8)

    uniq, counts = np.unique(label_data, return_counts=True)
    print(
        f"  Saved: {out_file} shape={label_data.shape} "
        f"labels={dict(zip(uniq.astype(int).tolist(), counts.tolist()))}"
    )


def preprocess_masks(config: dict) -> None:
    """Preprocess all tract masks for all subjects."""
    raw_dir = resolve_path(config, "data.paths.raw")
    output_dir = resolve_path(config, "data.paths.processed")
    ensure_dir(output_dir)

    structures = load_structures(config)
    if not structures:
        print("Error: no structures configured in config.")
        sys.exit(1)

    if not raw_dir.exists():
        print(f"Error: raw data directory not found: {raw_dir}")
        sys.exit(1)

    subjects = sorted(
        d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))
    )

    print(f"Processing masks for {len(subjects)} subjects (full T1 resolution)")
    for subject_id in subjects:
        process_mask(subject_id, str(raw_dir), output_dir, config, structures)
