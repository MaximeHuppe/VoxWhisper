# preprocess/preprocess_masks.py
"""Resample OpticNerveSeg labels onto the full T1 grid (no patch crop)."""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import (  # noqa: E402
    ensure_dir,
    load_config,
    parse_config_args,
    resolve_path,
)
from src.utils.nifti_io import (  # noqa: E402
    mask_path,
    save_nifti,
    subject_processed_dir,
)


def find_mask_path(mask_dir, subject_id, filename_template):
    filename = filename_template.format(subject_id=subject_id)
    candidate = os.path.join(mask_dir, subject_id, filename)
    if os.path.exists(candidate):
        return candidate

    flat = os.path.join(mask_dir, filename)
    if os.path.exists(flat):
        return flat
    return None


def resolve_reference_t1(raw_dir, subject_id, t1_cfg):
    """Locate the T1 volume used as the resampling reference grid."""
    primary = os.path.join(raw_dir, subject_id, "T1w", t1_cfg["filename"])
    if os.path.exists(primary):
        return primary
    return None


def process_subject_mask(subject_id, mask_dir, raw_dir, output_dir, config):
    print(f"Processing mask: {subject_id}")

    mask_cfg = config["data"]["masks"]
    t1_cfg = config["data"]["volumes"]["t1"]

    src_mask = find_mask_path(mask_dir, subject_id, mask_cfg["filename"])
    if src_mask is None:
        print(f"Warning: mask missing for {subject_id}. Skipping.")
        return

    t1_path = resolve_reference_t1(raw_dir, subject_id, t1_cfg)
    if t1_path is None:
        print(f"Warning: T1 reference missing for {subject_id}. Skipping mask.")
        return

    label_img = nib.load(src_mask)
    t1_img = nib.load(t1_path)

    # Resample integer labels onto the full T1 grid (nearest-neighbor)
    resampled = resample_from_to(label_img, t1_img, order=0)
    label_data = np.rint(resampled.get_fdata()).astype(np.uint8)

    ensure_dir(subject_processed_dir(output_dir, subject_id))
    out_file = mask_path(output_dir, subject_id)
    save_nifti(label_data, out_file, affine=t1_img.affine, dtype=np.uint8)

    uniq, counts = np.unique(label_data, return_counts=True)
    print(
        f"  Saved: {out_file} shape={label_data.shape} "
        f"labels={dict(zip(uniq.astype(int).tolist(), counts.tolist()))}"
    )


def preprocess_masks(config):
    mask_dir = resolve_path(config, "data.paths.raw_masks")
    raw_dir = resolve_path(config, "data.paths.raw")
    output_dir = resolve_path(config, "data.paths.processed")
    ensure_dir(output_dir)

    if not mask_dir.exists():
        print(f"Error: mask directory not found: {mask_dir}")
        sys.exit(1)

    if raw_dir.exists():
        subjects = sorted(
            d
            for d in os.listdir(raw_dir)
            if os.path.isdir(os.path.join(raw_dir, d))
        )
    else:
        subjects = sorted(
            d
            for d in os.listdir(mask_dir)
            if os.path.isdir(os.path.join(mask_dir, d))
        )

    print(f"Processing masks for {len(subjects)} subjects (full T1 resolution)")
    for subject_id in subjects:
        process_subject_mask(
            subject_id, str(mask_dir), str(raw_dir), output_dir, config
        )


if __name__ == "__main__":
    args = parse_config_args(description="Preprocess full-resolution segmentation masks")
    cfg = load_config(args.config)
    preprocess_masks(cfg)
