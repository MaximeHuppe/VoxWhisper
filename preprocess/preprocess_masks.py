# preprocess/preprocess_masks.py
"""Build integer label maps on the T1 grid (tract merge or legacy single-mask)."""
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
    load_structures,
    parse_config_args,
    resolve_path,
    active_modality_keys,
)
from src.data.nifti_io import (  # noqa: E402
    mask_path,
    resolve_raw_volume_path,
    save_nifti,
    subject_processed_dir,
)


########################################################
#                PATH UTILITY FUNCTIONS                #
########################################################

def find_mask_path(mask_dir, subject_id, filename_template):
    filename = filename_template.format(subject_id=subject_id)
    for candidate in (
        os.path.join(mask_dir, subject_id, filename),
        os.path.join(mask_dir, filename),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


########################################################
#                MASK PROCESSING FUNCTIONS             #
########################################################


def process_mask(subject_id, raw_dir, output_dir, config, structures):
    """
    Process tract masks for a given subject.
    
    Steps:
    1. Load T1 reference volume
    2. Load tract masks
    3. Resample tract masks to T1 reference volume (order=0)
    4. Create integer label map
    5. Save label map
    """

    print(f"Processing tract masks: {subject_id}")

    # 1. Load T1 reference volume
    primary, _ = active_modality_keys(config)
    primary_cfg = config["data"]["volumes"][primary]
    primary_path = resolve_raw_volume_path(raw_dir, subject_id, primary_cfg["filename"])
    if primary_path is None:
        print(f"Warning: T1 reference missing for {subject_id}. Skipping.")
        return

    t1_img = nib.load(str(primary_path))

    # 2. Create integer label map
    label_data = np.zeros(t1_img.shape[:3], dtype=np.uint8)

    # 3. Get the mask folder path (for the given subject)
    mask_dir = os.path.join(raw_dir, subject_id, config["data"]["masks"]["source"])

    # 4. Process each mask
    for name, label_val in structures["foreground"]:

        # 4.1. Get the mask path
        src = os.path.join(mask_dir, f"{name}.nii.gz")
        if not os.path.exists(src):
            print(f"  Warning: missing {name}")
            continue

        # 4.2. Resample the mask to the T1 reference volume
        resampled = resample_from_to(nib.load(src), t1_img, order=0)

        # 4.3. Create binary mask
        binary = resampled.get_fdata() > 0

        # 4.4. Check for overlapping voxels
        overlap = binary & (label_data > 0)
        if overlap.any():
            print(f"  Warning: {int(overlap.sum())} overlapping voxels for {name}")

        # 4.5. Update the label map
        label_data[binary] = label_val

    if not np.any(label_data):
        print(f"Warning: no tract masks found for {subject_id}. Skipping.")
        return

    # 5. Save the label map
    ensure_dir(subject_processed_dir(output_dir, subject_id))
    out_file = mask_path(output_dir, subject_id)
    save_nifti(label_data, out_file, affine=t1_img.affine, dtype=np.uint8)

    # 6. Print the label map statistics
    uniq, counts = np.unique(label_data, return_counts=True)
    print(
        f"  Saved: {out_file} shape={label_data.shape} "
        f"labels={dict(zip(uniq.astype(int).tolist(), counts.tolist()))}"
    )


def preprocess_masks(config):
    """
    Preprocess all masks for all subjects.
    
    Steps:
    1. Resolve the raw and output directories
    2. Load the structures
    3. Check if the raw directory exists
    4. Get the list of subjects
    5. Print the number of subjects
    6. Process each subject
    """

    # 1. Resolve the raw and output directories
    raw_dir = resolve_path(config, "data.paths.raw")
    output_dir = resolve_path(config, "data.paths.processed")
    ensure_dir(output_dir)

    # 2. Load the structures
    structures = load_structures(config)

    # 3. Check if the raw directory exists
    if not raw_dir.exists():
        print(f"Error: raw data directory not found: {raw_dir}")
        sys.exit(1)

    # 4. Get the list of subjects
    subjects = sorted(
        d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))
    )

    # 5. Process each subject
    print(f"Processing masks for {len(subjects)} subjects (full T1 resolution)")
    for subject_id in subjects:
            process_mask(
                subject_id, str(raw_dir), output_dir, config, structures
            )


if __name__ == "__main__":
    args = parse_config_args(description="Preprocess full-resolution segmentation masks")
    cfg = load_config(args.config)
    preprocess_masks(cfg)
