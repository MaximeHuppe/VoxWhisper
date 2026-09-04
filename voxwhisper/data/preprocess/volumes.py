"""Config-driven T1/FA preprocessing: full-resolution z-score normalised NIfTI."""
from __future__ import annotations

import os
import sys

import numpy as np
from tqdm import tqdm

from voxwhisper.config import (
    PRIMARY_MODALITY,
    SECONDARY_MODALITY,
    ensure_dir,
    resolve_path,
)
from voxwhisper.data.nifti_io import (
    load_nifti,
    normalize_intensity,
    resolve_raw_volume_path,
    save_nifti,
    subject_processed_dir,
    volume_path,
)


def load_and_process_volume(path, nonzero_only: bool = True):
    """Load NIfTI and z-score normalize at full resolution."""
    data, affine = load_nifti(path)
    normalized = normalize_intensity(data, nonzero_only=nonzero_only)
    return normalized, affine


def process_subject(subject_id: str, raw_dir: str, output_dir, config: dict) -> None:
    """Process a single subject's T1 and FA volumes."""
    print(f"Processing Subject: {subject_id}")

    volumes_cfg = config["data"]["volumes"]
    prep_cfg = config["preprocessing"]
    nonzero_only = bool(prep_cfg.get("zscore_nonzero_only", True))

    subject_dir = subject_processed_dir(output_dir, subject_id)
    ensure_dir(subject_dir)

    for modality in (PRIMARY_MODALITY, SECONDARY_MODALITY):
        if modality not in volumes_cfg:
            print(f"Warning: modality '{modality}' missing from data.volumes. Skipping subject.")
            return

        vol_cfg = volumes_cfg[modality]
        path = resolve_raw_volume_path(raw_dir, subject_id, vol_cfg["filename"])
        if path is None:
            print(
                f"Warning: {modality.upper()} file missing for {subject_id} "
                f"(looked for {vol_cfg['filename']}). Skipping."
            )
            return

        volume, affine = load_and_process_volume(path, nonzero_only=nonzero_only)
        out_file = volume_path(output_dir, subject_id, modality)
        save_nifti(volume, out_file, affine=affine, dtype=np.float32)
        print(
            f"  Saved {modality}: {out_file} "
            f"shape={volume.shape} range=[{volume.min():.3f}, {volume.max():.3f}]"
        )


def preprocess_volumes(config: dict) -> None:
    """Preprocess all T1 and FA volumes for all subjects."""
    raw_data_dir = resolve_path(config, "data.paths.raw")
    processed_data_dir = resolve_path(config, "data.paths.processed")
    ensure_dir(processed_data_dir)

    if not raw_data_dir.exists():
        print(f"Error: raw data directory not found: {raw_data_dir}")
        sys.exit(1)

    subjects = sorted(
        d for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
    )

    nonzero_only = bool(config["preprocessing"].get("zscore_nonzero_only", True))
    print(f"Found {len(subjects)} subjects in {raw_data_dir}")
    print(f"Normalization: z-score (nonzero_only={nonzero_only}); keeping full resolution")

    for subject in tqdm(subjects, desc="Processing subjects"):
        process_subject(subject, str(raw_data_dir), processed_data_dir, config)
