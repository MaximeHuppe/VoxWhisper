# preprocess/preprocess_volumes.py
"""Config-driven T1/T2 preprocessing: full-resolution normalized NIfTI."""
from __future__ import annotations

import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import (  # noqa: E402
    active_modality_keys,
    ensure_dir,
    load_config,
    parse_config_args,
    resolve_path,
)
from src.utils.nifti_io import (  # noqa: E402
    load_nifti,
    normalize_intensity,
    resolve_raw_volume_path,
    save_nifti,
    subject_processed_dir,
    volume_path,
)



########################################################
#                VOLUME PROCESSING FUNCTIONS          #
########################################################

def load_and_process_volume(path, normalization, nonzero_only):
    """Load NIfTI and z-score/minmax normalize at full resolution."""
    data, affine = load_nifti(path)
    normalized = normalize_intensity(
        data, method=normalization, nonzero_only=nonzero_only
    )
    return normalized, affine


def process_subject(subject_id, raw_dir, output_dir, config):
    """
    Process a single subject's volumes.
    
    Steps:
    1. Print the subject ID
    2. Get the primary and secondary modalities
    3. Get the volume configuration
    4. Get the normalization configuration
    5. Get the subject directory
    6. Ensure the subject directory exists
    7. Process each modality
    """

    # 1. Print the subject ID
    print(f"Processing Subject: {subject_id}")

    # 2. Get the primary and secondary modalities
    primary, secondary = active_modality_keys(config)

    # 3. Get the volume configuration and preprocessing configuration
    volumes_cfg = config["data"]["volumes"]
    prep_cfg = config["preprocessing"]
    normalization = prep_cfg["normalization"]
    nonzero_only = bool(prep_cfg.get("zscore_nonzero_only", True))

    subject_dir = subject_processed_dir(output_dir, subject_id)
    ensure_dir(subject_dir)

    # 4. Process each modality
    for modality in (primary, secondary):
        if modality not in volumes_cfg:
            print(f"Warning: modality '{modality}' missing from data.volumes. Skipping subject.")
            return

        # 4.1. Get the volume configuration
        vol_cfg = volumes_cfg[modality]
        path = resolve_raw_volume_path(raw_dir, subject_id, vol_cfg["filename"])
        if path is None:
            print(
                f"Warning: {modality.upper()} file missing for {subject_id} "
                f"(looked for {vol_cfg['filename']}). Skipping."
            )
            return

        # 4.2. Load and process the volume
        volume, affine = load_and_process_volume(
            path, normalization=normalization, nonzero_only=nonzero_only
        )

        # 4.3. Save the volume
        out_file = volume_path(output_dir, subject_id, modality)
        save_nifti(volume, out_file, affine=affine, dtype=np.float32)
        print(
            f"  Saved {modality}: {out_file} "
            f"shape={volume.shape} range=[{volume.min():.3f}, {volume.max():.3f}]"
        )


def preprocess_volumes(config):
    """
    Preprocess all volumes for all subjects.
    
    Steps:
    1. Resolve the raw and output directories
    2. Check if the raw directory exists
    3. Get the list of subjects
    4. Print the number of subjects
    5. Print the normalization configuration
    6. Process each subject
    """

    # 1. Resolve the raw and output directories
    raw_data_dir = resolve_path(config, "data.paths.raw")
    processed_data_dir = resolve_path(config, "data.paths.processed")

    # 2. Check if the raw directory exists
    ensure_dir(processed_data_dir)

    if not raw_data_dir.exists():
        print(f"Error: raw data directory not found: {raw_data_dir}")
        sys.exit(1)

    # 3. Get the list of subjects
    subjects = sorted(
        d
        for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
    )

    print(f"Found {len(subjects)} subjects in {raw_data_dir}")
    print(
        f"Normalization: {config['preprocessing']['normalization']} "
        f"(nonzero_only={config['preprocessing'].get('zscore_nonzero_only', True)}); "
        "keeping full resolution (patches sampled at train time)"
    )

    # 4. Process each subject
    for subject in tqdm(subjects, desc="Processing subjects"):
        process_subject(subject, str(raw_data_dir), processed_data_dir, config)


if __name__ == "__main__":
    args = parse_config_args(description="Preprocess full-resolution T1/T2 volumes")
    cfg = load_config(args.config)
    preprocess_volumes(cfg)
