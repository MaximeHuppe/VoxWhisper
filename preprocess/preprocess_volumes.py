# preprocess/preprocess_volumes.py
"""Config-driven T1/T2 preprocessing: full-resolution normalized NIfTI."""
from __future__ import annotations

import os
import sys

import nibabel as nib
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
    normalize_intensity,
    save_nifti,
    subject_processed_dir,
    volume_path,
)


def resolve_volume_path(raw_dir, subject_id, volume_cfg):
    """Resolve configured volume filename under the subject raw directory."""
    primary = os.path.join(raw_dir, subject_id, "T1w", volume_cfg["filename"])
    if os.path.exists(primary):
        return primary
    return None


def load_and_process_volume(path, normalization, nonzero_only):
    """
    Load NIfTI and normalize at full resolution.

    Spatial cropping is deferred to the Dataset (50/50 patch sampling).
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    normalized = normalize_intensity(
        data, method=normalization, nonzero_only=nonzero_only
    )
    return normalized, img.affine


def process_subject(subject_id, raw_dir, output_dir, config):
    print(f"Processing Subject: {subject_id}")

    primary, secondary = active_modality_keys(config)
    volumes_cfg = config["data"]["volumes"]
    prep_cfg = config["preprocessing"]
    normalization = prep_cfg["normalization"]
    nonzero_only = bool(prep_cfg.get("zscore_nonzero_only", True))

    subject_dir = subject_processed_dir(output_dir, subject_id)
    ensure_dir(subject_dir)

    for modality in (primary, secondary):
        if modality not in volumes_cfg:
            print(f"Warning: modality '{modality}' missing from data.volumes. Skipping subject.")
            return

        vol_cfg = volumes_cfg[modality]
        path = resolve_volume_path(raw_dir, subject_id, vol_cfg)
        if path is None:
            print(
                f"Warning: {modality.upper()} file missing for {subject_id} "
                f"(looked for {vol_cfg['filename']}). Skipping."
            )
            return

        volume, affine = load_and_process_volume(
            path,
            normalization=normalization,
            nonzero_only=nonzero_only,
        )
        out_file = volume_path(output_dir, subject_id, modality)
        save_nifti(volume, out_file, affine=affine, dtype=np.float32)
        print(
            f"  Saved {modality}: {out_file} "
            f"shape={volume.shape} range=[{volume.min():.3f}, {volume.max():.3f}]"
        )


def preprocess_volumes(config):
    raw_data_dir = resolve_path(config, "data.paths.raw")
    processed_data_dir = resolve_path(config, "data.paths.processed")
    ensure_dir(processed_data_dir)

    if not raw_data_dir.exists():
        print(f"Error: raw data directory not found: {raw_data_dir}")
        sys.exit(1)

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
    for subject in tqdm(subjects, desc="Processing subjects"):
        process_subject(subject, str(raw_data_dir), processed_data_dir, config)


if __name__ == "__main__":
    args = parse_config_args(description="Preprocess full-resolution T1/T2 volumes")
    cfg = load_config(args.config)
    preprocess_volumes(cfg)
