"""Config-driven volume preprocessing: z-score NIfTI on the native T1 grid.

Phase 1 writes T1 only (optional FreeSurfer brainmask). Phase 2 also writes FA
when ``data.volumes.fa`` is set. FA is never brain-masked with the T1 mask.
"""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from tqdm import tqdm

from voxwhisper.util.config import PRIMARY_MODALITY, SECONDARY_MODALITY, ensure_dir, resolve_path
from voxwhisper.data.nifti_io import (
    load_nifti,
    normalize_intensity,
    resolve_raw_volume_path,
    save_nifti,
    subject_processed_dir,
    volume_path,
)
from voxwhisper.util.stage import uses_secondary


def _resampled_brainmask(
    raw_dir: str,
    subject_id: str,
    config: dict,
    t1_img: nib.Nifti1Image,
) -> np.ndarray | None:
    brain_cfg = config.get("data", {}).get("volumes", {}).get("brainmask")
    if not brain_cfg:
        return None
    path = resolve_raw_volume_path(raw_dir, subject_id, brain_cfg["filename"])
    if path is None:
        print(f"  Warning: brainmask missing for {subject_id}")
        return None
    resampled = resample_from_to(nib.load(str(path)), t1_img, order=0)
    return resampled.get_fdata() > 0


def _write_volume(volume, affine, output_dir, subject_id: str, modality: str) -> None:
    ensure_dir(subject_processed_dir(output_dir, subject_id))
    out_file = volume_path(output_dir, subject_id, modality)
    save_nifti(volume, out_file, affine=affine, dtype=np.float32)
    # print(
    #     f"  Saved {modality}: {out_file} "
    #     f"shape={volume.shape} range=[{volume.min():.3f}, {volume.max():.3f}]"
    # )


def process_subject(subject_id: str, raw_dir: str, output_dir, config: dict) -> None:
    """Process T1, and FA when the stage config includes it."""
    # print(f"Processing Subject: {subject_id}")

    volumes_cfg = config["data"]["volumes"]
    prep_cfg = config["preprocessing"]
    nonzero_only = bool(prep_cfg.get("zscore_nonzero_only", True))
    apply_brainmask = bool(prep_cfg.get("apply_brainmask", True))

    t1_cfg = volumes_cfg[PRIMARY_MODALITY]
    path = resolve_raw_volume_path(raw_dir, subject_id, t1_cfg["filename"])
    if path is None:
        print(
            f"Warning: T1 file missing for {subject_id} "
            f"(looked for {t1_cfg['filename']}). Skipping."
        )
        return

    t1_img = nib.load(str(path))
    volume, affine = load_nifti(path)

    if apply_brainmask:
        mask = _resampled_brainmask(raw_dir, subject_id, config, t1_img)
        if mask is not None:
            if mask.shape != volume.shape:
                print(
                    f"  Warning: brainmask shape {mask.shape} != T1 {volume.shape} "
                    f"for {subject_id}; skipping mask apply"
                )
            else:
                volume = volume.copy()
                volume[~mask] = 0.0

    normalized = normalize_intensity(volume, nonzero_only=nonzero_only)
    _write_volume(normalized, affine, output_dir, subject_id, PRIMARY_MODALITY)

    if not uses_secondary(config) or SECONDARY_MODALITY not in volumes_cfg:
        return

    fa_cfg = volumes_cfg[SECONDARY_MODALITY]
    fa_path = resolve_raw_volume_path(raw_dir, subject_id, fa_cfg["filename"])
    if fa_path is None:
        print(
            f"Warning: FA file missing for {subject_id} "
            f"(looked for {fa_cfg['filename']}). Skipping FA."
        )
        return
    fa_vol, fa_affine = load_nifti(fa_path)
    fa_norm = normalize_intensity(fa_vol, nonzero_only=nonzero_only)
    _write_volume(fa_norm, fa_affine, output_dir, subject_id, SECONDARY_MODALITY)


def preprocess_volumes(config: dict, subject_ids=None) -> None:
    """Preprocess volumes for all (or a filtered list of) subjects."""
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
    if subject_ids is not None:
        allowed = set(subject_ids)
        subjects = [s for s in subjects if s in allowed]

    nonzero_only = bool(config["preprocessing"].get("zscore_nonzero_only", True))
    mods = "T1+FA" if uses_secondary(config) else "T1 only"
    print(f"Found {len(subjects)} subjects in {raw_data_dir}")
    print(f"Normalization: z-score (nonzero_only={nonzero_only}); {mods}")

    for subject in tqdm(subjects, desc="Processing subjects"):
        process_subject(subject, str(raw_data_dir), processed_data_dir, config)
