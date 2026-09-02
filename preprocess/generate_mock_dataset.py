"""Generate a synthetic processed dataset for unit and integration tests.

This module creates small NIfTI volumes that match the processed-data layout
expected by ``VoxWhisperDataset``:

    {processed_root}/{subject_id}/
        {primary}.nii.gz   — z-score normalised MRI-like volume
        {secondary}.nii.gz — secondary modality
        mask.nii.gz        — integer label map in [0, n_structures)

Usage
-----
Directly (e.g. from tests):

    from preprocess.generate_mock_dataset import make_mock_cohort
    make_mock_cohort(config, num_subjects=4)

The generated IDs are zero-padded integers: ``"000000"``, ``"000001"``, etc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np


def _write_nifti(
    data: np.ndarray,
    path: Path,
    affine: Optional[np.ndarray] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4, dtype=np.float64)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def make_mock_subject(
    subject_dir: Path,
    volume_shape: Tuple[int, int, int],
    modalities: Sequence[str],
    n_structures: int,
    rng: np.random.Generator,
) -> None:
    """Write synthetic NIfTI files for a single subject.

    Parameters
    ----------
    subject_dir   : output directory (created if absent).
    volume_shape  : (D, H, W) spatial size.
    modalities    : list of modality names to create (e.g. ["t1", "b0"]).
    n_structures  : total number of label classes including background.
    rng           : random number generator for reproducibility.
    """
    affine = np.eye(4, dtype=np.float64)

    for modality in modalities:
        vol = rng.standard_normal(volume_shape).astype(np.float32)
        _write_nifti(vol, subject_dir / f"{modality}.nii.gz", affine)

    # Integer mask: background (0) everywhere, with a small foreground blob
    # for each structure label so patch sampling finds foreground voxels.
    mask = np.zeros(volume_shape, dtype=np.int16)
    D, H, W = volume_shape
    blob_r = max(2, min(D, H, W) // 8)
    for label in range(1, n_structures):
        z = int(rng.integers(blob_r, D - blob_r))
        y = int(rng.integers(blob_r, H - blob_r))
        x = int(rng.integers(blob_r, W - blob_r))
        zs = slice(max(0, z - blob_r), min(D, z + blob_r))
        ys = slice(max(0, y - blob_r), min(H, y + blob_r))
        xs = slice(max(0, x - blob_r), min(W, x + blob_r))
        mask[zs, ys, xs] = label

    _write_nifti(mask.astype(np.float32), subject_dir / "mask.nii.gz", affine)


def make_mock_cohort(
    config: Mapping,
    num_subjects: int = 4,
    seed: int = 0,
) -> None:
    """Generate a full mock processed cohort under ``data.paths.processed``.

    Parameters
    ----------
    config       : loaded YAML config (must include ``data.paths.processed``,
                   ``data.modalities``, ``data.mock_volume_shape``, and
                   ``data.masks.structures`` or ``data.prompts``).
    num_subjects : how many synthetic subjects to create.
    seed         : RNG seed for reproducibility.
    """
    from src.utils.config import active_modality_keys, get_project_root

    processed_root = Path(config["data"]["paths"]["processed"])
    volume_shape = tuple(int(x) for x in config["data"].get("mock_volume_shape", [64, 64, 64]))
    primary_key, secondary_key = active_modality_keys(config)
    modalities = [primary_key, secondary_key]

    # Derive n_structures from the prompts list (includes background channel 0).
    n_structures = len(config["data"].get("prompts", ["background"]))

    rng = np.random.default_rng(seed)
    for i in range(num_subjects):
        subject_id = f"{i:06d}"
        subject_dir = processed_root / subject_id
        make_mock_subject(
            subject_dir=subject_dir,
            volume_shape=volume_shape,  # type: ignore[arg-type]
            modalities=modalities,
            n_structures=n_structures,
            rng=rng,
        )
