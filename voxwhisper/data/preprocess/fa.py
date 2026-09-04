"""Compute DTI Fractional Anisotropy (FA) maps from HCP diffusion data.

Pipeline position
-----------------
This module runs *before* ``volumes.py``.  It converts the heavy 4D diffusion
series (``Diffusion/data.nii.gz``) into a lightweight 3D FA map
(``dti_FA.nii.gz``) at the subject root, which ``volumes.py`` then normalises
and copies into the processed directory as the secondary modality.

Local layout
------------
    data/raw/{subject_id}/
        T1w_acpc_dc_restore_1.25.nii.gz
        tract_masks_1.25/
        Diffusion/
            data.nii.gz              ← 4D DWI series
            bvals
            bvecs
            nodif_brain_mask.nii.gz

Output
------
    data/raw/{subject_id}/dti_FA.nii.gz
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst import dti
from tqdm import tqdm

from voxwhisper.config import ensure_dir, resolve_path

logger = logging.getLogger(__name__)

# Name of the output file written to each subject's raw root directory.
FA_OUTPUT_FILENAME = "dti_FA.nii.gz"

_print_lock = Lock()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _diffusion_dir(raw_dir: Path, subject_id: str) -> Path:
    """Return the local diffusion directory for a subject."""
    return raw_dir / subject_id / "Diffusion"


def _required_diffusion_files(diff_dir: Path) -> dict[str, Path]:
    return {
        "data"  : diff_dir / "data.nii.gz",
        "bvals" : diff_dir / "bvals",
        "bvecs" : diff_dir / "bvecs",
        "mask"  : diff_dir / "nodif_brain_mask.nii.gz",
    }


# ---------------------------------------------------------------------------
# Per-subject DTI fitting
# ---------------------------------------------------------------------------

def compute_fa_subject(
    raw_dir: Path,
    subject_id: str,
    delete_raw: bool = False,
) -> Tuple[str, bool, str]:
    """Fit DTI and write an FA map for one subject.

    Parameters
    ----------
    raw_dir    : root raw-data directory (``data/raw``).
    subject_id : subject ID string.
    delete_raw : if ``True``, remove ``data.nii.gz`` after a successful write.

    Returns
    -------
    (subject_id, success, message)
    """
    diff_dir = _diffusion_dir(raw_dir, subject_id)
    files = _required_diffusion_files(diff_dir)
    output_path = raw_dir / subject_id / FA_OUTPUT_FILENAME

    if output_path.exists():
        return subject_id, True, "already exists, skipped"

    missing = [name for name, p in files.items() if not p.exists()]
    if missing:
        return subject_id, False, f"missing diffusion files: {', '.join(missing)}"

    try:
        dwi_img = nib.load(str(files["data"]))
        dwi_data = dwi_img.get_fdata(dtype=np.float32)

        mask_img = nib.load(str(files["mask"]))
        mask = mask_img.get_fdata().astype(np.uint8)

        bvals = np.loadtxt(str(files["bvals"]))
        bvecs = np.loadtxt(str(files["bvecs"]))
        gtab = gradient_table(bvals, bvecs=bvecs)

        tensor_model = dti.TensorModel(gtab)
        tensor_fit = tensor_model.fit(dwi_data, mask=mask)

        fa_map = tensor_fit.fa.astype(np.float32)
        fa_map = np.nan_to_num(fa_map, nan=0.0, posinf=0.0, neginf=0.0)
        fa_map = np.clip(fa_map, 0.0, 1.0)

        fa_img = nib.Nifti1Image(fa_map, affine=dwi_img.affine, header=dwi_img.header)
        fa_img.header.set_data_dtype(np.float32)
        ensure_dir(output_path.parent)
        nib.save(fa_img, str(output_path))

        if delete_raw and output_path.exists():
            files["data"].unlink()
            return subject_id, True, f"saved → {output_path}  (raw 4D deleted)"

        return subject_id, True, f"saved → {output_path}"

    except Exception as exc:  # noqa: BLE001
        return subject_id, False, f"error: {exc}"


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def _find_subjects(raw_dir: Path, subject_filter: Optional[str] = None) -> List[str]:
    """Return sorted subject IDs present in ``raw_dir``."""
    if not raw_dir.exists():
        return []
    subjects = sorted(
        d for d in os.listdir(raw_dir)
        if (raw_dir / d).is_dir() and d.isdigit() and len(d) == 6
    )
    if subject_filter is not None:
        if subject_filter not in subjects:
            raise ValueError(
                f"Subject '{subject_filter}' not found under {raw_dir}. "
                f"Available: {subjects[:10]}{'...' if len(subjects) > 10 else ''}"
            )
        return [subject_filter]
    return subjects


# ---------------------------------------------------------------------------
# Cohort-level orchestrator
# ---------------------------------------------------------------------------

def compute_fa_cohort(
    config: dict,
    subject_filter: Optional[str] = None,
    delete_raw: bool = False,
    max_workers: int = 4,
) -> None:
    """Compute FA maps for all (or one) subjects in the raw directory.

    Parameters
    ----------
    config        : loaded YAML config.
    subject_filter: process only this subject ID when provided.
    delete_raw    : remove ``data.nii.gz`` after each successful FA write.
    max_workers   : parallel worker threads.
    """
    raw_dir = resolve_path(config, "data.paths.raw")

    subjects = _find_subjects(raw_dir, subject_filter)
    if not subjects:
        logger.warning("No subjects found in %s — nothing to do.", raw_dir)
        return

    logger.info("Computing FA for %d subject(s)  |  workers=%d  |  delete_raw=%s",
                len(subjects), max_workers, delete_raw)

    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(compute_fa_subject, raw_dir, sid, delete_raw): sid
            for sid in subjects
        }
        with tqdm(total=len(subjects), desc="FA maps", unit="sub") as pbar:
            for future in as_completed(futures):
                sid, ok, msg = future.result()
                with _print_lock:
                    status = "OK" if ok else "FAIL"
                    tqdm.write(f"  [{status}] {sid}: {msg}")
                if not ok:
                    failed.append(sid)
                pbar.update(1)
                pbar.set_postfix(failed=len(failed))

    if failed:
        logger.warning("%d subject(s) failed: %s", len(failed), failed)
    else:
        logger.info("All FA maps completed successfully.")
