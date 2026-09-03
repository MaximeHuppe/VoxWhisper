"""Compute DTI Directionally Encoded Color FA (DEC-FA) from HCP diffusion data.

Pipeline position
-----------------
This script runs *before* ``preprocess_volumes.py``, same slot as
``compute_fa.py``.  It converts the 4D diffusion series
(``Diffusion/data.nii.gz``) into a 4D RGB volume
(``dti_DEC_FA.nii.gz``) at the subject root.

DEC-FA (Pajevic & Pierpaoli, 1999)
----------------------------------
Each voxel is an RGB triplet encoding the principal diffusion direction,
modulated by fractional anisotropy:

    R = |e1_x| * FA    (left–right)
    G = |e1_y| * FA    (anterior–posterior)
    B = |e1_z| * FA    (superior–inferior)

Values lie in ``[0, 1]``.  Do **not** z-score this volume — that would
destroy the directional colour encoding.  Set
``data.volumes.dec_fa.normalize: false`` (already the default in
``config/tracts.yaml``).

Local layout (after ``extract_hcp.py``)
---------------------------------------
    data/raw/{subject_id}/
        T1w_acpc_dc_restore_1.25.nii.gz
        Diffusion/
            data.nii.gz
            bvals
            bvecs
            nodif_brain_mask.nii.gz

Output
------
    data/raw/{subject_id}/dti_DEC_FA.nii.gz   ← 4D ``(D, H, W, 3)``, RGB
    data/raw/{subject_id}/dti_FA.nii.gz       ← 3D scalar FA, written only
        if it does not already exist (same DTI fit, no extra cost).

Switching the model to DEC-FA
-----------------------------
In the experiment YAML:

    data.modalities.secondary: dec_fa
    data.paths.processed: data/processed_T1_DEC_FA
    model.secondary_encoder.input_channels: 3

Then:

    python preprocess/compute_dec_fa.py --config config/tracts.yaml
    python pipeline/run_preprocess.py --config <your_dec_fa_config.yaml>

Usage
-----
    python preprocess/compute_dec_fa.py --config config/tracts.yaml
    python preprocess/compute_dec_fa.py --config config/tracts.yaml --delete-raw
    python preprocess/compute_dec_fa.py --config config/tracts.yaml --subject 599469
    python preprocess/compute_dec_fa.py --config config/tracts.yaml --workers 4
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst import dti
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess.compute_fa import (  # noqa: E402
    FA_OUTPUT_FILENAME,
    _diffusion_dir,
    _find_subjects,
    _required_diffusion_files,
)
from src.utils.config import ensure_dir, load_config, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)

# 4D RGB volume written to each subject's raw root directory.
DEC_FA_OUTPUT_FILENAME = "dti_DEC_FA.nii.gz"

_print_lock = Lock()


def _sanitise_fa(fa_map: np.ndarray) -> np.ndarray:
    fa_map = fa_map.astype(np.float32, copy=False)
    fa_map = np.nan_to_num(fa_map, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(fa_map, 0.0, 1.0)


def _save_nifti(data: np.ndarray, affine, header, path: Path) -> None:
    img = nib.Nifti1Image(data, affine=affine, header=header)
    img.header.set_data_dtype(np.float32)
    ensure_dir(path.parent)
    nib.save(img, str(path))


def compute_dec_fa_subject(
    raw_dir: Path,
    subject_id: str,
    delete_raw: bool = False,
) -> Tuple[str, bool, str]:
    """Fit DTI and write a DEC-FA RGB volume for one subject.

    Parameters
    ----------
    raw_dir    : root raw-data directory (``data/raw``).
    subject_id : 6-digit HCP subject ID string.
    delete_raw : if ``True``, remove ``data.nii.gz`` after a successful write.

    Returns
    -------
    (subject_id, success, message)
    """
    diff_dir = _diffusion_dir(raw_dir, subject_id)
    files = _required_diffusion_files(diff_dir)
    output_path = raw_dir / subject_id / DEC_FA_OUTPUT_FILENAME
    fa_path = raw_dir / subject_id / FA_OUTPUT_FILENAME

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

        fa_map = _sanitise_fa(tensor_fit.fa)
        rgb = dti.color_fa(fa_map, tensor_fit.evecs).astype(np.float32)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
        rgb = np.clip(rgb, 0.0, 1.0)

        _save_nifti(rgb, dwi_img.affine, dwi_img.header, output_path)

        extras: list[str] = []
        if not fa_path.exists():
            _save_nifti(fa_map, dwi_img.affine, dwi_img.header, fa_path)
            extras.append("also wrote dti_FA.nii.gz")

        if delete_raw and output_path.exists():
            files["data"].unlink()
            extras.append("raw 4D deleted")

        suffix = f"  ({'; '.join(extras)})" if extras else ""
        return subject_id, True, f"saved → {output_path}{suffix}"

    except Exception as exc:  # noqa: BLE001
        return subject_id, False, f"error: {exc}"


def compute_dec_fa_cohort(
    config: dict,
    subject_filter: Optional[str] = None,
    delete_raw: bool = False,
    max_workers: int = 4,
) -> None:
    """Compute DEC-FA maps for all (or one) subjects in the raw directory."""
    raw_dir = resolve_path(config, "data.paths.raw")

    subjects = _find_subjects(raw_dir, subject_filter)
    if not subjects:
        logger.warning("No subjects found in %s — nothing to do.", raw_dir)
        return

    logger.info(
        "Computing DEC-FA for %d subject(s)  |  workers=%d  |  delete_raw=%s",
        len(subjects),
        max_workers,
        delete_raw,
    )

    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(compute_dec_fa_subject, raw_dir, sid, delete_raw): sid
            for sid in subjects
        }
        with tqdm(total=len(subjects), desc="DEC-FA maps", unit="sub") as pbar:
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
        logger.info("All DEC-FA maps completed successfully.")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute DTI DEC-FA (RGB) maps from HCP diffusion data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/tracts.yaml",
        help="Path to YAML config (relative to project root or absolute)",
    )
    parser.add_argument(
        "--subject",
        default=None,
        metavar="ID",
        help="Process a single subject instead of the full cohort",
    )
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        default=False,
        help="Delete data.nii.gz after a successful DEC-FA write (saves ~1-2 GB/subject)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Number of parallel worker threads for DTI fitting",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = _parse_args()
    cfg = load_config(args.config)

    delete_raw = args.delete_raw or bool(
        cfg.get("data", {}).get("download", {}).get("delete_raw_4d", False)
    )

    compute_dec_fa_cohort(
        config=cfg,
        subject_filter=args.subject,
        delete_raw=delete_raw,
        max_workers=args.workers,
    )
