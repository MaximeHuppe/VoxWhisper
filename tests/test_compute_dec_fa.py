"""Tests for preprocess/compute_dec_fa.py.

Synthetic diffusion data (tiny volumes) so tests run on CPU in seconds.

Tested behaviour
----------------
- ``compute_dec_fa_subject`` writes a 4D RGB ``dti_DEC_FA.nii.gz``.
- RGB values are finite and clamped to ``[0, 1]``.
- Isotropic signal yields near-black (low FA) colour.
- A stick tensor along +x is encoded as red-dominant RGB.
- Scalar FA is written as a by-product when it is missing.
- Existing DEC-FA is skipped; ``--delete-raw`` removes the 4D DWI.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from preprocess.compute_fa import FA_OUTPUT_FILENAME
from preprocess.compute_dec_fa import (
    DEC_FA_OUTPUT_FILENAME,
    compute_dec_fa_subject,
)
from tests.test_compute_fa import _make_synthetic_diffusion


def _make_anisotropic_x_diffusion(diff_dir: Path, shape=(8, 8, 8)) -> Path:
    """Write DWI with a stick tensor along +x (high FA, red-dominant DEC-FA)."""
    diff_dir.mkdir(parents=True, exist_ok=True)

    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
    bvecs = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    np.savetxt(diff_dir / "bvals", bvals[np.newaxis, :], fmt="%.1f")
    np.savetxt(diff_dir / "bvecs", bvecs.T, fmt="%.6f")

    mask = np.ones(shape, dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(diff_dir / "nodif_brain_mask.nii.gz"))

    # Stick tensor: λ_x ≫ λ_y = λ_z  →  FA ≈ 1, e1 ≈ (1, 0, 0)
    evals = np.array([1.7e-3, 0.2e-3, 0.2e-3])
    adc = (bvecs**2) @ evals  # (7,)
    S0 = 1000.0
    signal = S0 * np.exp(-bvals * adc)
    dwi = np.broadcast_to(signal, (*shape, len(bvals))).copy().astype(np.float32)
    nib.save(nib.Nifti1Image(dwi, np.eye(4)), str(diff_dir / "data.nii.gz"))
    return diff_dir


@pytest.fixture()
def isotropic_raw_dir(tmp_path: Path):
    subject_id = "000001"
    _make_synthetic_diffusion(tmp_path / subject_id / "Diffusion")
    return tmp_path, subject_id


@pytest.fixture()
def anisotropic_raw_dir(tmp_path: Path):
    subject_id = "000002"
    _make_anisotropic_x_diffusion(tmp_path / subject_id / "Diffusion")
    return tmp_path, subject_id


class TestComputeDecFaSubject:
    def test_success_writes_rgb_file(self, isotropic_raw_dir):
        raw_dir, subject_id = isotropic_raw_dir
        sid, ok, msg = compute_dec_fa_subject(raw_dir, subject_id)
        assert sid == subject_id
        assert ok, f"Expected success, got: {msg}"
        output = raw_dir / subject_id / DEC_FA_OUTPUT_FILENAME
        assert output.exists(), "DEC-FA NIfTI was not written"
        rgb = nib.load(str(output)).get_fdata()
        assert rgb.ndim == 4 and rgb.shape[-1] == 3, f"Expected (D,H,W,3), got {rgb.shape}"

    def test_rgb_values_are_finite_and_bounded(self, isotropic_raw_dir):
        raw_dir, subject_id = isotropic_raw_dir
        compute_dec_fa_subject(raw_dir, subject_id)
        rgb = nib.load(str(raw_dir / subject_id / DEC_FA_OUTPUT_FILENAME)).get_fdata()
        assert np.all(np.isfinite(rgb))
        assert float(rgb.min()) >= 0.0
        assert float(rgb.max()) <= 1.0

    def test_isotropic_signal_is_near_black(self, isotropic_raw_dir):
        raw_dir, subject_id = isotropic_raw_dir
        compute_dec_fa_subject(raw_dir, subject_id)
        rgb = nib.load(str(raw_dir / subject_id / DEC_FA_OUTPUT_FILENAME)).get_fdata()
        assert float(rgb.mean()) < 0.2, f"Expected dark DEC-FA for isotropic signal, got {rgb.mean():.3f}"

    def test_x_stick_tensor_is_red_dominant(self, anisotropic_raw_dir):
        raw_dir, subject_id = anisotropic_raw_dir
        compute_dec_fa_subject(raw_dir, subject_id)
        rgb = nib.load(str(raw_dir / subject_id / DEC_FA_OUTPUT_FILENAME)).get_fdata()
        r, g, b = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
        assert r > 0.5, f"Expected high red for +x fibres, got R={r:.3f}"
        assert r > 2 * g and r > 2 * b, f"Expected R≫G,B, got R={r:.3f} G={g:.3f} B={b:.3f}"

    def test_writes_scalar_fa_when_missing(self, isotropic_raw_dir):
        raw_dir, subject_id = isotropic_raw_dir
        fa_path = raw_dir / subject_id / FA_OUTPUT_FILENAME
        assert not fa_path.exists()
        _, ok, msg = compute_dec_fa_subject(raw_dir, subject_id)
        assert ok
        assert fa_path.exists()
        assert "dti_FA" in msg

    def test_does_not_overwrite_existing_fa(self, isotropic_raw_dir):
        raw_dir, subject_id = isotropic_raw_dir
        fa_path = raw_dir / subject_id / FA_OUTPUT_FILENAME
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), str(fa_path))
        mtime = fa_path.stat().st_mtime
        compute_dec_fa_subject(raw_dir, subject_id)
        assert fa_path.stat().st_mtime == mtime

    def test_skips_when_dec_fa_already_exists(self, isotropic_raw_dir):
        raw_dir, subject_id = isotropic_raw_dir
        compute_dec_fa_subject(raw_dir, subject_id)
        mtime_1 = (raw_dir / subject_id / DEC_FA_OUTPUT_FILENAME).stat().st_mtime
        _, ok, msg = compute_dec_fa_subject(raw_dir, subject_id)
        mtime_2 = (raw_dir / subject_id / DEC_FA_OUTPUT_FILENAME).stat().st_mtime
        assert ok
        assert "skipped" in msg
        assert mtime_1 == mtime_2

    def test_fails_gracefully_on_missing_files(self, tmp_path):
        (tmp_path / "999999").mkdir()
        sid, ok, msg = compute_dec_fa_subject(tmp_path, "999999")
        assert sid == "999999"
        assert not ok
        assert "missing" in msg.lower()

    def test_delete_raw_removes_data_nii(self, isotropic_raw_dir):
        raw_dir, subject_id = isotropic_raw_dir
        data_nii = raw_dir / subject_id / "Diffusion" / "data.nii.gz"
        _, ok, _ = compute_dec_fa_subject(raw_dir, subject_id, delete_raw=True)
        assert ok
        assert not data_nii.exists()

    def test_affine_preserved(self, isotropic_raw_dir):
        raw_dir, subject_id = isotropic_raw_dir
        input_affine = nib.load(
            str(raw_dir / subject_id / "Diffusion" / "data.nii.gz")
        ).affine
        compute_dec_fa_subject(raw_dir, subject_id)
        output_affine = nib.load(
            str(raw_dir / subject_id / DEC_FA_OUTPUT_FILENAME)
        ).affine
        np.testing.assert_array_almost_equal(input_affine, output_affine)
