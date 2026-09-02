"""Tests for preprocess/compute_fa.py.

We use synthetic (tiny) diffusion data so the tests run on CPU in seconds
without the real HCP dataset.  A single "free-water" shell (b=1000) is
generated with isotropic tensors so the expected FA is 0.

Tested behaviour
----------------
- ``_diffusion_dir`` returns the correct path.
- ``_required_diffusion_files`` lists all four expected files.
- ``compute_fa_subject`` returns a success tuple and writes ``dti_FA.nii.gz``.
- ``compute_fa_subject`` skips gracefully when input files are missing.
- ``compute_fa_subject`` skips without re-computing when FA already exists.
- FA values are finite and clamped to ``[0, 1]``.
- ``--delete-raw`` removes ``data.nii.gz`` after a successful FA write.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from preprocess.compute_fa import (
    FA_OUTPUT_FILENAME,
    _diffusion_dir,
    _required_diffusion_files,
    compute_fa_subject,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_diffusion(diff_dir: Path, shape=(10, 10, 10)) -> Path:
    """Write minimal but valid diffusion files that dipy can fit.

    We create 7 directions (1 b=0 + 6 DWI) with an isotropic tensor,
    keeping the volume tiny so the test runs in under a second.
    """
    diff_dir.mkdir(parents=True, exist_ok=True)

    # --- gradient table: 1 b0 + 6 DWI shells along canonical axes ---
    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
    bvecs = np.array([
        [0.0,  0.0,  0.0],
        [1.0,  0.0,  0.0],
        [-1.0, 0.0,  0.0],
        [0.0,  1.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0,  0.0, -1.0],
    ], dtype=np.float64)
    np.savetxt(diff_dir / "bvals", bvals[np.newaxis, :], fmt="%.1f")
    np.savetxt(diff_dir / "bvecs", bvecs.T, fmt="%.6f")

    # --- brain mask: whole volume ---
    mask = np.ones(shape, dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(diff_dir / "nodif_brain_mask.nii.gz"))

    # --- 4D DWI series: Rician-noise-free isotropic signal ---
    # S = S0 * exp(-b * ADC) with ADC ~ 0.7e-3 mm²/s (isotropic water)
    adc = 0.7e-3
    S0 = 1000.0
    signal = S0 * np.exp(-bvals * adc)          # (7,)
    dwi = np.broadcast_to(signal, (*shape, len(bvals))).copy().astype(np.float32)
    nib.save(nib.Nifti1Image(dwi, np.eye(4)), str(diff_dir / "data.nii.gz"))

    return diff_dir


@pytest.fixture()
def subject_raw_dir(tmp_path: Path):
    """Set up a raw directory with one synthetic subject (ID '000001')."""
    subject_id = "000001"
    diff_dir = tmp_path / subject_id / "Diffusion"
    _make_synthetic_diffusion(diff_dir)
    return tmp_path, subject_id


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestPathHelpers:
    def test_diffusion_dir_structure(self, tmp_path):
        raw_dir = tmp_path / "raw"
        result = _diffusion_dir(raw_dir, "123456")
        assert result == raw_dir / "123456" / "Diffusion"

    def test_required_diffusion_files_keys(self, tmp_path):
        diff_dir = tmp_path / "Diffusion"
        files = _required_diffusion_files(diff_dir)
        assert set(files.keys()) == {"data", "bvals", "bvecs", "mask"}

    def test_required_diffusion_files_paths(self, tmp_path):
        diff_dir = tmp_path / "Diffusion"
        files = _required_diffusion_files(diff_dir)
        assert files["data"] == diff_dir / "data.nii.gz"
        assert files["bvals"] == diff_dir / "bvals"
        assert files["bvecs"] == diff_dir / "bvecs"
        assert files["mask"] == diff_dir / "nodif_brain_mask.nii.gz"


class TestComputeFaSubject:
    def test_success_writes_fa_file(self, subject_raw_dir):
        raw_dir, subject_id = subject_raw_dir
        sid, ok, msg = compute_fa_subject(raw_dir, subject_id)
        assert sid == subject_id
        assert ok, f"Expected success, got: {msg}"
        output = raw_dir / subject_id / FA_OUTPUT_FILENAME
        assert output.exists(), "FA NIfTI was not written"

    def test_fa_values_are_finite_and_bounded(self, subject_raw_dir):
        raw_dir, subject_id = subject_raw_dir
        compute_fa_subject(raw_dir, subject_id)
        fa = nib.load(str(raw_dir / subject_id / FA_OUTPUT_FILENAME)).get_fdata()
        assert np.all(np.isfinite(fa)), "FA map contains non-finite values"
        assert float(fa.min()) >= 0.0, "FA map contains negative values"
        assert float(fa.max()) <= 1.0, "FA map contains values > 1"

    def test_isotropic_signal_produces_low_fa(self, subject_raw_dir):
        """Isotropic tensors should yield FA close to 0 (not strictly 0 due
        to noise-free but still numerically imperfect fitting)."""
        raw_dir, subject_id = subject_raw_dir
        compute_fa_subject(raw_dir, subject_id)
        fa = nib.load(str(raw_dir / subject_id / FA_OUTPUT_FILENAME)).get_fdata()
        assert float(fa.mean()) < 0.2, f"Expected low FA for isotropic signal, got {fa.mean():.3f}"

    def test_skips_when_fa_already_exists(self, subject_raw_dir):
        raw_dir, subject_id = subject_raw_dir
        # First call writes the FA.
        compute_fa_subject(raw_dir, subject_id)
        mtime_1 = (raw_dir / subject_id / FA_OUTPUT_FILENAME).stat().st_mtime
        # Second call should skip.
        _, ok, msg = compute_fa_subject(raw_dir, subject_id)
        mtime_2 = (raw_dir / subject_id / FA_OUTPUT_FILENAME).stat().st_mtime
        assert ok
        assert "skipped" in msg
        assert mtime_1 == mtime_2, "FA file was overwritten on second call"

    def test_fails_gracefully_on_missing_files(self, tmp_path):
        """Subject directory exists but has no diffusion files."""
        (tmp_path / "999999").mkdir()
        sid, ok, msg = compute_fa_subject(tmp_path, "999999")
        assert sid == "999999"
        assert not ok
        assert "missing" in msg.lower()

    def test_delete_raw_removes_data_nii(self, subject_raw_dir):
        raw_dir, subject_id = subject_raw_dir
        data_nii = raw_dir / subject_id / "Diffusion" / "data.nii.gz"
        assert data_nii.exists()
        _, ok, _ = compute_fa_subject(raw_dir, subject_id, delete_raw=True)
        assert ok
        assert not data_nii.exists(), "data.nii.gz was not deleted with delete_raw=True"

    def test_affine_preserved_in_output(self, subject_raw_dir):
        """The output FA should carry the same affine as the input DWI."""
        raw_dir, subject_id = subject_raw_dir
        input_affine = nib.load(
            str(raw_dir / subject_id / "Diffusion" / "data.nii.gz")
        ).affine
        compute_fa_subject(raw_dir, subject_id)
        output_affine = nib.load(
            str(raw_dir / subject_id / FA_OUTPUT_FILENAME)
        ).affine
        np.testing.assert_array_almost_equal(input_affine, output_affine)
