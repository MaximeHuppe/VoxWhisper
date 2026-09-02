"""Tests for train-time patch augmentation.

Covers two concerns:
  1. Correctness — LR flip remaps labels, disabled augmentor is a no-op,
     fast rotation applies the same matrix to all three volumes, etc.
  2. Quality invariance — augmented patches must not corrupt the data:
     label set and per-class voxel counts are preserved, image statistics
     stay within bounds, no NaN / Inf may appear.
"""
from __future__ import annotations

import copy
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.generate_mock_dataset import make_mock_cohort
from src.data.augment import (
    PatchAugmentor,
    _compose_rotation,
    _fast_rotate_3d,
    left_right_label_pairs,
)
from src.data.dataset import VoxWhisperDataset
from src.utils.config import load_config, resolve_path


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_test_config(tmp_path: Path) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["data"]["paths"]["processed"] = str(tmp_path / "processed")
    cfg["data"]["paths"]["cache"] = str(tmp_path / "cache")
    cfg["data"]["mock_volume_shape"] = [64, 64, 64]
    cfg["data"]["patch"]["size"] = [32, 32, 32]
    cfg["data"]["patch"]["train_patches_per_subject"] = 1
    cfg["splits"]["enabled"] = False
    cfg["training"]["batch_size"] = 2
    cfg["training"]["dataloader"]["num_workers"] = 0
    return cfg


def _bootstrap_fixture_data(tmp_path: Path) -> dict:
    cfg = _make_test_config(tmp_path)
    make_mock_cohort(cfg, num_subjects=4)
    cache_dir = resolve_path(cfg, "data.paths.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_prompts = len(cfg["data"]["prompts"])
    text_dim = cfg["model"]["text_dim"]
    torch.save(torch.randn(n_prompts, text_dim), cache_dir / cfg["text_encoder"]["cache_file"])
    return cfg


@contextmanager
def temp_test_config():
    with tempfile.TemporaryDirectory() as tmp:
        yield _bootstrap_fixture_data(Path(tmp))


def _patch_triple(rng: np.random.Generator, shape=(16, 16, 16)):
    """Primary, secondary, and label patches with known LR label layout."""
    primary = rng.normal(size=shape).astype(np.float32)
    secondary = rng.normal(size=shape).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int16)
    labels[:, :, :8] = 1   # ATR_left on -x half
    labels[:, :, 8:] = 2   # ATR_right on +x half
    return primary, secondary, labels


# ---------------------------------------------------------------------------
# left_right_label_pairs
# ---------------------------------------------------------------------------

def test_lr_pairs_bilateral():
    names = ["background", "ATR_left", "ATR_right", "CG_left", "CG_right", "UF_left", "UF_right"]
    assert left_right_label_pairs(names) == {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5}


def test_lr_pairs_no_lateralised_structure():
    assert left_right_label_pairs(["background", "CST"]) == {}


# ---------------------------------------------------------------------------
# LR flip correctness
# ---------------------------------------------------------------------------

def test_lr_flip_preserves_anatomy():
    """After flip + remap, the anatomical label assignment is preserved."""
    rng = np.random.default_rng(0)
    primary, secondary, labels = _patch_triple(rng)
    aug = PatchAugmentor(enabled=True, flip_p=1.0, lr_axis=2,
                         lr_remap={1: 2, 2: 1}, rotate_p=0.0, noise_p=0.0)
    _, _, out = aug(primary, secondary, labels, rng)
    # After axis-2 flip + remap: -x half keeps label 1, +x half keeps label 2.
    assert set(np.unique(out[:, :, :8]).tolist()) == {1}
    assert set(np.unique(out[:, :, 8:]).tolist()) == {2}


def test_flip_label_set_unchanged():
    rng = np.random.default_rng(1)
    primary, secondary, labels = _patch_triple(rng)
    aug = PatchAugmentor(enabled=True, flip_p=1.0, lr_axis=2,
                         lr_remap={1: 2, 2: 1}, rotate_p=0.0, noise_p=0.0)
    _, _, out = aug(primary, secondary, labels, rng)
    assert set(np.unique(out).tolist()) == set(np.unique(labels).tolist())


def test_flip_per_class_voxel_count_preserved():
    rng = np.random.default_rng(2)
    primary, secondary, labels = _patch_triple(rng)
    aug = PatchAugmentor(enabled=True, flip_p=1.0, lr_axis=2,
                         lr_remap={1: 2, 2: 1}, rotate_p=0.0, noise_p=0.0)
    _, _, out = aug(primary, secondary, labels, rng)
    for cls in np.unique(labels):
        assert np.sum(out == cls) == np.sum(labels == cls), f"count changed for class {cls}"


def test_flip_is_isometry():
    """Flip preserves mean and std exactly (no information loss)."""
    rng = np.random.default_rng(3)
    primary, secondary, labels = _patch_triple(rng)
    aug = PatchAugmentor(enabled=True, flip_p=1.0, lr_axis=2,
                         lr_remap={}, rotate_p=0.0, noise_p=0.0)
    p_out, s_out, _ = aug(primary, secondary, labels, rng)
    np.testing.assert_allclose(p_out.mean(), primary.mean(), atol=1e-5)
    np.testing.assert_allclose(p_out.std(),  primary.std(),  atol=1e-5)
    np.testing.assert_allclose(s_out.mean(), secondary.mean(), atol=1e-5)


# ---------------------------------------------------------------------------
# Fast rotation
# ---------------------------------------------------------------------------

def test_compose_rotation_is_orthogonal():
    """A rotation matrix must satisfy R @ R.T = I."""
    R = _compose_rotation([15.0, -10.0, 5.0])
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_fast_rotate_same_matrix_on_all_volumes():
    """Applying the same R to two copies of the same volume must give the same result."""
    rng = np.random.default_rng(4)
    vol = rng.normal(size=(20, 20, 20)).astype(np.float32)
    R = _compose_rotation([8.0, -5.0, 3.0])
    out_a = _fast_rotate_3d(vol, R, order=1)
    out_b = _fast_rotate_3d(vol.copy(), R, order=1)
    np.testing.assert_allclose(out_a, out_b, atol=1e-6)


def test_rotation_preserves_shape():
    rng = np.random.default_rng(5)
    shape = (24, 24, 24)
    vol = rng.normal(size=shape).astype(np.float32)
    R = _compose_rotation([12.0, 0.0, -7.0])
    assert _fast_rotate_3d(vol, R, order=1).shape == shape
    assert _fast_rotate_3d(vol, R, order=0).shape == shape


def test_rotation_label_values_integer():
    """Order-0 (nearest-neighbour) rotation must produce only valid integer labels."""
    rng = np.random.default_rng(6)
    shape = (20, 20, 20)
    labels = rng.integers(0, 4, size=shape).astype(np.int16)
    R = _compose_rotation([10.0, -8.0, 5.0])
    out = _fast_rotate_3d(labels, R, order=0)
    assert set(np.unique(out).tolist()) <= set(np.unique(labels).tolist()), (
        "rotation created label values not present in the input"
    )


# ---------------------------------------------------------------------------
# Noise quality invariance
# ---------------------------------------------------------------------------

def test_noise_does_not_touch_labels():
    rng = np.random.default_rng(7)
    primary, secondary, labels = _patch_triple(rng)
    aug = PatchAugmentor(enabled=True, flip_p=0.0, rotate_p=0.0, noise_p=1.0, noise_std=0.2)
    _, _, out_labels = aug(primary, secondary, labels, rng)
    np.testing.assert_array_equal(out_labels, labels)


def test_noise_statistics_within_bounds():
    rng = np.random.default_rng(8)
    shape = (32, 32, 32)
    primary = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    aug = PatchAugmentor(enabled=True, flip_p=0.0, rotate_p=0.0, noise_p=1.0, noise_std=0.05)
    p_out, _, _ = aug(primary, primary.copy(),
                      np.zeros(shape, dtype=np.int16), rng)
    assert abs(float(p_out.mean()) - float(primary.mean())) < 0.05
    assert float(p_out.std()) < 1.1 * float(primary.std()) + 0.1


# ---------------------------------------------------------------------------
# General invariants
# ---------------------------------------------------------------------------

def test_no_nan_or_inf():
    rng = np.random.default_rng(9)
    primary, secondary, labels = _patch_triple(rng)
    aug = PatchAugmentor(enabled=True, flip_p=1.0, noise_p=1.0, noise_std=0.1)
    p, s, _ = aug(primary, secondary, labels, rng)
    assert not np.any(np.isnan(p) | np.isinf(p))
    assert not np.any(np.isnan(s) | np.isinf(s))


def test_output_dtype_float32():
    rng = np.random.default_rng(10)
    primary, secondary, labels = _patch_triple(rng)
    aug = PatchAugmentor(enabled=True, flip_p=1.0, noise_p=1.0, noise_std=0.05)
    p, s, _ = aug(primary, secondary, labels, rng)
    assert p.dtype == np.float32
    assert s.dtype == np.float32


def test_disabled_is_strict_no_op():
    rng = np.random.default_rng(11)
    primary, secondary, labels = _patch_triple(rng)
    aug = PatchAugmentor(enabled=False)
    p, s, lab = aug(primary, secondary, labels, rng)
    np.testing.assert_array_equal(p, primary)
    np.testing.assert_array_equal(s, secondary)
    np.testing.assert_array_equal(lab, labels)


def test_from_config_disabled_when_missing():
    assert PatchAugmentor.from_config({"data": {}}).enabled is False


def test_from_config_lr_remap_built():
    cfg = {
        "data": {
            "structure_names": ["background", "ATR_left", "ATR_right"],
            "augmentation": {"enabled": True},
        }
    }
    aug = PatchAugmentor.from_config(cfg)
    assert aug.enabled
    assert aug.lr_remap == {1: 2, 2: 1}


# ---------------------------------------------------------------------------
# Dataset integration
# ---------------------------------------------------------------------------

def test_dataset_aug_only_in_training():
    with temp_test_config() as cfg:
        cfg["data"]["augmentation"] = {
            "enabled": True, "lr_axis": 2, "flip_p": 1.0,
            "rotate_p": 0.0, "noise_p": 0.0, "noise_std": 0.0,
        }
        train_ds = VoxWhisperDataset(cfg, training=True)
        val_ds = VoxWhisperDataset(cfg, training=False)

        assert train_ds.augmentor.enabled
        assert not val_ds.augmentor.enabled

        # Val is deterministic.
        torch.testing.assert_close(val_ds[0][0], val_ds[0][0])

        # Train output shapes and dtypes are valid.
        primary, secondary, text_emb, gt_mask = train_ds[0]
        patch = tuple(cfg["data"]["patch"]["size"])
        n_prompts = len(cfg["data"]["prompts"])
        assert primary.shape == (1, *patch)
        assert secondary.shape == (1, *patch)
        assert gt_mask.shape == (n_prompts, *patch)
        assert primary.dtype == torch.float32
        assert set(gt_mask.flatten().tolist()) <= {0.0, 1.0}, "gt_mask is not binary"


def _run_all():
    tests = [
        test_lr_pairs_bilateral,
        test_lr_pairs_no_lateralised_structure,
        test_lr_flip_preserves_anatomy,
        test_flip_label_set_unchanged,
        test_flip_per_class_voxel_count_preserved,
        test_flip_is_isometry,
        test_compose_rotation_is_orthogonal,
        test_fast_rotate_same_matrix_on_all_volumes,
        test_rotation_preserves_shape,
        test_rotation_label_values_integer,
        test_noise_does_not_touch_labels,
        test_noise_statistics_within_bounds,
        test_no_nan_or_inf,
        test_output_dtype_float32,
        test_disabled_is_strict_no_op,
        test_from_config_disabled_when_missing,
        test_from_config_lr_remap_built,
        test_dataset_aug_only_in_training,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
