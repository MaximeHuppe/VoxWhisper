"""Integration tests: VoxWhisperDataset → DataLoader → VoxWhisper forward."""
from __future__ import annotations

import copy
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from voxwhisper.data.dataset import VoxWhisperDataset
from voxwhisper.models.vox_whisper import VoxWhisper
from voxwhisper.config import load_config, resolve_path
from voxwhisper.training.loss import DiceBCELoss


def _make_mock_cohort(cfg: dict, num_subjects: int = 4) -> None:
    """Create minimal synthetic processed data (no raw diffusion needed)."""
    import numpy as np
    import nibabel as nib
    from voxwhisper.data.nifti_io import volume_path, mask_path, subject_processed_dir
    from voxwhisper.config import ensure_dir

    processed_dir = Path(cfg["data"]["paths"]["processed"])
    vol_shape = tuple(cfg["data"].get("mock_volume_shape", [64, 64, 64]))
    n_prompts = len(cfg["data"]["prompts"])

    for i in range(num_subjects):
        subject_id = f"sub{i:04d}"
        subj_dir = subject_processed_dir(processed_dir, subject_id)
        ensure_dir(subj_dir)
        affine = np.eye(4)

        for modality in ("t1", "fa"):
            vol = np.random.randn(*vol_shape).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, affine), str(volume_path(processed_dir, subject_id, modality)))

        label = np.zeros(vol_shape, dtype=np.uint8)
        center = tuple(s // 2 for s in vol_shape)
        r = vol_shape[0] // 8
        label[
            center[0] - r : center[0] + r,
            center[1] - r : center[1] + r,
            center[2] - r : center[2] + r,
        ] = 1
        nib.save(nib.Nifti1Image(label, affine), str(mask_path(processed_dir, subject_id)))


def _make_test_config(tmp_path: Path) -> dict:
    """Copy default config with small volumes/patches under a temp directory."""
    cfg = copy.deepcopy(load_config())
    cfg["data"]["paths"]["processed"] = str(tmp_path / "processed")
    cfg["data"]["paths"]["cache"] = str(tmp_path / "cache")
    cfg["data"]["mock_volume_shape"] = [64, 64, 64]
    cfg["data"]["patch"]["size"] = [32, 32, 32]
    cfg["data"]["patch"]["train_patches_per_subject"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["dataloader"]["num_workers"] = 0
    return cfg


def _bootstrap_fixture_data(tmp_path: Path) -> dict:
    cfg = _make_test_config(tmp_path)
    _make_mock_cohort(cfg, num_subjects=4)

    cache_dir = resolve_path(cfg, "data.paths.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_prompts = len(cfg["data"]["prompts"])
    text_dim = cfg["model"]["text_dim"]
    embeddings = torch.randn(n_prompts, text_dim)
    cache_file = cache_dir / cfg["text_encoder"]["cache_file"]
    torch.save(embeddings, cache_file)
    return cfg


@contextmanager
def temp_test_config():
    with tempfile.TemporaryDirectory() as tmp:
        yield _bootstrap_fixture_data(Path(tmp))


def test_dataset_item_shapes_and_dtypes():
    with temp_test_config() as cfg:
        patch_size = tuple(cfg["data"]["patch"]["size"])
        n_prompts = len(cfg["data"]["prompts"])
        text_dim = cfg["model"]["text_dim"]

        dataset = VoxWhisperDataset(cfg, training=True)
        primary, secondary, text_emb, gt_mask = dataset[0]

        assert primary.dtype == torch.float32
        assert secondary.dtype == torch.float32
        assert gt_mask.dtype == torch.float32
        assert text_emb.dtype == torch.float32

        assert primary.shape == (1, *patch_size)
        assert secondary.shape == (1, *patch_size)
        assert gt_mask.shape == (n_prompts, *patch_size)
        assert text_emb.shape == (n_prompts, text_dim)


def test_dataset_patch_size_matches_config():
    with temp_test_config() as cfg:
        patch_size = tuple(cfg["data"]["patch"]["size"])
        dataset = VoxWhisperDataset(cfg, training=True)

        for idx in range(len(dataset)):
            primary, secondary, _, gt_mask = dataset[idx]
            assert tuple(primary.shape[1:]) == patch_size
            assert tuple(secondary.shape[1:]) == patch_size
            assert tuple(gt_mask.shape[1:]) == patch_size


def test_dataloader_batch_shapes():
    with temp_test_config() as cfg:
        patch_size = tuple(cfg["data"]["patch"]["size"])
        batch_size = cfg["training"]["batch_size"]
        n_prompts = len(cfg["data"]["prompts"])
        text_dim = cfg["model"]["text_dim"]

        dataset = VoxWhisperDataset(cfg, training=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        primary, secondary, text_emb, gt_mask = next(iter(loader))

        assert primary.shape == (batch_size, 1, *patch_size)
        assert secondary.shape == (batch_size, 1, *patch_size)
        assert text_emb.shape == (batch_size, n_prompts, text_dim)
        assert gt_mask.shape == (batch_size, n_prompts, *patch_size)


def test_model_forward_accepts_dataloader_batch():
    with temp_test_config() as cfg:
        dataset = VoxWhisperDataset(cfg, training=True)
        loader = DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
        )
        model = VoxWhisper.from_config(cfg)
        model.eval()

        primary, secondary, text_emb, _ = next(iter(loader))
        with torch.no_grad():
            predictions = model(primary, secondary, text_emb)

        assert len(predictions) == 3
        for pred in predictions:
            assert pred.ndim == 5
            assert pred.shape[0] == primary.shape[0]
            assert pred.shape[1] == text_emb.shape[1]


def test_model_output_spatial_dims_match_deep_supervision():
    with temp_test_config() as cfg:
        patch_size = tuple(cfg["data"]["patch"]["size"])
        n_prompts = len(cfg["data"]["prompts"])

        dataset = VoxWhisperDataset(cfg, training=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = VoxWhisper.from_config(cfg)
        model.eval()

        primary, secondary, text_emb, _ = next(iter(loader))
        with torch.no_grad():
            predictions = model(primary, secondary, text_emb)

        d, h, w = patch_size
        expected_spatial = [
            (d // 4, h // 4, w // 4),
            (d // 2, h // 2, w // 2),
            (d, h, w),
        ]
        for pred, spatial in zip(predictions, expected_spatial):
            assert pred.shape == (1, n_prompts, *spatial)


def test_training_step_with_dataloader_batch():
    with temp_test_config() as cfg:
        deep_sup_weights = cfg["training"]["deep_supervision_weights"]

        dataset = VoxWhisperDataset(cfg, training=True)
        loader = DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
        )
        model = VoxWhisper.from_config(cfg)
        criterion = DiceBCELoss()
        model.train()

        primary, secondary, text_emb, gt_mask = next(iter(loader))
        predictions = model(primary, secondary, text_emb)

        batch_loss = 0.0
        for idx, pred in enumerate(predictions):
            downsampled_target = F.interpolate(
                gt_mask,
                size=pred.shape[2:],
                mode="trilinear",
                align_corners=True,
            )
            batch_loss = batch_loss + deep_sup_weights[idx] * criterion(
                pred, downsampled_target
            )

        batch_loss.backward()
        assert batch_loss.item() > 0


def test_validation_uses_frozen_patches_per_subject():
    with temp_test_config() as cfg:
        n_patches = int(cfg["data"]["patch"]["val_patches_per_subject"])
        dataset = VoxWhisperDataset(cfg, training=False)
        assert len(dataset) == len(dataset.subject_ids) * n_patches

        epoch_a = [dataset[i] for i in range(len(dataset))]
        epoch_b = [dataset[i] for i in range(len(dataset))]
        for (p_a, s_a, _, gt_a), (p_b, s_b, _, gt_b) in zip(epoch_a, epoch_b):
            torch.testing.assert_close(p_a, p_b)
            torch.testing.assert_close(s_a, s_b)
            torch.testing.assert_close(gt_a, gt_b)

        other = VoxWhisperDataset(cfg, training=False)
        torch.testing.assert_close(dataset[0][0], other[0][0])

        # First frozen val crop is the foreground centroid — some tract channel
        # must be present, not necessarily the first positive label.
        _, _, _, gt_mask = dataset[0]
        assert gt_mask[1:].sum() > 0


def test_training_uses_adaptive_patch_sampling():
    """Train crops should vary across draws of the same subject (50/50 sampling)."""
    with temp_test_config() as cfg:
        dataset = VoxWhisperDataset(cfg, training=True)
        patches = [dataset[0][0] for _ in range(24)]
        differs = any(not torch.equal(patches[0], p) for p in patches[1:])
        assert differs, "expected adaptive train sampling to produce more than one crop"


def test_training_length_is_patches_per_subject():
    with temp_test_config() as cfg:
        cfg["data"]["patch"]["train_patches_per_subject"] = 4
        dataset = VoxWhisperDataset(cfg, training=True)
        assert len(dataset) == len(dataset.subject_ids) * 4


def test_gt_mask_channels_are_binary():
    with temp_test_config() as cfg:
        dataset = VoxWhisperDataset(cfg, training=True)
        _, _, _, gt_mask = dataset[0]

        unique_vals = set(gt_mask.flatten().tolist())
        assert unique_vals <= {0.0, 1.0}


def _run_all():
    tests = [
        test_dataset_item_shapes_and_dtypes,
        test_dataset_patch_size_matches_config,
        test_dataloader_batch_shapes,
        test_model_forward_accepts_dataloader_batch,
        test_model_output_spatial_dims_match_deep_supervision,
        test_training_step_with_dataloader_batch,
        test_validation_uses_frozen_patches_per_subject,
        test_training_uses_adaptive_patch_sampling,
        test_training_length_is_patches_per_subject,
        test_gt_mask_channels_are_binary,
    ]
    for test_fn in tests:
        test_fn()
        print(f"ok {test_fn.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
