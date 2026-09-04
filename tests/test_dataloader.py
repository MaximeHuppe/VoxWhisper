"""Integration tests: VoxDenseDataset → DataLoader → VoxDense forward."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from voxwhisper.util.config import resolve_path
from voxwhisper.data.dataset import VoxDenseDataset
from voxwhisper.models.vox_dense import VoxDense
from voxwhisper.training.loss import DiceBCELoss


def _make_mock_cohort(cfg: dict, num_subjects: int = 4) -> None:
    import nibabel as nib
    import numpy as np

    from voxwhisper.util.config import ensure_dir
    from voxwhisper.data.nifti_io import mask_path, subject_processed_dir, volume_path

    processed_dir = Path(cfg["data"]["paths"]["processed"])
    vol_shape = tuple(cfg["data"].get("mock_volume_shape", [32, 32, 32]))
    n_prompts = len(cfg["data"]["prompts"])

    for i in range(num_subjects):
        subject_id = f"sub{i:04d}"
        subj_dir = subject_processed_dir(processed_dir, subject_id)
        ensure_dir(subj_dir)
        affine = np.eye(4)

        vol = np.random.randn(*vol_shape).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine), str(volume_path(processed_dir, subject_id, "t1")))

        label = np.zeros(vol_shape, dtype=np.uint8)
        center = tuple(s // 2 for s in vol_shape)
        r = max(vol_shape[0] // 8, 2)
        label[
            center[0] - r : center[0] + r,
            center[1] - r : center[1] + r,
            center[2] - r : center[2] + r,
        ] = 1
        nib.save(nib.Nifti1Image(label, affine), str(mask_path(processed_dir, subject_id)))

    _ = n_prompts


def _bootstrap(tmp_config: dict) -> dict:
    cfg = copy.deepcopy(tmp_config)
    cfg["data"]["mock_volume_shape"] = [32, 32, 32]
    cfg["data"]["patch"]["size"] = [16, 16, 16]
    cfg["data"]["patch"]["train_patches_per_subject"] = 1
    cfg["training"]["batch_size"] = 2
    cfg["training"]["dataloader"]["num_workers"] = 0
    _make_mock_cohort(cfg, num_subjects=4)

    cache_dir = resolve_path(cfg, "data.paths.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_prompts = len(cfg["data"]["prompts"])
    text_dim = cfg["model"]["text_dim"]
    torch.save(torch.randn(n_prompts, text_dim), cache_dir / cfg["text_encoder"]["cache_file"])
    return cfg


@pytest.fixture()
def dense_cfg(tmp_config):
    return _bootstrap(tmp_config)


def test_dataset_item_shapes_and_dtypes(dense_cfg):
    patch_size = tuple(dense_cfg["data"]["patch"]["size"])
    k = int(dense_cfg["data"]["patch"]["prompts_per_crop"])
    text_dim = dense_cfg["model"]["text_dim"]

    dataset = VoxDenseDataset(dense_cfg, training=True)
    volume, text_emb, gt_mask = dataset[0]

    assert volume.dtype == torch.float32
    assert gt_mask.dtype == torch.float32
    assert text_emb.dtype == torch.float32
    assert volume.shape == (1, *patch_size)
    assert gt_mask.shape == (k, *patch_size)
    assert text_emb.shape == (k, text_dim)


def test_val_uses_all_prompts(dense_cfg):
    n_prompts = len(dense_cfg["data"]["prompts"])
    dataset = VoxDenseDataset(dense_cfg, training=False)
    _, text_emb, gt_mask = dataset[0]
    patch_size = tuple(dense_cfg["data"]["patch"]["size"])
    assert text_emb.shape[0] == n_prompts
    assert gt_mask.shape == (n_prompts, *patch_size)


def test_dataloader_batch_shapes(dense_cfg):
    patch_size = tuple(dense_cfg["data"]["patch"]["size"])
    batch_size = dense_cfg["training"]["batch_size"]
    k = int(dense_cfg["data"]["patch"]["prompts_per_crop"])
    text_dim = dense_cfg["model"]["text_dim"]

    dataset = VoxDenseDataset(dense_cfg, training=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    volume, text_emb, gt_mask = next(iter(loader))

    assert volume.shape == (batch_size, 1, *patch_size)
    assert text_emb.shape == (batch_size, k, text_dim)
    assert gt_mask.shape == (batch_size, k, *patch_size)


def test_model_forward_accepts_dataloader_batch(dense_cfg):
    dataset = VoxDenseDataset(dense_cfg, training=True)
    loader = DataLoader(dataset, batch_size=dense_cfg["training"]["batch_size"], shuffle=False)
    model = VoxDense.from_config(dense_cfg)
    model.eval()

    volume, text_emb, _ = next(iter(loader))
    with torch.no_grad():
        predictions = model(volume, text_emb)

    assert len(predictions) == 1
    for pred in predictions:
        assert pred.ndim == 5
        assert pred.shape[0] == volume.shape[0]
        assert pred.shape[1] == text_emb.shape[1]


def test_training_step_with_dataloader_batch(dense_cfg):
    deep_sup_weights = dense_cfg["training"]["deep_supervision_weights"]
    dataset = VoxDenseDataset(dense_cfg, training=True)
    loader = DataLoader(dataset, batch_size=dense_cfg["training"]["batch_size"], shuffle=False)
    model = VoxDense.from_config(dense_cfg)
    criterion = DiceBCELoss()
    model.train()

    volume, text_emb, gt_mask = next(iter(loader))
    predictions = model(volume, text_emb)

    batch_loss = 0.0
    for idx, pred in enumerate(predictions):
        downsampled_target = F.interpolate(
            gt_mask, size=pred.shape[2:], mode="trilinear", align_corners=True
        )
        batch_loss = batch_loss + deep_sup_weights[idx] * criterion(pred, downsampled_target)

    batch_loss.backward()
    assert batch_loss.item() > 0


def test_validation_uses_frozen_patches_per_subject(dense_cfg):
    n_patches = int(dense_cfg["data"]["patch"]["val_patches_per_subject"])
    dataset = VoxDenseDataset(dense_cfg, training=False)
    assert len(dataset) == len(dataset.subject_ids) * n_patches

    epoch_a = [dataset[i] for i in range(len(dataset))]
    epoch_b = [dataset[i] for i in range(len(dataset))]
    for (v_a, t_a, gt_a), (v_b, t_b, gt_b) in zip(epoch_a, epoch_b):
        torch.testing.assert_close(v_a, v_b)
        torch.testing.assert_close(t_a, t_b)
        torch.testing.assert_close(gt_a, gt_b)

    other = VoxDenseDataset(dense_cfg, training=False)
    torch.testing.assert_close(dataset[0][0], other[0][0])

    _, _, gt_mask = dataset[0]
    assert gt_mask[1:].sum() > 0


def test_training_length_is_patches_per_subject(dense_cfg):
    dense_cfg["data"]["patch"]["train_patches_per_subject"] = 4
    dataset = VoxDenseDataset(dense_cfg, training=True)
    assert len(dataset) == len(dataset.subject_ids) * 4


def test_gt_mask_channels_are_binary(dense_cfg):
    dataset = VoxDenseDataset(dense_cfg, training=True)
    _, _, gt_mask = dataset[0]
    unique_vals = set(gt_mask.flatten().tolist())
    assert unique_vals <= {0.0, 1.0}


def test_encoder_checkpoint_is_reloadable(dense_cfg, tmp_path):
    from voxwhisper.training.checkpoint import load_encoder_state, save_checkpoint

    model = VoxDense.from_config(dense_cfg)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, 1, model, torch.optim.Adam(model.parameters(), lr=1e-3), {"train_loss": 1.0}, dense_cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "encoder_state_dict" in payload

    fresh = VoxDense.from_config(dense_cfg)
    load_encoder_state(fresh, payload)
    for a, b in zip(model.encoder.parameters(), fresh.encoder.parameters()):
        torch.testing.assert_close(a, b)


def test_whisper_dataset_returns_t1_fa(tmp_whisper_config):
    import nibabel as nib
    import numpy as np

    from voxwhisper.data.dataset import VoxWhisperDataset
    from voxwhisper.data.nifti_io import mask_path, subject_processed_dir, volume_path
    from voxwhisper.util.config import ensure_dir, resolve_path
    from voxwhisper.util.stage import build_dataset, unpack_batch

    cfg = copy.deepcopy(tmp_whisper_config)
    cfg["data"]["patch"]["size"] = [16, 16, 16]
    processed = Path(cfg["data"]["paths"]["processed"])
    vol_shape = (32, 32, 32)
    affine = np.eye(4)
    for i in range(2):
        sid = f"sub{i:04d}"
        ensure_dir(subject_processed_dir(processed, sid))
        vol = np.random.randn(*vol_shape).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine), str(volume_path(processed, sid, "t1")))
        nib.save(nib.Nifti1Image(vol, affine), str(volume_path(processed, sid, "fa")))
        label = np.zeros(vol_shape, dtype=np.uint8)
        label[8:16, 8:16, 8:16] = 1
        nib.save(nib.Nifti1Image(label, affine), str(mask_path(processed, sid)))

    cache_dir = resolve_path(cfg, "data.paths.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        torch.randn(len(cfg["data"]["prompts"]), cfg["model"]["text_dim"]),
        cache_dir / cfg["text_encoder"]["cache_file"],
    )

    dataset = build_dataset(cfg, training=True)
    assert isinstance(dataset, VoxWhisperDataset)
    primary, secondary, text, gt = unpack_batch(dataset[0])
    patch = tuple(cfg["data"]["patch"]["size"])
    n_prompts = len(cfg["data"]["prompts"])
    assert primary.shape == (1, *patch)
    assert secondary.shape == (1, *patch)
    assert text.shape == (n_prompts, cfg["model"]["text_dim"])
    assert gt.shape == (n_prompts, *patch)
