"""Integration tests: VoxWhisperDataset → DataLoader → VoxWhisper forward."""
from __future__ import annotations

import copy
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess.generate_mock_dataset import make_mock_cohort
from src.dataset import VoxWhisperDataset
from src.models.vox_whisper import VoxWhisper
from src.utils.config import load_config, resolve_path
from src.utils.metrics import DiceBCELoss


def _make_test_config(tmp_path: Path) -> dict:
    """Copy default config with small volumes/patches under a temp directory."""
    cfg = copy.deepcopy(load_config())
    cfg["data"]["paths"]["processed"] = str(tmp_path / "processed")
    cfg["data"]["paths"]["cache"] = str(tmp_path / "cache")
    cfg["data"]["mock_volume_shape"] = [64, 64, 64]
    cfg["data"]["patch"]["size"] = [32, 32, 32]
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

        t1, t2, text_emb, gt_mask = next(iter(loader))

        assert t1.shape == (batch_size, 1, *patch_size)
        assert t2.shape == (batch_size, 1, *patch_size)
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

        t1, t2, text_emb, _ = next(iter(loader))
        with torch.no_grad():
            predictions = model(t1, t2, text_emb)

        assert len(predictions) == 3
        for pred in predictions:
            assert pred.ndim == 5
            assert pred.shape[0] == t1.shape[0]
            assert pred.shape[1] == text_emb.shape[1]


def test_model_output_spatial_dims_match_deep_supervision():
    with temp_test_config() as cfg:
        patch_size = tuple(cfg["data"]["patch"]["size"])
        n_prompts = len(cfg["data"]["prompts"])

        dataset = VoxWhisperDataset(cfg, training=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = VoxWhisper.from_config(cfg)
        model.eval()

        t1, t2, text_emb, _ = next(iter(loader))
        with torch.no_grad():
            predictions = model(t1, t2, text_emb)

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

        t1, t2, text_emb, gt_mask = next(iter(loader))
        predictions = model(t1, t2, text_emb)

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


def test_validation_dataset_uses_center_crop():
    """training=False should yield deterministic center patches."""
    with temp_test_config() as cfg:
        dataset = VoxWhisperDataset(cfg, training=False)

        primary_a, _, _, _ = dataset[0]
        primary_b, _, _, _ = dataset[0]

        torch.testing.assert_close(primary_a, primary_b)


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
        test_validation_dataset_uses_center_crop,
        test_gt_mask_channels_are_binary,
    ]
    for test_fn in tests:
        test_fn()
        print(f"ok {test_fn.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
