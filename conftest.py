"""Project-wide pytest configuration and shared fixtures.

Place fixtures here that are needed by tests in more than one subdirectory.
Fixtures that are local to a single test module can remain in that module.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_config() -> dict:
    """A minimal, in-memory config that matches the schema expected by
    ``VoxWhisperDataset``, the preprocessors, and the training loop.

    Uses tiny spatial dimensions so tests run in seconds on CPU.
    """
    return {
        "data": {
            "paths": {
                "processed": "",          # overridden per-test via tmp_config
                "raw": "",
                "runs": "",
                "splits": "",
                "cache": "",
            },
            "modalities": {
                "primary": "t1",
                "secondary": "b0",
            },
            "mock_volume_shape": [32, 32, 32],
            "masks": {
                "structures": ["background", "AF_L", "AF_R", "CST_L"],
                "directory": "tract_masks_1.25",
            },
            "prompts": ["background", "AF_L", "AF_R", "CST_L"],
        },
        "splits": {"enabled": False},
        "training": {
            "run_name": "test",
            "epochs": 2,
            "batch_size": 1,
            "learning_rate": 1e-4,
            "deep_supervision_weights": [1.0, 0.5, 0.25],
            "dataloader": {
                "num_workers": 0,
                "pin_memory": False,
                "shuffle": False,
                "drop_last": False,
            },
            "checkpoint": {
                "monitor": "loss",
                "every": 1,
                "keep": 2,
                "keep_periodic": False,
            },
            "seed": 42,
        },
        "inference": {"threshold": 0.5},
        "model": {
            "patch_size": [16, 16, 16],
            "patch_overlap": [4, 4, 4],
            "encoder": {
                "in_channels": 1,
                "base_filters": 8,
                "depth": 3,
            },
            "cross_attention": {
                "embed_dim": 64,
                "num_heads": 2,
                "dropout": 0.0,
            },
            "decoder": {
                "base_filters": 8,
            },
            "text_encoder": {
                "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                "embed_dim": 768,
                "cache_path": "",
            },
        },
    }


@pytest.fixture()
def tmp_config(tmp_path: Path, base_config: dict) -> dict:
    """A copy of ``base_config`` with all paths redirected to a temp directory.

    The temp directory is removed automatically after each test.
    """
    import copy
    cfg = copy.deepcopy(base_config)
    cfg["data"]["paths"]["processed"] = str(tmp_path / "processed")
    cfg["data"]["paths"]["raw"] = str(tmp_path / "raw")
    cfg["data"]["paths"]["runs"] = str(tmp_path / "runs")
    cfg["data"]["paths"]["splits"] = str(tmp_path / "splits")
    cfg["data"]["paths"]["cache"] = str(tmp_path / "cache")
    cfg["model"]["text_encoder"]["cache_path"] = str(tmp_path / "cache" / "embeddings.pt")
    return cfg


@pytest.fixture()
def mock_cohort(tmp_config: dict) -> dict:
    """Generate a small mock cohort under the processed path and return the config."""
    from preprocess.generate_mock_dataset import make_mock_cohort
    make_mock_cohort(tmp_config, num_subjects=4, seed=0)
    return tmp_config
