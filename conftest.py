"""Project-wide pytest configuration and shared fixtures.

Fixtures here are available to all test modules. Test-local fixtures can stay
in their respective modules.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def base_config() -> dict:
    """Minimal in-memory config matching the ``voxdense.yaml`` schema.

    Uses tiny spatial dimensions so tests run in seconds on CPU.
    Paths are empty strings; override them per-test via ``tmp_config``.
    """
    return {
        "data": {
            "paths": {
                "processed": "",
                "raw": "",
                "runs": "",
                "cache": "",
            },
            "volumes": {
                "t1": {"filename": "T1w.nii.gz"},
                "brainmask": {"filename": "brainmask_fs.nii.gz"},
                "wmparc": {"filename": "wmparc.nii.gz"},
            },
            "masks": {
                "structures": "config/structures_dense.json",
            },
            "nerve_masks": {"source": "nerve_masks_1.25"},
            "prompts": ["background", "left_thalamus", "right_thalamus", "brainstem"],
            "structure_names": ["background", "left_thalamus", "right_thalamus", "brainstem"],
            "patch": {
                "size": [16, 16, 16],
                "positive_ratio": 0.5,
                "train_patches_per_subject": 1,
                "val_patches_per_subject": 2,
                "positive_labels": [1, 2, 3],
                "prompts_per_crop": 2,
            },
        },
        "preprocessing": {
            "zscore_nonzero_only": True,
            "apply_brainmask": True,
        },
        "text_encoder": {
            "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "cache_file": "prompts_dense.pt",
        },
        "model": {
            "name": "VoxDense",
            "input_channels": 1,
            "text_dim": 8,
            "embed_dim": 16,
            "num_heads": 2,
            "encoder": {
                "channels": [8, 16],
                "strides": [2],
                "kernel_sizes": [3],
                "paddings": [1],
                "num_resblocks": [1],
            },
        },
        "splits": {
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "seed": 42,
            "manifest": "data/splits.json",
            "subject_split": "config/subject_split.json",
        },
        "training": {
            "run_name": "test",
            "seed": 42,
            "epochs": 2,
            "batch_size": 1,
            "learning_rate": 1e-4,
            "warmup_epochs": 0,
            "bce_weight": 0.5,
            "deep_supervision_weights": [1.0],
            "dataloader": {
                "num_workers": 0,
                "pin_memory": False,
                "shuffle": False,
                "drop_last": False,
            },
            "checkpoint": {
                "monitor": "dice",
                "keep": 2,
                "every": 1,
            },
        },
        "inference": {
            "threshold": 0.5,
            "overlap": 0.5,
            "mode": "gaussian",
            "sigma_scale": 0.125,
            "sw_batch_size": 1,
        },
        "logging": {"backend": "none"},
    }


@pytest.fixture()
def tmp_config(tmp_path: Path, base_config: dict) -> dict:
    """``base_config`` with all paths redirected to a temporary directory."""
    cfg = copy.deepcopy(base_config)
    cfg["data"]["paths"]["processed"] = str(tmp_path / "processed")
    cfg["data"]["paths"]["raw"] = str(tmp_path / "raw")
    cfg["data"]["paths"]["runs"] = str(tmp_path / "runs")
    cfg["data"]["paths"]["cache"] = str(tmp_path / "cache")
    cfg["splits"]["manifest"] = str(tmp_path / "splits.json")
    cfg["splits"]["subject_split"] = str(tmp_path / "subject_split.json")
    return cfg


@pytest.fixture()
def tmp_whisper_config(tmp_config: dict) -> dict:
    """Phase 2 (T1+FA nerve) config sharing the same temp paths as ``tmp_config``."""
    cfg = copy.deepcopy(tmp_config)
    cfg["model"]["name"] = "VoxWhisper"
    cfg["data"]["volumes"]["fa"] = {"filename": "dti_FA.nii.gz"}
    cfg["data"]["masks"]["source"] = "nerve_masks_1.25"
    cfg["data"]["prompts"] = ["background", "left_nerve", "right_nerve"]
    cfg["data"]["structure_names"] = ["background", "left_nerve", "right_nerve"]
    cfg["data"]["patch"]["positive_labels"] = [1, 2]
    cfg["text_encoder"]["cache_file"] = "prompts_nerve.pt"
    return cfg
