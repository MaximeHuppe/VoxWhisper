"""Global RNG seeding for reproducible training."""
from __future__ import annotations

import random
from typing import Mapping

import numpy as np
import torch


def get_training_seed(config: Mapping) -> int:
    """Resolve the training seed from config (``training.seed`` → ``splits.seed`` → 42)."""
    train_cfg = config.get("training", {})
    if train_cfg.get("seed") is not None:
        return int(train_cfg["seed"])
    return int(config.get("splits", {}).get("seed", 42))


def set_global_seed(seed: int) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch RNGs; return a torch Generator for DataLoader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
