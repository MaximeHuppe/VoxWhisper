# src/dataset.py
"""Config-driven dataset with 50/50 foreground-optimized patch sampling."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import active_modality_keys, resolve_path
from src.utils.nifti_io import (
    extract_patch_3d,
    label_to_multichannel,
    list_subject_ids,
    load_nifti,
    mask_path,
    random_valid_center,
    volume_center,
    volume_path,
)


class VoxWhisperDataset(Dataset):
    """
    Loads full-resolution T1/T2 + mask NIfTIs and returns fixed-size patches.

    Training (``training=True``):
      - 50% positive patches centered on ``patch.positive_labels`` voxels
      - 50% random patches anywhere in the volume

    Validation / test (``training=False``):
      - Deterministic center crop of ``patch.size``
    """

    def __init__(
        self,
        config: Mapping,
        subject_ids: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        training: bool = True,
    ):
        self.config = config
        self.training = training
        self.primary, self.secondary = active_modality_keys(config)
        self.processed_dir = resolve_path(config, "data.paths.processed")

        patch_cfg = config["data"]["patch"]
        self.patch_size = tuple(int(x) for x in patch_cfg["size"])
        self.positive_ratio = float(patch_cfg.get("positive_ratio", 0.5))
        self.positive_labels = [int(x) for x in patch_cfg.get("positive_labels", [1])]
        self.rng = np.random.default_rng()

        cache_dir = resolve_path(config, "data.paths.cache")
        cache_file = cache_dir / config["text_encoder"]["cache_file"]

        all_subjects = list_subject_ids(self.processed_dir)
        if not all_subjects:
            raise FileNotFoundError(
                f"No processed subjects found in {self.processed_dir}. "
                "Expected layout: {{processed}}/{{subject_id}}/t1.nii.gz"
            )

        if subject_ids is None and split is not None:
            subject_ids = self._load_split_subjects(config, split)

        if subject_ids is not None:
            allowed = set(subject_ids)
            all_subjects = [s for s in all_subjects if s in allowed]

        valid = []
        for sid in all_subjects:
            p1 = volume_path(self.processed_dir, sid, self.primary)
            p2 = volume_path(self.processed_dir, sid, self.secondary)
            if p1.exists() and p2.exists():
                valid.append(sid)
            else:
                print(
                    f"Warning: skipping {sid} "
                    f"(missing {self.primary} and/or {self.secondary})"
                )

        self.subject_ids = valid
        if not self.subject_ids:
            raise ValueError("Dataset is empty after subject filtering")

        if not cache_file.exists():
            raise FileNotFoundError(f"Text embedding cache not found: {cache_file}")

        try:
            embeddings = torch.load(cache_file, map_location="cpu", weights_only=True)
        except TypeError:
            embeddings = torch.load(cache_file, map_location="cpu")
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(0)
        self.text_embeddings = embeddings
        self.n_prompts = embeddings.shape[0]

        expected_prompts = len(config["data"]["prompts"])
        if self.n_prompts != expected_prompts:
            raise ValueError(
                f"Cached embeddings have {self.n_prompts} prompts, "
                f"but config lists {expected_prompts}"
            )

    @staticmethod
    def _load_split_subjects(config: Mapping, split: str) -> list[str]:
        manifest_rel = config["splits"]["manifest"]
        manifest_path = Path(manifest_rel)
        if not manifest_path.is_absolute():
            from src.utils.config import get_project_root

            manifest_path = get_project_root() / manifest_path

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Split manifest not found: {manifest_path}. "
                "Generate splits first or set splits.enabled=false."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if split not in manifest:
            raise KeyError(f"Split '{split}' not in manifest keys {list(manifest)}")
        return list(manifest[split])

    def __len__(self):
        return len(self.subject_ids)

    def _load_volume_np(self, subject_id: str, modality: str) -> np.ndarray:
        path = volume_path(self.processed_dir, subject_id, modality)
        if not path.exists():
            raise FileNotFoundError(f"Missing volume: {path}")
        data, _ = load_nifti(path)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume at {path}, got shape {data.shape}")
        return data

    def _load_label_np(self, subject_id: str, spatial_shape: Tuple[int, int, int]) -> np.ndarray:
        path = mask_path(self.processed_dir, subject_id)
        if not path.exists():
            return np.zeros(spatial_shape, dtype=np.int16)

        data, _ = load_nifti(path)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D integer mask at {path}, got shape {data.shape}")
        if data.shape != spatial_shape:
            raise ValueError(
                f"Mask shape {data.shape} != volume shape {spatial_shape} for {subject_id}"
            )
        return np.rint(data).astype(np.int16)

    def _sample_center(self, labels: np.ndarray) -> Tuple[int, int, int]:
        """50/50 positive (foreground) vs random center for training patches."""
        shape = labels.shape

        if self.training and self.rng.random() < self.positive_ratio:
            pos_mask = np.zeros(shape, dtype=bool)
            for lab in self.positive_labels:
                pos_mask |= labels == lab
            nerve_voxels = np.argwhere(pos_mask)
            if len(nerve_voxels) > 0:
                idx = int(self.rng.integers(0, len(nerve_voxels)))
                z, y, x = nerve_voxels[idx]
                return (int(z), int(y), int(x))
            # Fall through to random if no positive voxels

        if self.training:
            return random_valid_center(shape, self.patch_size, self.rng)

        # Validation / test: deterministic geometric center
        return volume_center(shape)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]

        t1_full = self._load_volume_np(subject_id, self.primary)
        t2_full = self._load_volume_np(subject_id, self.secondary)
        if t1_full.shape != t2_full.shape:
            raise ValueError(
                f"T1/T2 shape mismatch for {subject_id}: "
                f"{t1_full.shape} vs {t2_full.shape}"
            )

        labels_full = self._load_label_np(subject_id, t1_full.shape)
        center = self._sample_center(labels_full)

        t1_patch = extract_patch_3d(t1_full, center, self.patch_size)
        t2_patch = extract_patch_3d(t2_full, center, self.patch_size)
        label_patch = extract_patch_3d(labels_full, center, self.patch_size)

        gt_mask = label_to_multichannel(label_patch, self.n_prompts)

        primary_vol = torch.from_numpy(t1_patch).float().unsqueeze(0)
        secondary_vol = torch.from_numpy(t2_patch).float().unsqueeze(0)
        gt_mask_t = torch.from_numpy(gt_mask).float()

        return primary_vol, secondary_vol, self.text_embeddings, gt_mask_t
