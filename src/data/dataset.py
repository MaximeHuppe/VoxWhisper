"""Config-driven dataset: adaptive train patches, frozen val/test patches."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import active_modality_keys, resolve_path
from src.data.nifti_io import (
    as_channel_first,
    extract_patch_3d,
    label_to_multichannel,
    list_subject_ids,
    load_nifti,
    mask_path,
    random_valid_center,
    spatial_shape,
    volume_path,
)
from src.utils.seed import get_training_seed

logger = logging.getLogger(__name__)

Center = Tuple[int, int, int]


class VoxWhisperDataset(Dataset):
    """Patch-based dataset for dual-modality, language-conditioned segmentation.

    Training (``training=True``)
    ----------------------------
    Each ``__getitem__`` call samples one patch on the fly.  The dataset length
    is ``n_subjects * train_patches_per_subject``, so each subject contributes
    that many independently sampled crops per epoch.  Patches are drawn with
    ``positive_ratio`` probability centred on a foreground voxel, otherwise
    uniformly.  The last-loaded subject is cached per worker so consecutive
    draws of the same volume skip a disk read.

    Validation / test (``training=False``)
    ---------------------------------------
    ``val_patches_per_subject`` patch centres are generated once at construction
    (seeded by ``splits.seed``) and never change across epochs.  Consecutive
    calls for the same subject reuse an in-memory cache so each volume is
    loaded only once per epoch rather than once per patch.

    Parameters
    ----------
    config      : loaded YAML config dict (see ``src.utils.config.load_config``).
    subject_ids : explicit list of subject IDs.  Mutually exclusive with
                  ``split``.
    split       : name of a split in the manifest (``"train"``, ``"val"``,
                  ``"test"``).  Used only when ``subject_ids`` is None.
    training    : whether to use training-mode sampling (see above).
    """

    def __init__(
        self,
        config: Mapping,
        subject_ids: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        training: bool = True,
    ) -> None:
        self.config = config
        self.training = training
        self.primary, self.secondary = active_modality_keys(config)
        self.processed_dir = resolve_path(config, "data.paths.processed")

        patch_cfg = config["data"]["patch"]
        self.patch_size = tuple(int(x) for x in patch_cfg["size"])
        self.positive_ratio = float(patch_cfg.get("positive_ratio", 0.5))
        self.positive_labels = [int(x) for x in patch_cfg.get("positive_labels", [1])]
        self.train_patches_per_subject = int(patch_cfg.get("train_patches_per_subject", 1))
        self.val_patches_per_subject = int(patch_cfg.get("val_patches_per_subject", 4))
        if self.train_patches_per_subject < 1:
            raise ValueError("data.patch.train_patches_per_subject must be >= 1")
        self._train_seed = get_training_seed(config) if training else None
        self._rng: Optional[np.random.Generator] = None

        cache_dir = resolve_path(config, "data.paths.cache")
        cache_file = cache_dir / config["text_encoder"]["cache_file"]

        all_subjects = list_subject_ids(self.processed_dir)
        if not all_subjects:
            raise FileNotFoundError(
                f"No processed subjects found in {self.processed_dir}. "
                "Expected layout: {processed}/{subject_id}/t1.nii.gz"
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
                logger.warning(
                    "Skipping %s (missing %s and/or %s)", sid, self.primary, self.secondary
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

        # Frozen val items: list of (subject_id, center) tuples built once at
        # construction so the same patches are used every epoch.
        self._val_items: Optional[List[Tuple[str, Center]]] = None

        # _cache_sid = subject_id
        self._cache_sid: Optional[str] = None
        self._cache_vols = None

        if not training:
            if self.val_patches_per_subject < 1:
                raise ValueError("data.patch.val_patches_per_subject must be >= 1")
            self._val_items = self._build_val_items(config)

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_split_subjects(config: Mapping, split: str) -> List[str]:
        """Load subject IDs for ``split`` from the split manifest file."""
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

    # ------------------------------------------------------------------
    # Volume loading
    # ------------------------------------------------------------------

    def _load_volume_np(self, subject_id: str, modality: str) -> np.ndarray:
        path = volume_path(self.processed_dir, subject_id, modality)
        if not path.exists():
            raise FileNotFoundError(f"Missing volume: {path}")
        data, _ = load_nifti(path)
        if data.ndim not in (3, 4):
            raise ValueError(f"Expected 3D or 4D volume at {path}, got shape {data.shape}")
        return data

    def _load_label_np(self, subject_id: str, spatial_shape: Tuple[int, int, int]) -> np.ndarray:
        """Load the integer mask; returns zeros when no mask file is found."""
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

    def _load_subject(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        primary_full = self._load_volume_np(subject_id, self.primary)
        secondary_full = self._load_volume_np(subject_id, self.secondary)
        primary_spatial = spatial_shape(primary_full)
        if primary_spatial != spatial_shape(secondary_full):
            raise ValueError(
                f"Primary/secondary spatial shape mismatch for {subject_id}: "
                f"{primary_full.shape} vs {secondary_full.shape}"
            )
        labels_full = self._load_label_np(subject_id, primary_spatial)
        return primary_full, secondary_full, labels_full

    def _cached_subject(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cached volumes, reloading only when the subject changes."""
        if self._cache_sid != subject_id:
            self._cache_vols = self._load_subject(subject_id)
            self._cache_sid = subject_id
        return self._cache_vols  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Center sampling
    # ------------------------------------------------------------------

    def _get_rng(self) -> np.random.Generator:
        """Lazy per-worker RNG for reproducible patch sampling with num_workers > 0."""
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            worker_id = 0 if worker is None else worker.id
            self._rng = np.random.default_rng(self._train_seed + worker_id)
        return self._rng

    def _positive_mask(self, labels: np.ndarray) -> np.ndarray:
        mask = np.zeros(labels.shape, dtype=bool)
        for lab in self.positive_labels:
            mask |= labels == lab
        return mask

    def _positive_voxels(self, labels: np.ndarray) -> np.ndarray:
        """Return coordinates of all foreground voxels as shape (N, 3)."""
        return np.argwhere(self._positive_mask(labels))

    def _foreground_centroid(self, voxels: np.ndarray) -> Optional[Center]:
        if len(voxels) == 0:
            return None
        return tuple(int(round(c)) for c in voxels.mean(axis=0))  # type: ignore[return-value]

    def _n_positive_patches(self, n_patches: int) -> int:
        """Number of positive patches out of n_patches, ensuring at least one when ratio > 0."""
        n_pos = int(n_patches * self.positive_ratio)
        if n_patches > 0 and n_pos == 0 and self.positive_ratio > 0:
            n_pos = 1
        return min(n_pos, n_patches)

    def _build_val_items(self, config: Mapping) -> List[Tuple[str, Center]]:
        """Build fixed (subject, center) pairs for validation, seeded deterministically."""
        seed = int(config.get("splits", {}).get("seed", 42))
        rng = np.random.default_rng(seed)
        items: List[Tuple[str, Center]] = []
        for sid in self.subject_ids:
            primary = self._load_volume_np(sid, self.primary)
            labels = self._load_label_np(sid, spatial_shape(primary))
            for center in self._frozen_centers(labels, rng):
                items.append((sid, center))
        return items

    def _frozen_centers(
        self, labels: np.ndarray, rng: np.random.Generator
    ) -> List[Center]:
        """Sample ``val_patches_per_subject`` fixed centers with the positive ratio.

        The first positive center is always the foreground centroid (most
        representative), remaining positive centers are sampled from foreground
        voxels, and negative centers are sampled uniformly.
        """
        n_patches = self.val_patches_per_subject
        n_pos = self._n_positive_patches(n_patches)
        n_neg = n_patches - n_pos
        voxels = self._positive_voxels(labels)
        centers: List[Center] = []

        if n_pos > 0 and len(voxels) > 0:
            centers.append(self._foreground_centroid(voxels))  # type: ignore[arg-type]
            remaining = n_pos - 1
            if remaining > 0:
                replace = remaining > len(voxels)
                chosen = rng.choice(len(voxels), size=remaining, replace=replace)
                for i in np.atleast_1d(chosen):
                    z, y, x = voxels[int(i)]
                    centers.append((int(z), int(y), int(x)))
        else:
            for _ in range(n_pos):
                centers.append(random_valid_center(labels.shape, self.patch_size, rng))

        for _ in range(n_neg):
            centers.append(random_valid_center(labels.shape, self.patch_size, rng))
        return centers

    def _sample_training_center(self, labels: np.ndarray) -> Center:
        """Sample one patch center with the configured positive/negative ratio."""
        rng = self._get_rng()
        if rng.random() < self.positive_ratio:
            positive_voxels = self._positive_voxels(labels)
            if len(positive_voxels) > 0:
                idx = int(rng.integers(0, len(positive_voxels)))
                z, y, x = positive_voxels[idx]
                return (int(z), int(y), int(x))
        return random_valid_center(labels.shape, self.patch_size, rng)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._val_items is not None:
            return len(self._val_items)
        return len(self.subject_ids) * self.train_patches_per_subject

    def __getitem__(self, idx: int):
        if self.training:
            subject_id = self.subject_ids[idx // self.train_patches_per_subject]
            primary_full, secondary_full, labels_full = self._cached_subject(subject_id)
            center = self._sample_training_center(labels_full)
        else:
            subject_id, center = self._val_items[idx]  # type: ignore[index]
            primary_full, secondary_full, labels_full = self._cached_subject(subject_id)

        primary_patch = extract_patch_3d(primary_full, center, self.patch_size)
        secondary_patch = extract_patch_3d(secondary_full, center, self.patch_size)
        label_patch = extract_patch_3d(labels_full, center, self.patch_size)

        gt_mask = label_to_multichannel(label_patch, self.n_prompts)

        primary_vol = torch.from_numpy(as_channel_first(primary_patch)).float()
        secondary_vol = torch.from_numpy(as_channel_first(secondary_patch)).float()
        gt_mask_t = torch.from_numpy(gt_mask).float()

        return primary_vol, secondary_vol, self.text_embeddings, gt_mask_t
