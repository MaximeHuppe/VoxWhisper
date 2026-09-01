# src/dataset.py
"""Config-driven dataset: adaptive train patches, frozen val/test patches."""
from __future__ import annotations
import logging

import json
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import active_modality_keys, resolve_path
from src.utils.seed import get_training_seed
from src.utils.nifti_io import (
    extract_patch_3d,
    label_to_multichannel,
    list_subject_ids,
    load_nifti,
    mask_path,
    random_valid_center,
    volume_path,
)

logger = logging.getLogger(__name__)

Center = Tuple[int, int, int]


class VoxWhisperDataset(Dataset):
    """
    Loads full-resolution primary/secondary + mask NIfTIs and returns fixed-size patches.

    Training (``training=True``):
      - One adaptive 50/50 crop per subject, resampled every ``__getitem__``
      - Patch sampling uses ``training.seed`` (per DataLoader worker when applicable)

    Validation / test (``training=False``):
      - ``val_patches_per_subject`` frozen crops per subject (same 50/50 mix)
      - Centers are seeded by ``splits.seed`` and never change across epochs
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
        self.val_patches_per_subject = int(patch_cfg.get("val_patches_per_subject", 4))
        self._train_seed = get_training_seed(config) if training else None
        self._rng: Optional[np.random.Generator] = None

        cache_dir = resolve_path(config, "data.paths.cache")
        cache_file = cache_dir / config["text_encoder"]["cache_file"]

        # 1. List all subjects in the processed directory [processed/{subject_id}]
        # eg : ["802844", "802845", ...]
        all_subjects = list_subject_ids(self.processed_dir) 
        if not all_subjects:
            raise FileNotFoundError(
                f"No processed subjects found in {self.processed_dir}. "
                "Expected layout: {{processed}}/{{subject_id}}/t1.nii.gz"
            )

        # 2. If no subject IDs are provided, load the split subjects
        if subject_ids is None and split is not None:
            subject_ids = self._load_split_subjects(config, split)

        # 3. If subject IDs are provided, filter the subjects
        if subject_ids is not None:
            allowed = set(subject_ids)
            all_subjects = [s for s in all_subjects if s in allowed]

        # 4. Filter the subjects to only include those with valid volumes
        # valid = presence of both volumes (primary and secondary)
        valid = []
        for sid in all_subjects:
            p1 = volume_path(self.processed_dir, sid, self.primary)
            p2 = volume_path(self.processed_dir, sid, self.secondary)
            if p1.exists() and p2.exists():
                valid.append(sid)
            else:
                logger.warning("Skipping %s (missing %s and/or %s)", sid, self.primary, self.secondary,)

        self.subject_ids = valid
        if not self.subject_ids:
            raise ValueError("Dataset is empty after subject filtering")

        # 5. Load the text embeddings
        if not cache_file.exists():
            raise FileNotFoundError(f"Text embedding cache not found: {cache_file}")

        # 5.1. Load the text embeddings
        try:
            embeddings = torch.load(cache_file, map_location="cpu", weights_only=True)
        except TypeError:
            embeddings = torch.load(cache_file, map_location="cpu")

        # 5.2. Squeeze the embeddings if they are 3D
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(0)
        self.text_embeddings = embeddings
        self.n_prompts = embeddings.shape[0]

        # 5.3. Check if the number of prompts is expected
        expected_prompts = len(config["data"]["prompts"])
        if self.n_prompts != expected_prompts:
            raise ValueError(
                f"Cached embeddings have {self.n_prompts} prompts, "
                f"but config lists {expected_prompts}"
            )

        # 6. Initialize the validation items
        # _val_items = [(subject_id, center)]
        self._val_items: Optional[List[Tuple[str, Center]]] = None

        # _cache_sid = subject_id
        self._cache_sid: Optional[str] = None

        # _cache_vols = (primary_full, secondary_full, labels_full)
        self._cache_vols = None

        # If not training, build the validation items
        if not training:
            if self.val_patches_per_subject < 1:
                raise ValueError("data.patch.val_patches_per_subject must be >= 1")
            self._val_items = self._build_val_items(config)


    ########################################################
    #               SUBJECT LOADING FUNCTIONS              #
    ########################################################
    @staticmethod
    def _load_split_subjects(config: Mapping, split: str) -> list[str]:
        """
        Load the split subjects from the manifest file.
        
        Steps:
        1. Get the manifest path
        2. Check if the manifest path is absolute
        3. If not, get the project root
        4. Check if the manifest path exists
        5. Load the manifest
        6. Check if the split is in the manifest
        7. Return the list of subjects
        """

        # 1. Get the manifest path
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

    ########################################################
    #               I/O HELPER FUNCTIONS                   #
    ########################################################

    def _load_volume_np(self, subject_id: str, modality: str) -> np.ndarray:
        """
        Load a volume from the processed directory.
        
        Steps:
        1. Get the volume path
        2. Check if the volume path exists
        3. Load the volume
        4. Check if the volume is 3D
        5. Return the volume
        """
        path = volume_path(self.processed_dir, subject_id, modality)
        if not path.exists():
            raise FileNotFoundError(f"Missing volume: {path}")
        data, _ = load_nifti(path)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume at {path}, got shape {data.shape}")
        return data

    def _load_label_np(self, subject_id: str, spatial_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Load a label volume from the processed directory.
        
        Steps:
        1. Get the label path
        2. Check if the label path exists
        3. If not, return a zero array
        4. Load the label volume
        5. Check if the label volume is 3D
        6. Check if the label volume shape matches the spatial shape
        7. Return the label volume
        """

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

    def _load_subject(self, subject_id: str):
        """
        Load a subject from the processed directory.
        
        Input:
        --------------
        - subject_id: str
        --------------
        Output:
        --------------
        - primary_full: np.ndarray
        - secondary_full: np.ndarray
        - labels_full: np.ndarray
        --------------
        Steps:
        ---------------
        1. Load the primary volume
        2. Load the secondary volume
        3. Check if the primary and secondary volumes have the same shape
        4. Load the label volume
        5. Return the primary, secondary, and label volumes
        """

        primary_full = self._load_volume_np(subject_id, self.primary)
        secondary_full = self._load_volume_np(subject_id, self.secondary)
        if primary_full.shape != secondary_full.shape:
            raise ValueError(
                f"Primary/secondary shape mismatch for {subject_id}: "
                f"{primary_full.shape} vs {secondary_full.shape}"
            )
        labels_full = self._load_label_np(subject_id, primary_full.shape)
        return primary_full, secondary_full, labels_full

    def _cached_subject(self, subject_id: str):
        """
        Load a subject from the processed directory and cache it.
        
        Steps:
        1. Check if the subject ID is different from the cached subject ID
        2. If it is, load the subject and cache it
        3. Return the cached subject
        """
        if self._cache_sid != subject_id:
            self._cache_vols = self._load_subject(subject_id)
            self._cache_sid = subject_id
        return self._cache_vols

    ########################################################
    #               CENTER SAMPLING FUNCTIONS              #
    ########################################################

    def _get_rng(self) -> np.random.Generator:
        """Lazy per-worker RNG so patch sampling is reproducible with ``num_workers > 0``."""
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            worker_id = 0 if worker is None else worker.id
            self._rng = np.random.default_rng(self._train_seed + worker_id)
        return self._rng

    def _positive_mask(self, labels: np.ndarray) -> np.ndarray:
        pos_mask = np.zeros(labels.shape, dtype=bool)
        for lab in self.positive_labels:
            pos_mask |= labels == lab
        return pos_mask

    def _positive_voxels(self, labels: np.ndarray) -> np.ndarray:
        """
        Get the voxels that are positive for the positive labels.
        
        Steps:
        1. Get the positive mask
        2. Get the voxels that are True
        3. Return the voxels that are True
        """
        return np.argwhere(self._positive_mask(labels))

    def _foreground_centroid(self, voxels: np.ndarray) -> Optional[Center]:
        if len(voxels) == 0:
            return None
        return tuple(int(round(c)) for c in voxels.mean(axis=0))

    def _n_positive_patches(self, n_patches: int) -> int:
        """
        Get the number of positive patches.
        
        Steps:
        1. Calculate the number of positive patches
        2. If there are no positive patches and the positive ratio is greater than 0, set the number of positive patches to 1
        3. Return the minimum of the number of positive patches and the number of patches
        """
        n_pos = int(n_patches * self.positive_ratio)
        if n_patches > 0 and n_pos == 0 and self.positive_ratio > 0:
            n_pos = 1
        return min(n_pos, n_patches)

    def _build_val_items(self, config: Mapping) -> List[Tuple[str, Center]]:
        """
        Build the validation items.
        
        Steps:
        1. Get the seed
        2. Create a random number generator
        3. Create a list of validation items
        4. For each subject, load the labels
        5. For each center, add the subject and center to the list
        6. Return the list of validation items
        """

        seed = int(config.get("splits", {}).get("seed", 42))
        rng = np.random.default_rng(seed)
        items: List[Tuple[str, Center]] = []
        for sid in self.subject_ids:
            primary = self._load_volume_np(sid, self.primary)
            labels = self._load_label_np(sid, primary.shape)
            for center in self._frozen_centers(labels, rng):
                items.append((sid, center))
        return items

    def _frozen_centers(self, labels: np.ndarray, rng: np.random.Generator) -> List[Center]:
        """
        Get the frozen centers.
        
        Steps:
        1. Get the number of patches
        2. Get the number of positive patches
        3. Get the number of negative patches
        4. Get the voxels that are positive
        5. Create a list of centers
        6. If there are positive patches and voxels, get the centroid
        7. If the centroid is not None, add it to the list
        8. If there are remaining positive patches, replace the voxels with the remaining positive patches
        9. If there are remaining negative patches, add random valid centers
        10. Return the list of centers
        """

        n_patches = self.val_patches_per_subject
        n_pos = self._n_positive_patches(n_patches)
        n_neg = n_patches - n_pos
        voxels = self._positive_voxels(labels)
        centers: List[Center] = []

        if n_pos > 0 and len(voxels) > 0:
            centers.append(self._foreground_centroid(voxels))
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
        """
        Sample a training center.
        
        Steps:
        1. If the random number is less than the positive ratio, get the voxels that are positive
        2. If there are positive voxels, sample a random voxel
        3. If there are no positive voxels, sample a random valid center
        4. Return the center
        """
        rng = self._get_rng()
        if rng.random() < self.positive_ratio:
            positive_voxels = self._positive_voxels(labels)
            if len(positive_voxels) > 0:
                idx = int(rng.integers(0, len(positive_voxels)))
                z, y, x = positive_voxels[idx]
                return (int(z), int(y), int(x))

        return random_valid_center(labels.shape, self.patch_size, rng)


    ########################################################
    #                      API FUNCTIONS                   #
    ########################################################

    def __len__(self):
        """
        Return the number of items in the dataset.
        
        Steps:
        1. If validation items are provided, return the number of validation items
        2. Otherwise, return the number of subjects
        """
        if self._val_items is not None:
            return len(self._val_items)
        return len(self.subject_ids)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Steps:
        1. If training, sample a training center
        2. Otherwise, get the validation item
        3. Extract the patch from the primary, secondary, and label volumes
        4. Convert the label patch to a multi-channel mask
        5. Return the primary, secondary, text embeddings, and label mask
        """
        if self.training:
            subject_id = self.subject_ids[idx]
            primary_full, secondary_full, labels_full = self._load_subject(subject_id)
            center = self._sample_training_center(labels_full)
        else:
            subject_id, center = self._val_items[idx]
            primary_full, secondary_full, labels_full = self._cached_subject(subject_id)

        primary_patch = extract_patch_3d(primary_full, center, self.patch_size)
        secondary_patch = extract_patch_3d(secondary_full, center, self.patch_size)
        label_patch = extract_patch_3d(labels_full, center, self.patch_size)

        gt_mask = label_to_multichannel(label_patch, self.n_prompts)

        primary_vol = torch.from_numpy(primary_patch).float().unsqueeze(0)
        secondary_vol = torch.from_numpy(secondary_patch).float().unsqueeze(0)
        gt_mask_t = torch.from_numpy(gt_mask).float()

        return primary_vol, secondary_vol, self.text_embeddings, gt_mask_t
