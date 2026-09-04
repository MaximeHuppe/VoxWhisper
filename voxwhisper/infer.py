"""Full-volume inference via MONAI sliding windows with Gaussian blending."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from voxwhisper.util.config import resolve_path
from voxwhisper.data.nifti_io import load_nifti, mask_path, volume_path

RoiSize = Union[Sequence[int], int]

# Fixed modalities for this branch.
_PRIMARY_CHANNELS = 1
_SECONDARY_CHANNELS = 1


class SlidingWindowPredictorDense:
    """Adapt VoxDense's ``(volume, text)`` forward to MONAI's single-tensor API."""

    def __init__(self, model: nn.Module, text_embeddings: torch.Tensor) -> None:
        self.model = model
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings[0]
        if text_embeddings.dim() != 2:
            raise ValueError(
                "text_embeddings must be [N_T, dim] or [B, N_T, dim], "
                f"got shape {tuple(text_embeddings.shape)}"
            )
        self.text_embeddings = text_embeddings.detach().contiguous()

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError(
                f"Expected T1 input [B, 1, D, H, W], got {tuple(inputs.shape)}"
            )
        text = self.text_embeddings.to(device=inputs.device, dtype=inputs.dtype)
        text = text.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        predictions = self.model(inputs, text)
        if isinstance(predictions, (list, tuple)):
            return predictions[-1]
        return predictions


def predict_dense_volume(
    model: nn.Module,
    volume: torch.Tensor,
    text_embeddings: torch.Tensor,
    roi_size: RoiSize = (128, 128, 128),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    mode: str = "gaussian",
    sigma_scale: float = 0.125,
    padding_mode: str = "constant",
    progress: bool = False,
    sw_device: Optional[torch.device] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sliding-window inference for a T1-only ``VoxDense`` model."""
    if volume.ndim != 5:
        raise ValueError(f"Expected volume [B, C, D, H, W], got {tuple(volume.shape)}")

    predictor = SlidingWindowPredictorDense(model, text_embeddings)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return sliding_window_inference(
                inputs=volume,
                roi_size=tuple(int(x) for x in roi_size),
                sw_batch_size=int(sw_batch_size),
                predictor=predictor,
                overlap=float(overlap),
                mode=mode,
                sigma_scale=sigma_scale,
                padding_mode=padding_mode,
                progress=progress,
                sw_device=sw_device,
                device=device,
            )
    finally:
        if was_training:
            model.train()


class SlidingWindowPredictor:
    """Adapt VoxWhisper's ``(primary, secondary, text)`` forward to MONAI's single-tensor API.

    MONAI feeds overlapping crops of shape ``[B, C, D, H, W]``.  We pack
    primary and secondary along the channel axis (both are 1-channel for T1+FA),
    split them here, broadcast the cached prompt embeddings to the batch, and
    return only the full-resolution decoder stage for stitching.
    """

    def __init__(self, model: nn.Module, text_embeddings: torch.Tensor) -> None:
        self.model = model
        self.primary_channels = _PRIMARY_CHANNELS
        self.secondary_channels = _SECONDARY_CHANNELS
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings[0]
        if text_embeddings.dim() != 2:
            raise ValueError(
                "text_embeddings must be [N_T, dim] or [B, N_T, dim], "
                f"got shape {tuple(text_embeddings.shape)}"
            )
        self.text_embeddings = text_embeddings.detach().contiguous()

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        expected_c = self.primary_channels + self.secondary_channels
        if inputs.ndim != 5 or inputs.shape[1] != expected_c:
            raise ValueError(
                "Expected concatenated primary/secondary input "
                f"[B, {expected_c}, D, H, W], got {tuple(inputs.shape)}"
            )
        primary = inputs[:, : self.primary_channels]
        secondary = inputs[:, self.primary_channels :]
        text = self.text_embeddings.to(device=inputs.device, dtype=inputs.dtype)
        text = text.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        predictions = self.model(primary, secondary, text)
        if isinstance(predictions, (list, tuple)):
            return predictions[-1]
        return predictions


def predict_full_volume(
    model: nn.Module,
    primary: torch.Tensor,
    secondary: torch.Tensor,
    text_embeddings: torch.Tensor,
    roi_size: RoiSize = (128, 128, 128),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    mode: str = "gaussian",
    sigma_scale: float = 0.125,
    padding_mode: str = "constant",
    progress: bool = False,
    sw_device: Optional[torch.device] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run overlapping 3D crops through ``model`` and blend onto the native grid.

    Parameters
    ----------
    primary, secondary : ``[B, 1, D, H, W]`` volumes on the same grid.
    text_embeddings    : ``[N_T, dim]`` or ``[B, N_T, dim]`` cached embeddings.
    roi_size           : patch size the model was trained on.

    Returns
    -------
    logits : ``[B, N_T, D, H, W]`` full-resolution logits.
    """
    if primary.shape[0] != secondary.shape[0] or primary.shape[2:] != secondary.shape[2:]:
        raise ValueError(
            f"Primary/secondary shape mismatch: {tuple(primary.shape)} vs {tuple(secondary.shape)}"
        )
    if primary.ndim != 5:
        raise ValueError(f"Expected primary [B, C, D, H, W], got {tuple(primary.shape)}")

    inputs = torch.cat([primary, secondary], dim=1)
    predictor = SlidingWindowPredictor(model, text_embeddings)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return sliding_window_inference(
                inputs=inputs,
                roi_size=tuple(int(x) for x in roi_size),
                sw_batch_size=int(sw_batch_size),
                predictor=predictor,
                overlap=float(overlap),
                mode=mode,
                sigma_scale=sigma_scale,
                padding_mode=padding_mode,
                progress=progress,
                sw_device=sw_device,
                device=device,
            )
    finally:
        if was_training:
            model.train()


def load_text_embeddings(config: Mapping, map_location: str = "cpu") -> torch.Tensor:
    cache_dir = resolve_path(config, "data.paths.cache")
    cache_file = cache_dir / config["text_encoder"]["cache_file"]
    if not cache_file.exists():
        raise FileNotFoundError(f"Text embedding cache not found: {cache_file}")
    try:
        embeddings = torch.load(cache_file, map_location=map_location, weights_only=True)
    except TypeError:
        embeddings = torch.load(cache_file, map_location=map_location)
    if embeddings.ndim == 3:
        embeddings = embeddings.squeeze(0)
    return embeddings


def load_subject_for_inference(
    config: Mapping,
    subject_id: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Load full-resolution T1, FA, optional mask, and T1 affine.

    Returns
    -------
    primary, secondary : float32 arrays ``(D, H, W)``
    labels             : int16 array ``(D, H, W)`` or ``None``
    affine             : (4, 4) voxel-to-world matrix
    """
    processed_dir = resolve_path(config, "data.paths.processed")
    primary, affine = load_nifti(volume_path(processed_dir, subject_id, "t1"))
    secondary, _ = load_nifti(volume_path(processed_dir, subject_id, "fa"))
    if primary.shape != secondary.shape:
        raise ValueError(
            f"T1/FA shape mismatch for {subject_id}: {primary.shape} vs {secondary.shape}"
        )

    labels: Optional[np.ndarray] = None
    mpath = mask_path(processed_dir, subject_id)
    if mpath.exists():
        labels_f, _ = load_nifti(mpath)
        if labels_f.ndim != 3:
            raise ValueError(f"Expected 3D mask at {mpath}, got {labels_f.shape}")
        if labels_f.shape != primary.shape:
            raise ValueError(
                f"Mask shape {labels_f.shape} != primary shape {primary.shape} "
                f"for {subject_id}"
            )
        labels = np.rint(labels_f).astype(np.int16)

    return primary, secondary, labels, affine


def load_dense_subject_for_inference(
    config: Mapping,
    subject_id: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Load full-resolution T1, optional mask, and T1 affine."""
    processed_dir = resolve_path(config, "data.paths.processed")
    volume, affine = load_nifti(volume_path(processed_dir, subject_id, "t1"))

    labels: Optional[np.ndarray] = None
    mpath = mask_path(processed_dir, subject_id)
    if mpath.exists():
        labels_f, _ = load_nifti(mpath)
        if labels_f.ndim != 3:
            raise ValueError(f"Expected 3D mask at {mpath}, got {labels_f.shape}")
        if labels_f.shape != volume.shape:
            raise ValueError(
                f"Mask shape {labels_f.shape} != T1 shape {volume.shape} for {subject_id}"
            )
        labels = np.rint(labels_f).astype(np.int16)

    return volume, labels, affine


def volume_to_tensor(volume: np.ndarray) -> torch.Tensor:
    """``(D, H, W)`` NumPy → ``[1, 1, D, H, W]`` float tensor."""
    return torch.from_numpy(np.ascontiguousarray(volume)).float().unsqueeze(0).unsqueeze(0)


def volumes_to_tensors(
    primary: np.ndarray, secondary: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(D, H, W)`` NumPy → ``[1, 1, D, H, W]`` float tensors."""
    primary_t = torch.from_numpy(np.ascontiguousarray(primary)).float().unsqueeze(0).unsqueeze(0)
    secondary_t = torch.from_numpy(np.ascontiguousarray(secondary)).float().unsqueeze(0).unsqueeze(0)
    return primary_t, secondary_t


def logits_to_label_map(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over prompt channels → integer labels ``[B, D, H, W]``."""
    return logits.argmax(dim=1)
