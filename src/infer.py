"""Full-volume inference via MONAI sliding windows with Gaussian blending."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from src.utils.config import resolve_path
from src.data.nifti_io import load_nifti, mask_path, volume_path

RoiSize = Union[Sequence[int], int]


class SlidingWindowPredictor:
    """
    Adapt VoxWhisper's ``(primary, secondary, text)`` forward to MONAI's single-tensor API.

    MONAI feeds overlapping crops of shape ``[B, C, D, H, W]``. We pack primary
    and secondary along the channel axis, split them here, broadcast the cached
    prompt embeddings to the sliding-window batch, and return only the
    full-resolution decoder stage so stitching matches the native primary grid.

    Implemented as a plain callable (not an ``nn.Module``) so wrapping the
    trained model does not re-register its parameters as a submodule.
    """

    def __init__(
        self,
        model: nn.Module,
        text_embeddings: torch.Tensor,
        primary_channels: int = 1,
        secondary_channels: int = 1,
    ):
        self.model = model
        self.primary_channels = int(primary_channels)
        self.secondary_channels = int(secondary_channels)
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
        c1 = self.primary_channels
        primary = inputs[:, :c1]
        secondary = inputs[:, c1:]
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
    """
    Run overlapping 3D crops through ``model`` and blend them onto the native grid.

    Parameters
    ----------
    primary, secondary :
        ``[B, C, D, H, W]`` volumes sharing the same spatial grid (primary is
        the output space).
    text_embeddings :
        Cached prompt tokens, ``[N_T, dim]`` or ``[B, N_T, dim]``.
    roi_size :
        Patch size the model was trained on (typically ``data.patch.size``).
    overlap :
        Fractional window overlap. ``0.5`` is a 50% step; higher overlap is
        smoother and slower.
    mode :
        ``"gaussian"`` down-weights patch borders so seams from missing context
        do not dominate the blended volume.

    Returns
    -------
    logits : torch.Tensor
        Full-resolution prompt logits ``[B, N_T, D, H, W]`` on ``device``.
    """
    if primary.shape[0] != secondary.shape[0] or primary.shape[2:] != secondary.shape[2:]:
        raise ValueError(
            f"Primary/secondary shape mismatch: {tuple(primary.shape)} vs {tuple(secondary.shape)}"
        )
    if primary.ndim != 5:
        raise ValueError(f"Expected primary [B, C, D, H, W], got {tuple(primary.shape)}")

    inputs = torch.cat([primary, secondary], dim=1)
    predictor = SlidingWindowPredictor(
        model,
        text_embeddings,
        primary_channels=primary.shape[1],
        secondary_channels=secondary.shape[1],
    )

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
    """
    Load full-resolution primary and secondary volumes, optional integer mask,
    and the primary affine.

    Returns
    -------
    primary, secondary : float32 arrays ``(D, H, W)``
    labels : int16 array ``(D, H, W)`` or ``None`` if no mask is on disk
    affine : (4, 4) voxel-to-world matrix from the primary NIfTI
    """
    from src.utils.config import active_modality_keys

    primary_key, secondary_key = active_modality_keys(config)
    processed_dir = resolve_path(config, "data.paths.processed")

    primary, affine = load_nifti(volume_path(processed_dir, subject_id, primary_key))
    secondary, _ = load_nifti(volume_path(processed_dir, subject_id, secondary_key))
    if primary.shape != secondary.shape:
        raise ValueError(
            f"Primary/secondary shape mismatch for {subject_id}: "
            f"{primary.shape} vs {secondary.shape}"
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
