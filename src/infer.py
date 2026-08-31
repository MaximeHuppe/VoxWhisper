"""Full-volume inference via MONAI sliding windows with Gaussian blending."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from src.utils.config import resolve_path
from src.utils.nifti_io import load_nifti, mask_path, volume_path

RoiSize = Union[Sequence[int], int]


class SlidingWindowPredictor:
    """
    Adapt VoxWhisper's ``(t1, t2, text)`` forward to MONAI's single-tensor API.

    MONAI feeds overlapping crops of shape ``[B, C, D, H, W]``. We pack T1 and
    T2 as two channels, split them here, broadcast the cached prompt embeddings
    to the sliding-window batch, and return only the full-resolution decoder
    stage so stitching matches the native T1 grid.

    Implemented as a plain callable (not an ``nn.Module``) so wrapping the
    trained model does not re-register its parameters as a submodule.
    """

    def __init__(self, model: nn.Module, text_embeddings: torch.Tensor):
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
        if inputs.ndim != 5 or inputs.shape[1] != 2:
            raise ValueError(
                "Expected concatenated T1/T2 input [B, 2, D, H, W], "
                f"got {tuple(inputs.shape)}"
            )
        t1 = inputs[:, 0:1]
        t2 = inputs[:, 1:2]
        text = self.text_embeddings.to(device=inputs.device, dtype=inputs.dtype)
        text = text.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        predictions = self.model(t1, t2, text)
        if isinstance(predictions, (list, tuple)):
            return predictions[-1]
        return predictions


def predict_full_volume(
    model: nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
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
    t1, t2 :
        ``[B, 1, D, H, W]`` volumes in the same voxel grid (T1 is the output space).
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
    if t1.shape != t2.shape:
        raise ValueError(
            f"T1/T2 shape mismatch: {tuple(t1.shape)} vs {tuple(t2.shape)}"
        )
    if t1.ndim != 5 or t1.shape[1] != 1:
        raise ValueError(f"Expected T1 [B, 1, D, H, W], got {tuple(t1.shape)}")

    inputs = torch.cat([t1, t2], dim=1)
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
    """
    Load full-resolution T1, T2, optional integer mask, and the T1 affine.

    Returns
    -------
    t1, t2 : float32 arrays ``(D, H, W)``
    labels : int16 array ``(D, H, W)`` or ``None`` if no mask is on disk
    affine : (4, 4) voxel-to-world matrix from the T1 NIfTI
    """
    from src.utils.config import active_modality_keys

    primary, secondary = active_modality_keys(config)
    processed_dir = resolve_path(config, "data.paths.processed")

    t1, affine = load_nifti(volume_path(processed_dir, subject_id, primary))
    t2, _ = load_nifti(volume_path(processed_dir, subject_id, secondary))
    if t1.shape != t2.shape:
        raise ValueError(
            f"T1/T2 shape mismatch for {subject_id}: {t1.shape} vs {t2.shape}"
        )

    labels: Optional[np.ndarray] = None
    mpath = mask_path(processed_dir, subject_id)
    if mpath.exists():
        labels_f, _ = load_nifti(mpath)
        if labels_f.ndim != 3:
            raise ValueError(f"Expected 3D mask at {mpath}, got {labels_f.shape}")
        if labels_f.shape != t1.shape:
            raise ValueError(
                f"Mask shape {labels_f.shape} != T1 shape {t1.shape} for {subject_id}"
            )
        labels = np.rint(labels_f).astype(np.int16)

    return t1, t2, labels, affine


def volumes_to_tensors(
    t1: np.ndarray, t2: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(D, H, W)`` NumPy → ``[1, 1, D, H, W]`` float tensors."""
    t1_t = torch.from_numpy(np.ascontiguousarray(t1)).float().unsqueeze(0).unsqueeze(0)
    t2_t = torch.from_numpy(np.ascontiguousarray(t2)).float().unsqueeze(0).unsqueeze(0)
    return t1_t, t2_t


def logits_to_label_map(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over prompt channels → integer labels ``[B, D, H, W]``."""
    return logits.argmax(dim=1)
