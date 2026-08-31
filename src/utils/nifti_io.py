"""Shared NIfTI I/O and geometry helpers for preprocessing and training."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np

PathLike = Union[str, Path]


def center_crop_or_pad_3d(
    volume: np.ndarray,
    target_shape: Sequence[int] = (128, 128, 128),
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Center crop or pad a 3D array to ``target_shape``.

    Returns
    -------
    output : np.ndarray
        Cropped/padded volume.
    offset : (oz, oy, ox)
        Voxel offset of the output origin in the *input* grid
        (negative when padding). Used to update NIfTI affines.
    """
    spatial_shape = volume.shape
    target_shape = tuple(int(x) for x in target_shape)
    output = np.zeros(target_shape, dtype=volume.dtype)

    slices_in = []
    slices_out = []
    offset = []

    for i in range(3):
        if spatial_shape[i] >= target_shape[i]:
            start = (spatial_shape[i] - target_shape[i]) // 2
            slices_in.append(slice(start, start + target_shape[i]))
            slices_out.append(slice(0, target_shape[i]))
            offset.append(start)
        else:
            start = (target_shape[i] - spatial_shape[i]) // 2
            slices_in.append(slice(0, spatial_shape[i]))
            slices_out.append(slice(start, start + spatial_shape[i]))
            offset.append(-start)

    output[tuple(slices_out)] = volume[tuple(slices_in)]
    return output, (offset[0], offset[1], offset[2])


def adjust_affine_for_crop(
    affine: np.ndarray,
    offset: Tuple[int, int, int],
) -> np.ndarray:
    """Shift affine origin so cropped voxels keep world coordinates."""
    new_affine = affine.copy()
    voxel_offset = np.array(offset, dtype=np.float64)
    new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ voxel_offset
    return new_affine


def identity_affine(spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Simple RAS-like identity affine with given voxel spacing."""
    affine = np.eye(4, dtype=np.float64)
    for i, s in enumerate(spacing):
        affine[i, i] = float(s)
    return affine


def save_nifti(
    data: np.ndarray,
    path: PathLike,
    affine: Optional[np.ndarray] = None,
    dtype: Optional[np.dtype] = None,
) -> Path:
    """
    Save a NumPy array as ``.nii.gz``.

    ``data`` may be 3D ``(D, H, W)`` or 4D ``(D, H, W, C)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = identity_affine()
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    img = nib.Nifti1Image(arr, affine)
    nib.save(img, str(path))
    return path


def load_nifti(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file; returns ``(data, affine)`` as float32 data."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def normalize_intensity(
    volume: np.ndarray,
    method: str = "zscore",
    nonzero_only: bool = True,
) -> np.ndarray:
    """
    Normalize a 3D volume.

    For ``zscore`` with ``nonzero_only=True``, mean/std are computed on
    voxels with absolute value > 0 (simple foreground / non-air mask).
    """
    volume = volume.astype(np.float32, copy=False)

    if method == "minmax":
        if nonzero_only:
            mask = np.abs(volume) > 0
            if not np.any(mask):
                return volume
            vol_min = float(volume[mask].min())
            vol_max = float(volume[mask].max())
        else:
            vol_min = float(volume.min())
            vol_max = float(volume.max())
        if vol_max - vol_min > 0:
            return (volume - vol_min) / (vol_max - vol_min)
        return volume

    if method == "zscore":
        if nonzero_only:
            mask = np.abs(volume) > 0
            if not np.any(mask):
                return volume
            mean = float(volume[mask].mean())
            std = float(volume[mask].std())
        else:
            mean = float(volume.mean())
            std = float(volume.std())
        if std > 0:
            return (volume - mean) / std
        return volume - mean

    raise ValueError(f"Unknown normalization method: {method}")


def label_to_multichannel(label_volume: np.ndarray, n_channels: int) -> np.ndarray:
    """Integer label map → float multi-channel one-hot ``[N_T, D, H, W]``."""
    channels = [(label_volume == k).astype(np.float32) for k in range(n_channels)]
    return np.stack(channels, axis=0)


def subject_processed_dir(processed_root: PathLike, subject_id: str) -> Path:
    return Path(processed_root) / subject_id


def volume_path(processed_root: PathLike, subject_id: str, modality: str) -> Path:
    return subject_processed_dir(processed_root, subject_id) / f"{modality}.nii.gz"


def mask_path(processed_root: PathLike, subject_id: str) -> Path:
    return subject_processed_dir(processed_root, subject_id) / "mask.nii.gz"


def extract_patch_3d(
    volume: np.ndarray,
    center: Sequence[int],
    patch_size: Sequence[int],
) -> np.ndarray:
    """
    Extract a fixed-size 3D patch centered at ``center``, zero-padding at borders.

    Parameters
    ----------
    volume : np.ndarray
        3D array ``(D, H, W)``.
    center : (z, y, x)
        Patch center in volume coordinates.
    patch_size : (pz, py, px)
        Output spatial size.

    Returns
    -------
    patch : np.ndarray
        Array of shape ``patch_size`` (same dtype as ``volume``).
    """
    if volume.ndim != 3:
        raise ValueError(f"extract_patch_3d expects 3D volume, got shape {volume.shape}")

    patch_size = tuple(int(x) for x in patch_size)
    center = [int(c) for c in center]
    output = np.zeros(patch_size, dtype=volume.dtype)

    src_slices = []
    dst_slices = []
    for i in range(3):
        half = patch_size[i] // 2
        src_start = center[i] - half
        src_end = src_start + patch_size[i]

        dst_start = max(0, -src_start)
        dst_end = patch_size[i] - max(0, src_end - volume.shape[i])
        src_start_clipped = max(0, src_start)
        src_end_clipped = min(volume.shape[i], src_end)

        src_slices.append(slice(src_start_clipped, src_end_clipped))
        dst_slices.append(slice(dst_start, dst_end))

    if all(s.start < s.stop for s in src_slices) and all(
        s.start < s.stop for s in dst_slices
    ):
        output[tuple(dst_slices)] = volume[tuple(src_slices)]
    return output


def random_valid_center(
    volume_shape: Sequence[int],
    patch_size: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[int, int, int]:
    """
    Sample a random center so the patch lies fully inside the volume when possible.

    If the volume is smaller than the patch along an axis, the center is clamped
    to the middle of that axis (patch will be padded).
    """
    center = []
    for i in range(3):
        half = int(patch_size[i]) // 2
        lo = half
        hi = int(volume_shape[i]) - (int(patch_size[i]) - half)
        if hi <= lo:
            center.append(int(volume_shape[i]) // 2)
        else:
            center.append(int(rng.integers(lo, hi)))
    return (center[0], center[1], center[2])


def volume_center(volume_shape: Sequence[int]) -> Tuple[int, int, int]:
    """Geometric center of a 3D volume."""
    return tuple(int(s) // 2 for s in volume_shape)


def list_subject_ids(processed_root: PathLike) -> list[str]:
    """
    List subject IDs under processed root.

    Supports:
    - New layout: ``{processed}/{subject_id}/t1.nii.gz``
    - Legacy NPZ: ``{processed}/{subject_id}_preprocessed.npz``
    """
    root = Path(processed_root)
    if not root.exists():
        return []

    subjects = set()
    for child in root.iterdir():
        if child.is_dir():
            if any(child.glob("*.nii.gz")) or any(child.glob("*.nii")):
                subjects.add(child.name)
        elif child.name.endswith("_preprocessed.npz"):
            subjects.add(child.name.split("_")[0])
    return sorted(subjects)
