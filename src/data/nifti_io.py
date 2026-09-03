"""Shared NIfTI I/O and geometry helpers for preprocessing and training."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np

PathLike = Union[str, Path]


########################################################
#          VOLUME I/O AND PROCESSING HELPERS          #
########################################################

def load_nifti(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file; returns ``(data, affine)`` as float32 data."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def spatial_shape(volume: np.ndarray) -> Tuple[int, int, int]:
    """Return ``(D, H, W)`` for a 3D volume or a 4D ``(D, H, W, C)`` volume."""
    if volume.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")
    d, h, w = volume.shape[:3]
    return int(d), int(h), int(w)


def as_channel_first(volume: np.ndarray) -> np.ndarray:
    """Convert ``(D, H, W)`` or ``(D, H, W, C)`` to contiguous ``(C, D, H, W)``."""
    if volume.ndim == 3:
        return np.ascontiguousarray(volume[np.newaxis, ...])
    if volume.ndim == 4:
        return np.ascontiguousarray(np.moveaxis(volume, -1, 0))
    raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")


def normalize_intensity(
    volume: np.ndarray,
    method: str = "zscore",
    nonzero_only: bool = True,
) -> np.ndarray:
    """
    Normalize a 3D volume, or a 4D volume channel-wise.

    For ``zscore`` with ``nonzero_only=True``, mean/std are computed on
    voxels with absolute value > 0 (simple foreground / non-air mask).
    4D ``(D, H, W, C)`` inputs (e.g. DEC-FA) are normalized independently
    per channel.  Prefer ``normalize: false`` on RGB volumes so the
    directional colour encoding is preserved.
    """
    volume = volume.astype(np.float32, copy=False)
    if volume.ndim == 4:
        channels = [
            normalize_intensity(volume[..., c], method=method, nonzero_only=nonzero_only)
            for c in range(volume.shape[-1])
        ]
        return np.stack(channels, axis=-1)

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


def identity_affine(spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Simple RAS-like identity affine with given voxel spacing."""
    affine = np.eye(4, dtype=np.float64)
    for i, s in enumerate(spacing):
        affine[i, i] = float(s)
    return affine


def label_to_multichannel(label_volume: np.ndarray, n_channels: int) -> np.ndarray:
    """Integer label map → float multi-channel one-hot ``[N_T, D, H, W]``."""
    channels = [(label_volume == k).astype(np.float32) for k in range(n_channels)]
    return np.stack(channels, axis=0)


########################################################
#          SUBJECT PROCESSING AND PATH HELPERS         #
########################################################

def subject_processed_dir(processed_root: PathLike, subject_id: str) -> Path:
    """Get the processed directory for a given subject."""
    return Path(processed_root) / subject_id

def volume_path(processed_root: PathLike, subject_id: str, modality: str) -> Path:
    """Get the volume path for a given subject and modality."""
    return subject_processed_dir(processed_root, subject_id) / f"{modality}.nii.gz"


def mask_path(processed_root: PathLike, subject_id: str) -> Path:
    """Get the mask path for a given subject."""
    return subject_processed_dir(processed_root, subject_id) / "mask.nii.gz"


def required_processed_paths(
    processed_root: PathLike,
    subject_id: str,
    primary: str,
    secondary: str,
) -> dict[str, Path]:
    """Primary volume, secondary volume, and integer mask for one subject."""
    return {
        primary: volume_path(processed_root, subject_id, primary),
        secondary: volume_path(processed_root, subject_id, secondary),
        "mask": mask_path(processed_root, subject_id),
    }


def subject_is_complete(
    processed_root: PathLike,
    subject_id: str,
    primary: str,
    secondary: str,
) -> bool:
    """True when both modalities and the mask exist as files."""
    return all(
        path.is_file()
        for path in required_processed_paths(
            processed_root, subject_id, primary, secondary
        ).values()
    )


def resolve_raw_volume_path(
    raw_dir: PathLike,
    subject_id: str,
    filename: str,
) -> Optional[Path]:
    """
    Resolve a raw volume path for a given subject and filename.
    
    Steps:
    1. Get the root directory
    2. Check if the candidate paths exist
    3. Return the first existing path
    """
    root = Path(raw_dir) / subject_id
    for candidate in (root / "T1w" / filename, root / filename):
        if candidate.exists():
            return candidate
    return None


########################################################
#          SUBJECT LISTING FUNCTIONS                   #
########################################################

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


########################################################
#          PATCH EXTRACTION AND CENTER SAMPLING        #
########################################################


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
        3D array ``(D, H, W)`` or 4D array ``(D, H, W, C)``.
    center : (z, y, x)
        Patch center in volume coordinates.
    patch_size : (pz, py, px)
        Output spatial size.

    Returns
    -------
    patch : np.ndarray
        Array of shape ``patch_size`` (3D) or ``(*patch_size, C)`` (4D),
        same dtype as ``volume``.
    """

    # 1. Check if the volume is 3D or 4D (D, H, W[, C])
    if volume.ndim not in (3, 4):
        raise ValueError(f"extract_patch_3d expects 3D or 4D volume, got shape {volume.shape}")

    spatial = spatial_shape(volume)

    # 2. Convert the patch size to integers
    patch_size = tuple(int(x) for x in patch_size)

    # 3. Convert the center to integers
    center = [int(c) for c in center]

    # 4. Create an output array of the same shape as the patch size
    if volume.ndim == 3:
        output = np.zeros(patch_size, dtype=volume.dtype)
    else:
        output = np.zeros((*patch_size, volume.shape[3]), dtype=volume.dtype)

    # 5. Create source and destination slices
    src_slices = []
    dst_slices = []

    # 6. For each axis, calculate the source and destination slices
    for i in range(3):
        half = patch_size[i] // 2
        src_start = center[i] - half
        src_end = src_start + patch_size[i]

        dst_start = max(0, -src_start)
        dst_end = patch_size[i] - max(0, src_end - spatial[i])
        src_start_clipped = max(0, src_start)
        src_end_clipped = min(spatial[i], src_end)

        src_slices.append(slice(src_start_clipped, src_end_clipped))
        dst_slices.append(slice(dst_start, dst_end))

    # 7. Check if the source and destination slices are valid
    if all(s.start < s.stop for s in src_slices) and all(
        s.start < s.stop for s in dst_slices
    ):
        # 8. Extract the patch from the volume
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

    Steps:
    1. For each axis, calculate the half size
    2. Calculate the lower and upper bounds
    3. If the upper bound is less than or equal to the lower bound, set the center to the middle of the axis
    4. Otherwise, sample a random integer between the lower and upper bounds
    5. Return the center
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




