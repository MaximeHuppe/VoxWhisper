"""Train-time patch augmentation for dual-modality 3D MRI segmentation.

Transforms are chosen around two constraints for this dataset:

- **RAS orientation.** HCP ACPC-aligned volumes are stored with axes
  (W=R, H=A, D=S).  The brain is anatomically symmetric only along L-R,
  so only the LR flip is valid.  Flipping A-P or S-I creates anatomically
  impossible patches (UF curves wrong way, ATR projects toward occipital
  pole) which would mislead the text-conditioned decoder.

- **Label integrity.** Transforms applied to the integer label patch must
  produce only the label values that exist in the input — no interpolation
  artefacts.  The LR flip achieves this exactly.  The fast rotation uses a
  single nearest-neighbour pass (never three sequential ones) to limit
  staircase artefacts at tract boundaries.

Spatial transforms are applied identically to (primary, secondary, label).
Noise is per-modality image only and never touches labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

VolumeTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]


def left_right_label_pairs(structure_names: Sequence[str]) -> dict[int, int]:
    """Bidirectional label remap for ``*_left`` / ``*_right`` pairs.

    Structure names must be ordered by ascending integer label, as injected by
    ``load_config``.  Returns ``{label_a: label_b, ...}``.

    Example::

        left_right_label_pairs(["background", "ATR_left", "ATR_right"])
        # → {1: 2, 2: 1}
    """
    index = {name: i for i, name in enumerate(structure_names)}
    remap: dict[int, int] = {}
    for name, label in index.items():
        if not name.endswith("_left"):
            continue
        right_name = name[: -len("_left")] + "_right"
        if right_name in index:
            right_label = index[right_name]
            remap[label] = right_label
            remap[right_label] = label
    return remap


def _compose_rotation(angles_deg: Sequence[float]) -> np.ndarray:
    """3×3 rotation matrix from Euler angles (degrees) about axes 0, 1, 2.

    Returns R = R2 @ R1 @ R0 (applied right-to-left, i.e. axis-0 first).
    """
    matrices = []
    for i, angle in enumerate(angles_deg):
        a = np.deg2rad(float(angle))
        c, s = np.cos(a), np.sin(a)
        if i == 0:
            matrices.append(np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64))
        elif i == 1:
            matrices.append(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64))
        else:
            matrices.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64))
    R = matrices[2] @ matrices[1] @ matrices[0]
    return R


def _fast_rotate_3d(volume: np.ndarray, R: np.ndarray, order: int) -> np.ndarray:
    """Rotate ``volume`` about its center in a single resampling pass.

    ``R`` is the 3×3 forward rotation matrix.  ``ndimage.affine_transform``
    maps output → input coordinates, so we pass ``R.T`` (the inverse rotation
    for an orthogonal matrix) and compute the matching center offset.

    Using one pass instead of three sequential ``ndimage.rotate`` calls is
    ~3× faster and avoids compounding interpolation artefacts on the label patch.
    """
    center = (np.array(volume.shape, dtype=np.float64) - 1) / 2.0
    # output_coord = R @ (input_coord - center) + center
    # input_coord  = R.T @ output_coord + (center - R.T @ center)
    offset = center - R.T @ center
    out = ndimage.affine_transform(
        volume,
        R.T,
        offset=offset,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )
    if order == 0:
        out = np.rint(out).astype(volume.dtype, copy=False)
    return out.astype(np.float32 if np.issubdtype(volume.dtype, np.floating) else volume.dtype)


@dataclass
class PatchAugmentor:
    """Augmentor for aligned ``(primary, secondary, label)`` triples.

    Parameters
    ----------
    enabled :
        Master switch.  ``__call__`` is a no-op when ``False``.
    flip_p :
        Probability of flipping along the left-right axis (``lr_axis``).
        Only the LR flip is enabled by default; A-P and S-I flips are not
        anatomically valid for brain tract segmentation in RAS space.
    lr_axis :
        The left-right spatial axis in ``(D, H, W)`` patches.  For HCP
        ACPC-aligned volumes loaded without reorientation this is axis 2 (W=R).
        Flips on this axis also apply ``lr_remap``.
    lr_remap :
        Bidirectional label map built from ``left_right_label_pairs``.
    rotate_p :
        Probability of applying a small rotation.  Disabled by default
        (``0.0``) because nearest-neighbour resampling on thin tract labels
        introduces boundary artefacts.  Enable only after validating on a
        pilot run.
    max_angle_deg :
        Maximum rotation angle per axis when ``rotate_p > 0``.
    noise_p :
        Probability of adding Gaussian noise to each image patch.
    noise_std :
        Noise magnitude as a fraction of the patch's own std (e.g. 0.05 = 5 %).
    """

    enabled: bool = True
    flip_p: float = 0.5
    lr_axis: int = 2
    lr_remap: dict[int, int] = field(default_factory=dict)
    rotate_p: float = 0.0
    max_angle_deg: float = 10.0
    noise_p: float = 0.5
    noise_std: float = 0.05

    @classmethod
    def from_config(cls, config: Mapping) -> "PatchAugmentor":
        """Build from ``config["data"]["augmentation"]``.

        Augmentation is **disabled** when the section is absent or
        ``enabled: false``.
        """
        aug = (config.get("data") or {}).get("augmentation") or {}
        if not aug.get("enabled", False):
            return cls(enabled=False)

        structure_names = (config.get("data") or {}).get("structure_names") or []

        return cls(
            enabled=True,
            flip_p=float(aug.get("flip_p", 0.5)),
            lr_axis=int(aug.get("lr_axis", 2)),
            lr_remap=left_right_label_pairs(structure_names),
            rotate_p=float(aug.get("rotate_p", 0.0)),
            max_angle_deg=float(aug.get("max_angle_deg", 10.0)),
            noise_p=float(aug.get("noise_p", 0.5)),
            noise_std=float(aug.get("noise_std", 0.05)),
        )

    def __call__(
        self,
        primary: np.ndarray,
        secondary: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> VolumeTriple:
        """Return augmented ``(primary, secondary, labels)``."""
        if not self.enabled:
            return primary, secondary, labels

        if primary.ndim != 3:
            raise ValueError(f"Expected 3D patches, got shape {primary.shape}")
        if primary.shape != secondary.shape or primary.shape != labels.shape:
            raise ValueError(
                "All three patches must have the same shape; got "
                f"primary={primary.shape}, secondary={secondary.shape}, "
                f"labels={labels.shape}"
            )

        primary = primary.astype(np.float32, copy=True)
        secondary = secondary.astype(np.float32, copy=True)
        labels = labels.copy()

        # LR flip: exact, no interpolation.
        if rng.random() < self.flip_p:
            primary = np.ascontiguousarray(np.flip(primary, axis=self.lr_axis))
            secondary = np.ascontiguousarray(np.flip(secondary, axis=self.lr_axis))
            labels = np.ascontiguousarray(np.flip(labels, axis=self.lr_axis))
            if self.lr_remap:
                remapped = labels.copy()
                for src, dst in self.lr_remap.items():
                    remapped[labels == src] = dst
                labels = remapped

        # Rotation: opt-in, single resampling pass.
        if self.rotate_p > 0 and rng.random() < self.rotate_p:
            angles = rng.uniform(-self.max_angle_deg, self.max_angle_deg, size=3)
            R = _compose_rotation(angles)
            primary = _fast_rotate_3d(primary, R, order=1)
            secondary = _fast_rotate_3d(secondary, R, order=1)
            labels = _fast_rotate_3d(labels, R, order=0)

        # Noise: per-modality, never touches labels.
        primary = self._add_noise(primary, rng)
        secondary = self._add_noise(secondary, rng)
        return primary, secondary, labels

    def _add_noise(self, volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.noise_p <= 0 or self.noise_std <= 0 or rng.random() >= self.noise_p:
            return volume
        std = float(volume.std())
        if std <= 0:
            return volume
        return volume + rng.normal(0.0, self.noise_std * std, size=volume.shape).astype(np.float32)
