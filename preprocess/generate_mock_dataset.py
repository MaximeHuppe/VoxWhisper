# preprocess/generate_mock_dataset.py
"""Generate a synthetic full-resolution T1/T2 NIfTI cohort for patch sampling."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import (  # noqa: E402
    active_modality_keys,
    ensure_dir,
    load_config,
    parse_config_args,
    resolve_path,
)
from src.utils.nifti_io import (  # noqa: E402
    identity_affine,
    mask_path,
    save_nifti,
    subject_processed_dir,
    volume_path,
)


def make_mock_cohort(config, num_subjects=10):
    output_dir = resolve_path(config, "data.paths.processed")
    ensure_dir(output_dir)

    primary, secondary = active_modality_keys(config)
    n_prompts = len(config["data"]["prompts"])
    shape = tuple(config["data"]["mock_volume_shape"])
    patch_size = tuple(config["data"]["patch"]["size"])
    positive_labels = list(config["data"]["patch"].get("positive_labels", [1]))
    affine = identity_affine()

    for dim, pdim in zip(shape, patch_size):
        if dim < pdim:
            raise ValueError(
                f"mock_volume_shape {shape} must be >= patch.size {patch_size}"
            )

    print(
        f"Generating {num_subjects} mock subjects "
        f"({primary}+{secondary}) at shape {shape}, patch={patch_size}..."
    )
    for i in range(num_subjects):
        subject_id = f"SUBJ{i:03d}"
        ensure_dir(subject_processed_dir(output_dir, subject_id))

        for modality in (primary, secondary):
            volume = np.random.randn(*shape).astype(np.float32)
            save_nifti(
                volume,
                volume_path(output_dir, subject_id, modality),
                affine=affine,
                dtype=np.float32,
            )

        # Sparse foreground blob near volume center (simulates optic nerve)
        labels = np.zeros(shape, dtype=np.uint8)
        cz, cy, cx = [s // 2 for s in shape]
        label_id = positive_labels[0] if positive_labels else 1
        labels[cz - 2 : cz + 3, cy - 2 : cy + 3, cx - 4 : cx + 5] = label_id
        # Optionally sprinkle a second structure if prompts have >2 classes
        if n_prompts > 2 and len(positive_labels) > 1:
            labels[cz - 1 : cz + 2, cy - 6 : cy - 3, cx - 1 : cx + 2] = positive_labels[1]

        save_nifti(
            labels,
            mask_path(output_dir, subject_id),
            affine=affine,
            dtype=np.uint8,
        )

    print(f"Mock cohort generated successfully in {output_dir}.")


if __name__ == "__main__":
    args = parse_config_args(description="Generate mock full-resolution NIfTI dataset")
    cfg = load_config(args.config)
    make_mock_cohort(cfg)
