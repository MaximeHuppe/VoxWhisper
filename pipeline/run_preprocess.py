# pipeline/run_preprocess.py — volumes → masks → prompt embeddings
"""Full preprocessing pipeline for VoxWhisper.

Pipeline overview
-----------------
The steps below must be run in order.  Step 0 is only needed when the
secondary modality is ``fa`` or ``dec_fa``; it is skipped automatically
for ``b0`` / ``t2``.

Step 0  (optional) — DTI-derived maps
    Scalar FA::

        python preprocess/compute_fa.py --config config/tracts.yaml

    3-channel DEC-FA (RGB, principal-direction colouring)::

        python preprocess/compute_dec_fa.py --config config/tracts.yaml

    Both read each subject's 4D diffusion series from
    ``data/raw/{subject_id}/Diffusion/`` and write a map to the subject
    root (``dti_FA.nii.gz`` or ``dti_DEC_FA.nii.gz``).  Run once before
    Steps 1–4.

    Add ``--delete-raw`` to remove ``Diffusion/data.nii.gz`` afterwards and
    reclaim ~1–2 GB per subject.  Add ``--workers N`` to parallelise across
    subjects.

Step 1 — preprocess volumes
    Loads T1 / B0 / FA / DEC-FA from ``data/raw/``, applies z-score
    normalisation (skipped when ``data.volumes.<mod>.normalize`` is false,
    as for DEC-FA), and writes ``{primary}.nii.gz`` and
    ``{secondary}.nii.gz`` to the processed directory.

Step 2 — preprocess masks
    Merges per-tract binary masks from ``tract_masks_1.25/`` into a single
    integer label map ``mask.nii.gz`` in the processed directory.

Step 3 — drop incomplete subjects
    Removes processed subjects missing any required file (primary, secondary,
    mask) to keep the dataset consistent.

Step 4 — cache prompt embeddings
    Encodes the tract-name prompts with PubMedBERT and writes a ``{}.pt``
    cache file used by the DataLoader.

This script runs Steps 1–4 in sequence.  Run Step 0 manually beforehand
when ``data.modalities.secondary`` is ``fa`` or ``dec_fa``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from preprocess.cache_embedding import cache_embedding
from preprocess.preprocess_masks import preprocess_masks
from preprocess.preprocess_volumes import preprocess_volumes
from src.utils.config import (
    active_modality_keys,
    load_config,
    parse_config_args,
    resolve_path,
)
from src.data.nifti_io import required_processed_paths, subject_is_complete


def drop_incomplete_processed(config) -> list[str]:
    """Delete processed subject dirs missing primary, secondary, or mask."""
    processed = resolve_path(config, "data.paths.processed")
    primary, secondary = active_modality_keys(config)
    if not processed.is_dir():
        return []

    removed: list[str] = []
    for child in sorted(processed.iterdir()):
        if not child.is_dir():
            continue
        if subject_is_complete(processed, child.name, primary, secondary):
            continue
        missing = [
            name
            for name, path in required_processed_paths(
                processed, child.name, primary, secondary
            ).items()
            if not path.is_file()
        ]
        shutil.rmtree(child)
        removed.append(child.name)
        print(f"  Removed {child.name} (missing {', '.join(missing)})")
    return removed


_DTI_SECONDARY = {
    "fa": ("dti_FA.nii.gz", "preprocess/compute_fa.py"),
    "dec_fa": ("dti_DEC_FA.nii.gz", "preprocess/compute_dec_fa.py"),
}


def _warn_if_fa_needed(config) -> None:
    """Print a reminder when a DTI-derived secondary map is missing on disk."""
    _, secondary = active_modality_keys(config)
    if secondary not in _DTI_SECONDARY:
        return
    filename, script = _DTI_SECONDARY[secondary]
    raw_dir = resolve_path(config, "data.paths.raw")
    if not raw_dir.is_dir():
        return
    subjects = [d for d in raw_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    missing = [s for s in subjects if not (s / filename).exists()]
    if missing:
        print(
            f"[Warning] {len(missing)} subject(s) have no {filename} "
            f"but secondary modality is '{secondary}'.  Run Step 0 first:\n"
            f"  python {script} --config <your_config.yaml>\n"
        )


def run_preprocess(config) -> None:
    """Run Steps 1–4 of the preprocessing pipeline.

    Step 0 (FA / DEC-FA map generation) is optional and must be run
    separately when ``data.modalities.secondary`` is ``fa`` or ``dec_fa``.
    See the module docstring for the full pipeline overview.
    """
    _warn_if_fa_needed(config)
    print("=== Step 1/4: preprocess volumes ===")
    preprocess_volumes(config)

    print("=== Step 2/4: preprocess masks ===")
    preprocess_masks(config)

    print("=== Step 3/4: drop incomplete processed subjects ===")
    removed = drop_incomplete_processed(config)
    if removed:
        print(f"Removed {len(removed)} incomplete subject(s)")
    else:
        print("No incomplete subjects")

    print("=== Step 4/4: cache prompt embeddings ===")
    prompts = config["data"]["prompts"]
    model_name = config["text_encoder"]["model_name"]
    cache_dir = resolve_path(config, "data.paths.cache")
    cache_file = cache_dir / config["text_encoder"]["cache_file"]
    cache_embedding(prompts, Path(cache_file), model_name)
    print(f"Preprocessing complete. Embeddings at {cache_file}")


if __name__ == "__main__":
    args = parse_config_args(description="Run VoxWhisper preprocessing pipeline")
    cfg = load_config(args.config)
    run_preprocess(cfg)
