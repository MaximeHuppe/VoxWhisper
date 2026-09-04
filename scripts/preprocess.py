"""Run the VoxWhisper preprocessing pipeline (Steps 1–4).

Steps
-----
0. FA map generation  (run scripts/compute_fa.py first if needed)
1. Preprocess volumes  (T1 + FA → z-score normalised NIfTI)
2. Preprocess masks    (tract masks → integer label map on T1 grid)
3. Drop incomplete subjects
4. Cache prompt embeddings
"""
from __future__ import annotations

import argparse
from pathlib import Path

from voxwhisper.config import load_config, resolve_path
from voxwhisper.data.nifti_io import list_subject_ids, subject_is_complete
from voxwhisper.data.preprocess.volumes import preprocess_volumes
from voxwhisper.data.preprocess.masks import preprocess_masks
from voxwhisper.data.preprocess.embeddings import cache_embedding


def _warn_if_fa_needed(config) -> None:
    """Remind the user to run compute_fa.py when FA maps are missing."""
    raw_dir = resolve_path(config, "data.paths.raw")
    if not raw_dir.is_dir():
        return
    subjects = [d for d in raw_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    missing = [s for s in subjects if not (s / "dti_FA.nii.gz").exists()]
    if missing:
        print(
            f"[Warning] {len(missing)} subject(s) have no dti_FA.nii.gz. "
            "Run Step 0 first:\n"
            "  python scripts/compute_fa.py --config config/best_config.yaml\n"
        )


def drop_incomplete_processed(config) -> list[str]:
    """Remove processed subject folders that are missing T1, FA, or mask."""
    processed_dir = resolve_path(config, "data.paths.processed")
    subjects = list_subject_ids(processed_dir)
    removed = []
    for sid in subjects:
        if not subject_is_complete(processed_dir, sid, "t1", "fa"):
            import shutil
            subj_path = processed_dir / sid
            if subj_path.is_dir():
                shutil.rmtree(subj_path)
            removed.append(sid)
    return removed


def run_preprocess(config) -> None:
    """Run Steps 1–4 of the preprocessing pipeline."""
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run VoxWhisper preprocessing pipeline")
    parser.add_argument("--config", default="config/best_config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_preprocess(cfg)


if __name__ == "__main__":
    main()
