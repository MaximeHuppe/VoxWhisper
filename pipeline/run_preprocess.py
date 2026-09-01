# pipeline/run_preprocess.py — volumes → masks → prompt embeddings
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
from src.utils.nifti_io import required_processed_paths, subject_is_complete


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


def run_preprocess(config) -> None:
    """Normalize volumes, build integer masks, then cache prompt embeddings."""
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
