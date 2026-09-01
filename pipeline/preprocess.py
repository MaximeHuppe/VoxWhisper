# pipeline/preprocess.py — volumes → masks → prompt embeddings
from __future__ import annotations

from pathlib import Path

from preprocess.cache_embedding import cache_embedding
from preprocess.preprocess_masks import preprocess_masks
from preprocess.preprocess_volumes import preprocess_volumes
from src.utils.config import load_config, parse_config_args, resolve_path


def run_preprocess(config) -> None:
    """Normalize volumes, build integer masks, then cache prompt embeddings."""
    print("=== Step 1/3: preprocess volumes ===")
    preprocess_volumes(config)

    print("=== Step 2/3: preprocess masks ===")
    preprocess_masks(config)

    print("=== Step 3/3: cache prompt embeddings ===")
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
