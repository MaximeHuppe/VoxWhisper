"""Inspect cached PubMedBERT prompt embeddings.

Usage (from the project root):

    python cache/print_embedding.py
    python cache/print_embedding.py --cache cache/prompts_tracts.pt
"""
from __future__ import annotations

import argparse
import os

import torch

from src.utils.config import load_config


def inspect_prompt_embeddings(cache_path: str) -> None:
    if not os.path.exists(cache_path):
        print(f"Error: cache file '{cache_path}' not found.")
        print("Run 'python preprocess/cache_embedding.py' to generate it.")
        return

    try:
        embeddings = torch.load(cache_path, map_location="cpu", weights_only=True).squeeze()
    except TypeError:
        embeddings = torch.load(cache_path, map_location="cpu").squeeze()

    cfg = load_config()
    prompts = cfg["data"].get("prompts", [])
    labels = prompts if len(prompts) == embeddings.shape[0] else [f"Class {i}" for i in range(embeddings.shape[0])]

    print("=" * 52)
    print("     VOXWHISPER PROMPT EMBEDDINGS INSPECTOR")
    print("=" * 52)
    print(f"  File    : {cache_path}")
    print(f"  Shape   : {list(embeddings.shape)}  (N_prompts × embed_dim)")
    print(f"  dtype   : {embeddings.dtype}")
    print("=" * 52)

    for idx, label in enumerate(labels):
        vec = embeddings[idx] if embeddings.ndim > 1 else embeddings
        _print_vector(idx, label, vec)


def _print_vector(index: int, label: str, vector: torch.Tensor) -> None:
    print(f"\nPrompt {index}: '{label}'")
    print(f"  mean={vector.mean():.4f}  std={vector.std():.4f}  "
          f"min={vector.min():.4f}  max={vector.max():.4f}")
    first = [round(x, 4) for x in vector[:5].tolist()]
    last  = [round(x, 4) for x in vector[-5:].tolist()]
    print(f"  first 5: {first}")
    print(f"  last  5: {last}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect cached prompt embeddings")
    parser.add_argument(
        "--cache",
        default="cache/prompts_tracts.pt",
        help="Path to the .pt embedding file (default: cache/prompts_tracts.pt)",
    )
    args = parser.parse_args()
    inspect_prompt_embeddings(args.cache)
