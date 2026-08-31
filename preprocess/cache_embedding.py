# preprocess/cache_embedding.py
"""Cache PubMedBERT prompt embeddings using config prompts and paths."""
from __future__ import annotations

import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import (  # noqa: E402
    ensure_dir,
    load_config,
    parse_config_args,
    resolve_path,
)


def cache_prompt_embeddings(prompt_list, output_path, model_name):
    """
    Pass medical prompts through a frozen clinical language model,
    pool token-level embeddings per phrase, and save them to disk.
    """
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"Tokenizing and encoding prompts: {prompt_list}")
    # We pad and truncate to convert the list of phrases into a clean tensor
    inputs = tokenizer(prompt_list, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.last_hidden_state shape: [Num_Phrases, Seq_Len, 768]
        token_embeddings = outputs.last_hidden_state
        
        # MEAN POOLING: Average along the Seq_Len dimension (dim=1)
        # This reduces the shape of each phrase to a single 768-dimensional vector
        # Shape transition: [Num_Phrases, Seq_Len, 768] -> [Num_Phrases, 768]
        embeddings = token_embeddings.mean(dim=1)

    ensure_dir(output_path.parent)
    torch.save(embeddings, output_path)
    print(f"Cached text embeddings saved to: {output_path} (Final Shape: {embeddings.shape})")


if __name__ == "__main__":
    args = parse_config_args(description="Cache clinical prompt embeddings")
    cfg = load_config(args.config)

    prompts = cfg["data"]["prompts"]
    model_name = cfg["text_encoder"]["model_name"]
    cache_dir = resolve_path(cfg, "data.paths.cache")
    cache_file = cache_dir / cfg["text_encoder"]["cache_file"]

    cache_prompt_embeddings(prompts, cache_file, model_name)
