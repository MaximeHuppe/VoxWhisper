"""Cache PubMedBERT prompt embeddings for the configured prompts."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from voxwhisper.util.config import ensure_dir


def cache_embedding(prompt_list: list, output_path: Path, model_name: str) -> None:
    """Encode ``prompt_list`` with ``model_name`` and save to ``output_path``.

    Uses mean-pooled last hidden states (attention-mask weighted).
    """
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"Tokenizing and encoding prompts: {prompt_list}")
    inputs = tokenizer(prompt_list, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # [N, L, 1]
        embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

    ensure_dir(output_path.parent)
    torch.save(embeddings, output_path)
    print(f"Cached text embeddings saved to: {output_path} (shape: {embeddings.shape})")
