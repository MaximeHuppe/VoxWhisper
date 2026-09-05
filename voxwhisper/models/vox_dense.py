"""VoxDense: T1-only language-grounded volumetric segmentation (Phase 1)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import PromptDecoder, bottleneck_spatial_size
from .decoder import Decoder
from .encoder import Encoder


def _fmt_count(n: int) -> str:
    return f"{n:,}"


class VoxDense(nn.Module):
    """Single-encoder promptable UNet for dense brain structures.

    Forward: ``encoder(T1) → PromptDecoder → Decoder``.

    No FA encoder and no cross-volume attention. Phase 2 composes
    ``VoxWhisper`` around the frozen ``encoder`` weights saved here.
    """

    def __init__(
        self,
        input_channels: int = 1,
        text_dim: int = 768,
        embed_dim: int = 256,
        channels: list | None = None,
        strides: list | None = None,
        kernel_sizes: list | None = None,
        paddings: list | None = None,
        num_resblocks: list | None = None,
        num_heads: int = 4,
        pe_size: tuple = (16, 16, 16),
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256]
        if strides is None:
            strides = [2, 2, 2]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if paddings is None:
            paddings = [1, 1, 1]
        if num_resblocks is None:
            num_resblocks = [1, 1, 1]
        pe_size = tuple(int(x) for x in pe_size)

        self.encoder = Encoder(
            input_channels=input_channels,
            channels=channels,
            strides=strides,
            kernel_sizes=kernel_sizes,
            paddings=paddings,
            num_resblocks=num_resblocks,
        )
        self.prompt_decoder = PromptDecoder(
            text_dim=text_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_size=pe_size[0],
            h_size=pe_size[1],
            w_size=pe_size[2],
        )
        self.decoder = Decoder(channels=channels, query_dim=embed_dim)

    @classmethod
    def from_config(cls, config: dict) -> "VoxDense":
        """Construct the model from the ``model`` section of a YAML config."""
        model_cfg = config["model"]
        enc = model_cfg["encoder"]
        pe_size = bottleneck_spatial_size(
            config["data"]["patch"]["size"], enc["strides"]
        )
        return cls(
            input_channels=model_cfg["input_channels"],
            text_dim=model_cfg["text_dim"],
            embed_dim=model_cfg["embed_dim"],
            channels=enc["channels"],
            strides=enc["strides"],
            kernel_sizes=enc["kernel_sizes"],
            paddings=enc["paddings"],
            num_resblocks=enc["num_resblocks"],
            num_heads=model_cfg["num_heads"],
            pe_size=pe_size,
        )

    def param_counts(self) -> dict[str, int]:
        """Return parameter counts for the full model and top-level blocks."""
        total = sum(p.numel() for p in self.parameters())
        counts: dict[str, int] = {"total": total}
        for name, child in self.named_children():
            counts[name] = sum(p.numel() for p in child.parameters())
        return counts

    def print_param_counts(self) -> None:
        """Print a compact parameter breakdown (total + top-level blocks)."""
        counts = self.param_counts()
        total = counts["total"]
        print(f"  Params     total={_fmt_count(total)}")
        for name, child in self.named_children():
            n = counts[name]
            share = (100.0 * n / total) if total else 0.0
            print(f"             {name:<22} {_fmt_count(n):>12}  ({share:5.1f}%)")

    def forward(self, volume: torch.Tensor, text_embeddings: torch.Tensor):
        bottleneck, skips = self.encoder(volume)
        aligned_queries = self.prompt_decoder(
            text_embeddings=text_embeddings,
            fused_visual_features=bottleneck,
        )
        return self.decoder(
            fused_bottleneck=bottleneck,
            skips=skips,
            aligned_queries=aligned_queries,
        )
