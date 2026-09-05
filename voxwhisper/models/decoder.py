"""3D upsampling decoder with hierarchical vision-language fusion.

Architecture
------------
The decoder unrolls the encoder's skip connections bottom-up.  At each scale
it performs:

  1. Trilinear upsample to the skip's spatial resolution.
  2. Skip concatenation + 3×3×3 conv to merge channels.
  3. ``StageVLFusionBlock``: channel-wise sigmoid gating conditioned on the
     language queries, followed by a scaled dot-product to produce per-prompt
     mask logits at this resolution.

All three decoder outputs (1/4, 1/2, full resolution) are returned for deep
supervision during training; only the last (full-resolution) output is used at
inference.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageVLFusionBlock(nn.Module):
    """Vision-language fusion at a single decoder scale.

    Given visual feature maps and the language-aligned query tokens produced by
    ``PromptDecoder``, this block:

    1. **Per-prompt channel gating** – each query is projected to the visual
       channel dimension and applies its own sigmoid gate.  A previous mean-pool
       over prompts made the shared gate depend on ``N_T`` and broke train/val
       when training used ``prompts_per_crop`` but validation used all prompts.
    2. **Mask projection** – computes per-voxel, per-prompt affinities via a
       scaled dot-product between the gated (per-prompt) queries and every
       spatial voxel, producing logit maps of shape ``[B, N_T, D, H, W]``.

    The scaling factor ``1 / sqrt(visual_channels)`` prevents the dot products
    from growing large when ``visual_channels`` is large, avoiding sigmoid
    saturation in the mask head.

    Parameters
    ----------
    visual_channels:
        Spatial feature channels at this decoder stage (e.g. 64, 32, 16).
    query_dim:
        Dimensionality of the aligned language queries from ``PromptDecoder``
        (equal to the model's global ``embed_dim``).
    """

    def __init__(self, visual_channels: int, query_dim: int = 128) -> None:
        super().__init__()
        self.visual_channels = visual_channels
        self.scale = 1.0 / math.sqrt(visual_channels)

        self.query_adapter = nn.Sequential(
            nn.Linear(query_dim, visual_channels),
            nn.ReLU(inplace=True),
            nn.Linear(visual_channels, visual_channels),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        aligned_queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        visual_features : [B, C_vis, D, H, W]
        aligned_queries : [B, N_T, C_query]

        Returns
        -------
        modulated_vis  : [B, C_vis, D, H, W]  – unchanged visual features
        mask_logits    : [B, N_T, D, H, W]    – per-prompt spatial logits
        """
        B, C_vis, D, H, W = visual_features.shape

        # Project each query token to the visual channel space: [B, N_T, C_vis]
        projected_queries = self.query_adapter(aligned_queries)
        # Per-prompt gate — independent of how many prompts are in the batch.
        gated_queries = projected_queries * torch.sigmoid(projected_queries)

        # Scaled dot-product: gated queries × voxels → [B, N_T, D*H*W] → [B, N_T, D, H, W]
        flat_vis = visual_features.view(B, C_vis, D * H * W)
        mask_logits = (gated_queries @ flat_vis) * self.scale
        mask_logits = mask_logits.view(B, -1, D, H, W)

        return visual_features, mask_logits


class Decoder(nn.Module):
    """3D upsampling decoder with per-scale vision-language fusion.

    Parameters
    ----------
    channels:
        The same channel list passed to ``Encoder``.  The last entry is the
        bottleneck width; the preceding entries are the skip widths (shallowest
        to deepest).
    query_dim:
        Dimension of the aligned language queries (model ``embed_dim``).

    Forward
    -------
    fused_bottleneck : [B, channels[-1], D_b, H_b, W_b]
        Output of ``CrossVolumeAttention`` (aligned multi-modal bottleneck).
    skips : list of Tensors [shallowest … deepest]
        Skip connections from the primary encoder.
    aligned_queries : [B, N_T, query_dim]
        Language-aligned tokens from ``PromptDecoder``.

    Returns
    -------
    stage_predictions : list of 3 Tensors
        [0] ``[B, N_T, D//4, H//4, W//4]``  (lowest resolution)
        [1] ``[B, N_T, D//2, H//2, W//2]``
        [2] ``[B, N_T, D,    H,    W   ]``  (full resolution)
    """

    def __init__(
        self,
        channels: Optional[List[int]] = None,
        query_dim: int = 128,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [16, 32, 64, 128]

        skip_channels = channels[:-1]  # [16, 32, 64] for default config

        self.up_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()

        in_ch = channels[-1]
        for skip_ch in reversed(skip_channels):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_ch + skip_ch, skip_ch, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm3d(skip_ch),
                    nn.ReLU(inplace=True),
                )
            )
            self.fusion_blocks.append(
                StageVLFusionBlock(visual_channels=skip_ch, query_dim=query_dim)
            )
            in_ch = skip_ch

    def forward(
        self,
        fused_bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
        aligned_queries: torch.Tensor,
    ) -> List[torch.Tensor]:
        dec_features = fused_bottleneck
        stage_predictions: List[torch.Tensor] = []

        # Reverse skips: from deepest → shallowest to match upsampling order
        for skip, up_block, fusion_block in zip(
            reversed(skips), self.up_blocks, self.fusion_blocks
        ):
            dec_features = F.interpolate(
                dec_features,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=True,
            )
            dec_features = up_block(torch.cat([dec_features, skip], dim=1))
            dec_features, mask_logits = fusion_block(dec_features, aligned_queries)
            stage_predictions.append(mask_logits)

        return stage_predictions
