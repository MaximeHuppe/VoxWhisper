"""Cross-modal and language-to-visual attention modules for VoxWhisper.

Modules
-------
PositionalEncoding3D
    Learned, decomposed 3D positional embeddings added to flattened spatial
    token sequences before attention.  Uses three independent parameter tensors
    (one per axis) that sum via broadcasting, keeping parameter count linear in
    the spatial grid size rather than cubic.

CrossVolumeAttention
    Aligns the secondary modality (T2 / Diffusion / B0) to the primary (T1) coordinate grid.
    Primary bottleneck tokens act as Queries; secondary tokens are Keys and
    Values.  Positional encoding is added to both Q and K so the attention
    pattern learns spatial correspondence.  Values intentionally do *not*
    receive positional encoding — they carry feature content, not position.

PromptDecoder
    Aligns frozen language embeddings to the fused visual bottleneck.  Text
    tokens act as Queries (no positional encoding, since language has no spatial
    meaning); visual tokens are Keys+PE and Values (no PE).  The text tokens
    update by attending over the full 3D feature map, producing language-aligned
    visual queries for the decoder.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


def bottleneck_spatial_size(
    patch_size: Sequence[int],
    strides: Sequence[Union[int, Sequence[int]]],
) -> Tuple[int, int, int]:
    """Compute the spatial size of the encoder bottleneck.

    Applies each stride sequentially as an integer floor-division.

    Parameters
    ----------
    patch_size : (D, H, W) input patch size.
    strides    : per-stage stride values (scalar or 3-tuple each).

    Returns
    -------
    (D_b, H_b, W_b) bottleneck spatial size.

    Raises
    ------
    ValueError if ``patch_size`` is not length-3, any stride < 1, or the
    bottleneck collapses to 0 along any axis.
    """
    size = [int(x) for x in patch_size]
    if len(size) != 3:
        raise ValueError(f"patch_size must be length 3, got {patch_size}")
    for stride in strides:
        step = int(stride[0] if isinstance(stride, (list, tuple)) else stride)
        if step < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        size = [dim // step for dim in size]
        if any(dim < 1 for dim in size):
            raise ValueError(
                f"Bottleneck collapsed to {tuple(size)} from patch "
                f"{tuple(int(x) for x in patch_size)} and strides {list(strides)}"
            )
    return (size[0], size[1], size[2])


class PositionalEncoding3D(nn.Module):
    """Decomposed learned 3D positional embeddings.

    Instead of a single ``(D*H*W, C)`` table (which would require cubic memory),
    three independent parameter vectors along D, H, W are summed by broadcasting:

        pos[d, h, w] = pos_d[d] + pos_h[h] + pos_w[w]

    This keeps the parameter count at ``3 × grid_size × embed_dim``.
    The spatial grid is fixed at construction time; the runtime size must match.

    Parameters
    ----------
    embed_dim            : Feature dimension C.
    d_size, h_size, w_size : Fixed spatial grid size (bottleneck resolution).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        d_size: int = 16,
        h_size: int = 16,
        w_size: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.d_size = int(d_size)
        self.h_size = int(h_size)
        self.w_size = int(w_size)

        self.pos_d = nn.Parameter(torch.randn(1, self.d_size, 1, 1, embed_dim))
        self.pos_h = nn.Parameter(torch.randn(1, 1, self.h_size, 1, embed_dim))
        self.pos_w = nn.Parameter(torch.randn(1, 1, 1, self.w_size, embed_dim))

    def forward(
        self,
        x_tokens: torch.Tensor,
        spatial_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """Add positional encoding to a flattened spatial sequence.

        Parameters
        ----------
        x_tokens     : [B, D*H*W, C]
        spatial_size : (D, H, W) matching the token sequence length.
                       Defaults to the trained grid size.

        Returns
        -------
        Tensor [B, D*H*W, C] with positional encoding added.
        """
        D, H, W = spatial_size or (self.d_size, self.h_size, self.w_size)
        D, H, W = int(D), int(H), int(W)
        if (D, H, W) != (self.d_size, self.h_size, self.w_size):
            raise ValueError(
                f"Runtime spatial size {(D, H, W)} does not match "
                f"trained grid {(self.d_size, self.h_size, self.w_size)}"
            )
        if x_tokens.shape[1] != D * H * W:
            raise ValueError(
                f"Sequence length {x_tokens.shape[1]} does not match "
                f"spatial_size={D, H, W} (expected {D * H * W} tokens)"
            )

        grid = self.pos_d + self.pos_h + self.pos_w  # [1, d_size, h_size, w_size, C]
        return x_tokens + grid.reshape(1, D * H * W, self.embed_dim)


class CrossVolumeAttention(nn.Module):
    """Cross-modal spatial alignment: secondary → primary coordinate grid.

    Primary bottleneck features act as Queries; secondary features are Keys and
    Values.  Both Q and K receive positional encoding so the attention pattern
    learns spatial correspondence between the two modalities.  Values are left
    without PE because they carry feature *content* rather than position.

    The module outputs fused primary-space tokens that encode information from
    both modalities, then reshapes them back to the original 3D grid.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        d_size: int = 16,
        h_size: int = 16,
        w_size: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.pos_primary = PositionalEncoding3D(embed_dim, d_size, h_size, w_size)
        self.pos_sec = PositionalEncoding3D(embed_dim, d_size, h_size, w_size)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        primary_features: torch.Tensor,
        secondary_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        primary_features   : [B, C, D_p, H_p, W_p]
        secondary_features : [B, C, D_s, H_s, W_s]

        Returns
        -------
        Tensor [B, C, D_p, H_p, W_p] — fused primary-space feature map.
        """
        B, C, D_p, H_p, W_p = primary_features.shape
        _, _, D_s, H_s, W_s = secondary_features.shape

        q_tokens = primary_features.reshape(B, C, D_p * H_p * W_p).transpose(1, 2)
        kv_tokens = secondary_features.reshape(B, C, D_s * H_s * W_s).transpose(1, 2)

        q_with_pe = self.pos_primary(q_tokens, (D_p, H_p, W_p))
        k_with_pe = self.pos_sec(kv_tokens, (D_s, H_s, W_s))

        # Values are not positionally encoded: they carry feature content.
        attn_out, _ = self.mha(query=q_with_pe, key=k_with_pe, value=kv_tokens)
        fused = self.norm(attn_out + q_tokens)

        return fused.transpose(1, 2).reshape(B, C, D_p, H_p, W_p)


class PromptDecoder(nn.Module):
    """Language-to-visual alignment: text queries attend over the fused visual map.

    Text prompt tokens (from frozen PubMedBERT embeddings, projected to
    ``embed_dim``) act as Queries.  The flattened fused visual bottleneck
    provides Keys+PE and Values (no PE).  This asymmetry is intentional: Keys
    carry spatial position so the text tokens know *where* to attend; Values
    carry the feature content that updates the text tokens.

    The output is a set of ``N_T`` language-aligned queries, one per prompt,
    that are passed to the decoder's ``StageVLFusionBlock`` at every scale.
    """

    def __init__(
        self,
        text_dim: int = 768,
        embed_dim: int = 128,
        num_heads: int = 4,
        d_size: int = 16,
        h_size: int = 16,
        w_size: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.pos_encoder = PositionalEncoding3D(embed_dim, d_size, h_size, w_size)
        self.text_projection = nn.Linear(text_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        text_embeddings: torch.Tensor,
        fused_visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        text_embeddings      : [B, N_T, text_dim]
        fused_visual_features: [B, C, D, H, W]  (primary-space bottleneck)

        Returns
        -------
        aligned_queries : [B, N_T, embed_dim]
        """
        B, C, D, H, W = fused_visual_features.shape

        q_tokens = self.text_projection(text_embeddings)  # [B, N_T, embed_dim]

        kv_tokens = fused_visual_features.reshape(B, C, D * H * W).transpose(1, 2)
        k_with_pe = self.pos_encoder(kv_tokens, (D, H, W))

        # Values are not positionally encoded: they carry feature content.
        attn_out, _ = self.mha(query=q_tokens, key=k_with_pe, value=kv_tokens)
        return self.norm(attn_out + q_tokens)
