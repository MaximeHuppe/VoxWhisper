import torch
import torch.nn as nn
import torch.nn.functional as F


def bottleneck_spatial_size(patch_size, strides):
    """Spatial size after successive encoder downsamples: patch / product(strides)."""
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
                f"Bottleneck collapsed to {tuple(size)} from patch {tuple(int(x) for x in patch_size)} "
                f"and strides {list(strides)}"
            )
    return tuple(size)


class PositionalEncoding3D(nn.Module):
    """
    Generate learned 3D positional embeddings for a spatial grid.
    Uses broadcasting to create a unified coordinate grid from 3 parameter axes.
    Interpolates to the runtime spatial size when it differs from the learned grid.
    """

    def __init__(self, embed_dim=128, d_size=16, h_size=16, w_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_size = int(d_size)
        self.h_size = int(h_size)
        self.w_size = int(w_size)

        self.pos_d = nn.Parameter(torch.randn(1, self.d_size, 1, 1, embed_dim))
        self.pos_h = nn.Parameter(torch.randn(1, 1, self.h_size, 1, embed_dim))
        self.pos_w = nn.Parameter(torch.randn(1, 1, 1, self.w_size, embed_dim))

    def forward(self, x_tokens, spatial_size=None):
        # x_tokens: Shape [B, N_vox, C]
        D, H, W = spatial_size or (self.d_size, self.h_size, self.w_size)
        D, H, W = int(D), int(H), int(W)
        if x_tokens.shape[1] != D * H * W:
            raise ValueError(
                f"Sequence length {x_tokens.shape[1]} does not match spatial size {(D, H, W)}"
            )

        grid = self.pos_d + self.pos_h + self.pos_w
        if (D, H, W) != (self.d_size, self.h_size, self.w_size):
            grid = F.interpolate(
                grid.permute(0, 4, 1, 2, 3),
                size=(D, H, W),
                mode="trilinear",
                align_corners=True,
            ).permute(0, 2, 3, 4, 1)

        flat_grid = grid.reshape(1, D * H * W, self.embed_dim)
        return x_tokens + flat_grid



class CrossVolumeAttention(nn.Module):
    """
    Aligns secondary-modality visual features to the primary (output-space) grid.
    Primary bottleneck features act as Queries; secondary features are Keys and Values.
    """

    def __init__(self, embed_dim=128, num_heads=4, d_size=16, h_size=16, w_size=16):
        super().__init__()
        self.embed_dim = embed_dim

        self.pos_primary = PositionalEncoding3D(embed_dim, d_size, h_size, w_size)
        self.pos_sec = PositionalEncoding3D(embed_dim, d_size, h_size, w_size)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, primary_features, secondary_features):
        # primary_features:   Shape [B, C, D_p, H_p, W_p]
        # secondary_features: Shape [B, C, D_s, H_s, W_s]
        B, C, D_p, H_p, W_p = primary_features.shape
        _, _, D_s, H_s, W_s = secondary_features.shape

        q_tokens = primary_features.reshape(B, C, D_p * H_p * W_p).transpose(1, 2)
        kv_tokens = secondary_features.reshape(B, C, D_s * H_s * W_s).transpose(1, 2)

        q_tokens_with_pos = self.pos_primary(q_tokens, (D_p, H_p, W_p))
        k_tokens_with_pos = self.pos_sec(kv_tokens, (D_s, H_s, W_s))

        attn_out, _ = self.mha(query=q_tokens_with_pos, key=k_tokens_with_pos, value=kv_tokens)
        fused_tokens = self.norm(attn_out + q_tokens)

        return fused_tokens.transpose(1, 2).reshape(B, C, D_p, H_p, W_p)


class PromptDecoder(nn.Module):
    """
    Aligns raw text prompt embeddings to the consolidated multi-modal visual map.
    Text prompt tokens act as Queries; fused primary + secondary visual features are Keys/Values.
    """

    def __init__(self, text_dim=768, embed_dim=128, num_heads=4, d_size=16, h_size=16, w_size=16):
        super().__init__()
        self.embed_dim = embed_dim

        # Instantiate 3D Positional Encoder for the visual key/value path
        self.pos_encoder = PositionalEncoding3D(embed_dim, d_size, h_size, w_size )
        
        # Projects raw text embeddings (e.g. from PubMedBERT) to match visual channel depth
        self.text_projection = nn.Linear(text_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text_embeddings, fused_visual_features):
        # text_embeddings:      Shape [B, N_T, text_dim]  (where N_T is prompt length)
        # fused_visual_features: Shape [B, C, D_t1, H_t1, W_t1] (aligned spatial bottleneck)
        B, C, D_t1, H_t1, W_t1 = fused_visual_features.shape

        # 1. Project the text features to the shared embedding space dimension
        # [B, N_T, text_dim] -> [B, N_T, C]
        q_tokens = self.text_projection(text_embeddings)

        # 2. Flatten the fused 3D visual map to serve as the visual key/value memory
        # [B, C, D_t1, H_t1, W_t1] -> [B, C, D_t1*H_t1*W_t1] -> [B, D_t1*H_t1*W_t1, C]
        kv_tokens = fused_visual_features.reshape(B, C, D_t1 * H_t1 * W_t1).transpose(1, 2)
        k_tokens_with_pos = self.pos_encoder(kv_tokens, (D_t1, H_t1, W_t1))

        attn_out, _ = self.mha(query=q_tokens, key=k_tokens_with_pos, value=kv_tokens)

        # 3. Residual and Normalization
        aligned_queries = self.norm(attn_out + q_tokens) # Shape: [B, N_T, C]

        return aligned_queries
