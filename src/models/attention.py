import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding3D(nn.Module):
    """
    Generate learned 3D positional embeddings for a spatial grid.
    Uses broadcasting to create a unified coordinate grid from 3 parameter axes.
    """

    def __init__(self, embed_dim=128, d_size=16, h_size=16, w_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_size = d_size
        self.h_size = h_size
        self.w_size = w_size

        # Create 3 separate learnable parameter matrices representing the 3D axes
        self.pos_d = nn.Parameter(torch.randn(1, d_size, 1, 1, embed_dim))
        self.pos_h = nn.Parameter(torch.randn(1, 1, h_size, 1, embed_dim))
        self.pos_w = nn.Parameter(torch.randn(1, 1, 1, w_size, embed_dim))


    def forward(self, x_tokens):
        # x_tokens: Shape [B, N_vox, C] (where N_vox = d_size * h_size * w_size)
        B, N_vox, C = x_tokens.shape 
        assert N_vox == self.d_size * self.h_size * self.w_size, "Sequence length does not match grid dimensions"

        # 1. Broadcast add the 3 axes to form a 3D coordinate grid
        # [1, D, 1, 1, C] + [1, 1, H, 1, C] + [1, 1, 1, W, C] -> [1, D, H, W, C]
        grid = self.pos_d + self.pos_h + self.pos_w

        # 2. Flatten the spatial grid to match the token shape 
        # [1, D, H, W, C] -> [1, D*H*W, C]
        flat_grid = grid.view(1, self.d_size * self.h_size * self.w_size, self.embed_dim)

        # 3. Add posotional embedding to the input tokens (broadcast over batch B)
        return x_tokens + flat_grid



class CrossVolumeAttention(nn.Module):
    """
    Aligns secondary-modality visual features (e.g. T2 W/O diffusion) to the T1 spatial grid.
    The T1 bottleneck features act as Queries (defining the spatial output grid),
    while the secondary features act as Keys and Values.
    """

    def __init__(self, embed_dim=128, num_heads=4, d_size=16, h_size=16, w_size=16):
        super().__init__()
        self.embed_dim = embed_dim

        # Instantiate 3D positional encoders for both modality branches
        self.pos_t1 = PositionalEncoding3D(embed_dim, d_size, h_size, w_size)
        self.pos_sec = PositionalEncoding3D(embed_dim, d_size, h_size, w_size)
        
        # Standard PyTorch Multihead Attention
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Normalization and residual scaling
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, t1_features, secondary_features):
        # t1_features (Query Source):            Shape [B, C, D_t1, H_t1, W_t1]
        # secondary_features (Key/Value Source): Shape [B, C, D_s, H_s, W_s]
        B, C, D_t1, H_t1, W_t1 = t1_features.shape
        _, _, D_s, H_s, W_s = secondary_features.shape

        # 1. Flatten the 3D spatial dimensions of both volumes into sequences of tokens
        # [B, C, D, H, W] -> [B, C, D*H*W] -> [B, D*H*W, C]
        q_tokens = t1_features.view(B, C, D_t1 * H_t1 * W_t1).transpose(1, 2)  # Shape: [B, N_t1, C]
        kv_tokens = secondary_features.view(B, C, D_s * H_s * W_s).transpose(1, 2)  # Shape: [B, N_s, C]

        # 2. Add 3D Positional Embedddings to Queries (Q) and Keys (K)
        # We leave the Values (V) raw so the features are not distorted by coordinate values
        q_tokens_with_pos = self.pos_t1(q_tokens)
        k_tokens_with_pos = self.pos_sec(kv_tokens)

        # 3. Run Cross-Attention
        # Query: T1 (with pos), Keys: T2 (with pos), Values: T2 (raw)
        # Output shape: [B, N_t1, C]
        attn_out, _ = self.mha(query=q_tokens_with_pos, key=k_tokens_with_pos, value=kv_tokens)

        # 4. Residual connection and Layer Normalization
        fused_tokens = self.norm(attn_out + q_tokens) # Shape: [B, N_t1, C]

        # 5. Reshape back into a 3D spatial grid (matching T1 spatial dimensions)
        # [B, N_t1, C] -> [B, C, N_t1] -> [B, C, D_t1, H_t1, W_t1]
        fused_spatial = fused_tokens.transpose(1, 2).view(B, C, D_t1, H_t1, W_t1)
        return fused_spatial


class PromptDecoder(nn.Module):
    """
    Aligns raw text prompt embeddings to the consolidated multi-modal visual map.
    Text prompt tokens act as Queries; fused T1 + secondary visual features are Keys/Values.
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
        kv_tokens = fused_visual_features.view(B, C, D_t1 * H_t1 * W_t1).transpose(1, 2) # Shape: [B, N_t1, C]

        # 3. Add 3D Positional Embeddings to the visual Keys (K)
        # Allows prompt queries to map semantic terms directly to physical coordinates
        k_tokens_with_pos = self.pos_encoder(kv_tokens)

        # 4. Run Language-to-Visual Cross-Attention
        # Query: Text, Key: Visual (with Pos), Value: Visual (Raw features)
        # Output shape: [B, N_T, C]
        attn_out, _ = self.mha(query=q_tokens, key=k_tokens_with_pos, value=kv_tokens)

        # 4. Residual and Normalization
        aligned_queries = self.norm(attn_out + q_tokens) # Shape: [B, N_T, C]

        return aligned_queries
