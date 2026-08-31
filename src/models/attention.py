import torch.nn as nn


class CrossVolumeAttention(nn.Module):
    """
    Aligns secondary-modality visual features (e.g. T2 W/O diffusion) to the T1 spatial grid.
    The T1 bottleneck features act as Queries (defining the spatial output grid),
    while the secondary features act as Keys and Values.
    """

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        
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

        # 2. Run Cross-Attention
        # Query: T1, Keys/Values: Diffusion
        # Output shape: [B, N_t1, C]
        attn_out, _ = self.mha(query=q_tokens, key=kv_tokens, value=kv_tokens)

        # 3. Residual connection and Layer Normalization
        fused_tokens = self.norm(attn_out + q_tokens) # Shape: [B, N_t1, C]

        # 4. Reshape back into a 3D spatial grid (matching T1 spatial dimensions)
        # [B, N_t1, C] -> [B, C, N_t1] -> [B, C, D_t1, H_t1, W_t1]
        fused_spatial = fused_tokens.transpose(1, 2).view(B, C, D_t1, H_t1, W_t1)
        return fused_spatial


class PromptDecoder(nn.Module):
    """
    Aligns raw text prompt embeddings to the consolidated multi-modal visual map.
    Text prompt tokens act as Queries; fused T1 + secondary visual features are Keys/Values.
    """

    def __init__(self, text_dim=768, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        
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

        # 3. Run Language-to-Visual Cross-Attention
        # Output shape: [B, N_T, C]
        attn_out, _ = self.mha(query=q_tokens, key=kv_tokens, value=kv_tokens)

        # 4. Residual and Normalization
        aligned_queries = self.norm(attn_out + q_tokens) # Shape: [B, N_T, C]

        return aligned_queries
