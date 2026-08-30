# src/models/vox_whisper.py
import torch
import torch.nn as nn
from .encoder import Encoder
from .attention import CrossVolumeAttention, PromptDecoder
from .decoder import Decoder

class VoxWhisper(nn.Module):
    """
    VoxWhisper: A 3D Multi-Modal, Language-Grounded Volumetric Segmentation Model.
    Fuses unregistered T1 and Diffusion MRIs, aligns them with clinical prompts,
    and reconstructs prompt-conditioned segmentation masks in the T1 coordinate space.
    """
    def __init__(self, 
                 input_channels=1, 
                 text_dim=768, 
                 embed_dim=128, 
                 channels=[16, 32, 64, 128],
                 num_heads=4):
        super().__init__()
        
        # 1. Visual Encoders (T1 Branch & Diffusion Branch)
        # T1 Encoder captures skip connections to reconstruct high-resolution features
        self.t1_encoder = Encoder(
            input_channels=input_channels, 
            channels=channels, 
            strides=[2, 2, 2], 
            num_resblocks=[1, 1, 1]
        )
        
        # Diffusion Encoder processes the guidance volume
        self.diff_encoder = Encoder(
            input_channels=input_channels, 
            channels=channels, 
            strides=[2, 2, 2], 
            num_resblocks=[1, 1, 1]
        )

        # 2. Visual-to-Visual Alignment (Fuses Diffusion into T1 coordinate space)
        self.cross_volume_attention = CrossVolumeAttention(embed_dim=embed_dim, num_heads=num_heads)

        # 3. Language-to-Visual Alignment (Fuses text prompts with the consolidated visual map)
        self.prompt_decoder = PromptDecoder(text_dim=text_dim, embed_dim=embed_dim, num_heads=num_heads)

        # 4. Hierarchical Decoder with Channel Modulation & Deep Supervision
        self.decoder = Decoder(channels=channels, query_dim=embed_dim)

    def forward(self, t1_volume, diff_volume, text_embeddings):
        # t1_volume:       Shape [B, 1, D_t1, H_t1, W_t1]
        # diff_volume:     Shape [B, 1, D_diff, H_diff, W_diff]  (unregistered)
        # text_embeddings: Shape [B, N_T, 768] (cached text embeddings)

        # ==========================================
        # STEP 1: Feature Extraction (Encoders)
        # ==========================================
        # T1 path captures skip connections
        t1_bottleneck, skip3, skip2, skip1 = self.t1_encoder(t1_volume)
        # t1_bottleneck shape: [B, 128, D_t1//8, H_t1//8, W_t1//8]
        
        # Diffusion path handles independent extraction
        diff_bottleneck, _, _, _ = self.diff_encoder(diff_volume)
        # diff_bottleneck shape: [B, 128, D_diff//8, H_diff//8, W_diff//8]

        # ==========================================
        # STEP 2: Spatial Alignment (Cross-Volume MHA)
        # ==========================================
        # Projects Diffusion features to align with the T1 bottleneck layout
        fused_visual_map = self.cross_volume_attention(
            t1_features=t1_bottleneck, 
            diff_features=diff_bottleneck
        ) # Shape: [B, 128, D_t1//8, H_t1//8, W_t1//8]

        # ==========================================
        # STEP 3: Semantic Alignment (Prompt Decoder MHA)
        # ==========================================
        # Aligns text tokens to the unified spatial visual feature map
        aligned_queries = self.prompt_decoder(
            text_embeddings=text_embeddings, 
            fused_visual_features=fused_visual_map
        ) # Shape: [B, N_T, 128]

        # ==========================================
        # STEP 4: Reconstruction (Multi-Scale Decoder)
        # ==========================================
        skips = [skip1, skip2, skip3]
        
        # Decoder performs channel-wise scaling and outputs multi-resolution masks
        stage_predictions = self.decoder(
            fused_bottleneck=fused_visual_map, 
            skips=skips, 
            aligned_queries=aligned_queries
        )
        
        # Returns list of 3D mask logits for deep supervision loss computation:
        # stage_predictions[0]: Lowest resolution [B, N_T, D_t1//4, H_t1//4, W_t1//4]
        # stage_predictions[1]: Middle resolution [B, N_T, D_t1//2, H_t1//2, W_t1//2]
        # stage_predictions[2]: Final resolution  [B, N_T, D_t1, H_t1, W_t1]
        return stage_predictions