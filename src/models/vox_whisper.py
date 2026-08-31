# src/models/vox_whisper.py
import torch
import torch.nn as nn
from .encoder import Encoder
from .attention import CrossVolumeAttention, PromptDecoder
from .decoder import Decoder


class VoxWhisper(nn.Module):
    """
    VoxWhisper: A 3D Multi-Modal, Language-Grounded Volumetric Segmentation Model.
    Fuses unregistered T1 and T2 structural MRIs, aligns them with clinical prompts,
    and reconstructs prompt-conditioned segmentation masks in the T1 coordinate space.
    """

    def __init__(
        self,
        input_channels=1,
        text_dim=768,
        embed_dim=128,
        channels=None,
        strides=None,
        kernel_sizes=None,
        paddings=None,
        num_resblocks=None,
        num_heads=4,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64, 128]
        if strides is None:
            strides = [2, 2, 2]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if paddings is None:
            paddings = [1, 1, 1]
        if num_resblocks is None:
            num_resblocks = [1, 1, 1]

        # 1. Visual Encoders (T1 Branch & T2 Branch)
        encoder_kwargs = dict(
            input_channels=input_channels,
            channels=channels,
            strides=strides,
            kernel_sizes=kernel_sizes,
            paddings=paddings,
            num_resblocks=num_resblocks,
        )
        self.t1_encoder = Encoder(**encoder_kwargs)
        self.t2_encoder = Encoder(**encoder_kwargs)

        # 2. Visual-to-Visual Alignment (Fuses T2 into T1 coordinate space)
        self.cross_volume_attention = CrossVolumeAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )

        # 3. Language-to-Visual Alignment
        self.prompt_decoder = PromptDecoder(
            text_dim=text_dim, embed_dim=embed_dim, num_heads=num_heads
        )

        # 4. Hierarchical Decoder with Channel Modulation & Deep Supervision
        self.decoder = Decoder(channels=channels, query_dim=embed_dim)

    @classmethod
    def from_config(cls, config):
        """Construct the model from the ``model`` section of a YAML config."""
        model_cfg = config["model"]
        enc = model_cfg["encoder"]
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
        )

    def forward(self, t1_volume, t2_volume, text_embeddings):
        # t1_volume:       Shape [B, 1, D_t1, H_t1, W_t1]
        # t2_volume:       Shape [B, 1, D_t2, H_t2, W_t2]  (may be unregistered)
        # text_embeddings: Shape [B, N_T, text_dim]

        # ==========================================
        # STEP 1: Feature Extraction (Encoders)
        # ==========================================
        # T1 path captures skip connections
        t1_bottleneck, skip3, skip2, skip1 = self.t1_encoder(t1_volume)
        # t1_bottleneck shape: [B, 128, D_t1//8, H_t1//8, W_t1//8]

        # T2 path does not capture skip connections
        t2_bottleneck, _, _, _ = self.t2_encoder(t2_volume)
        # t2_bottleneck shape: [B, 128, D_t2//8, H_t2//8, W_t2//8]

        # ==========================================
        # STEP 2: Spatial Alignment (Cross-Volume MHA)
        # ==========================================
        # Projects Diffusion features to align with the T1 bottleneck layout
        fused_visual_map = self.cross_volume_attention(
            t1_features=t1_bottleneck,
            secondary_features=t2_bottleneck,
        )
        # fused_visual_map shape: [B, 128, D_t1//8, H_t1//8, W_t1//8]

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
            aligned_queries=aligned_queries,
        )
        
        # Returns list of 3D mask logits for deep supervision loss computation:
        # stage_predictions[0]: Lowest resolution [B, N_T, D_t1//4, H_t1//4, W_t1//4]
        # stage_predictions[1]: Middle resolution [B, N_T, D_t1//2, H_t1//2, W_t1//2]
        # stage_predictions[2]: Final resolution  [B, N_T, D_t1, H_t1, W_t1]
        return stage_predictions
