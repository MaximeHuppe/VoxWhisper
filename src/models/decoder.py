import torch
import torch.nn as nn
import torch.nn.functional as F

class StageVLFusionBlock(nn.Module):
    """
    Integrates the aligned text queries with visual features at a specific scale.
    Performs channel-wise modulation (gating) and projects the intermediate 
    segmentation masks using a spatial-semantic dot product.
    """
    def __init__(self, visual_channels, query_dim=128):
        super().__init__()
        self.visual_channels = visual_channels
        
        # Projects global queries to match scale-specific visual channels
        self.query_adapter = nn.Sequential(
            nn.Linear(query_dim, visual_channels),
            nn.ReLU(),
            nn.Linear(visual_channels, visual_channels)
        )

    def forward(self, visual_features, aligned_queries):
        # visual_features: Shape [B, C_vis, D, H, W]
        # aligned_queries:  Shape [B, N_T, C_query]
        B, C_vis, D, H, W = visual_features.shape
        N_T = aligned_queries.shape[1]

        # 1. Project aligned queries to match this scale's visual channel depth
        # [B, N_T, C_query] -> [B, N_T, C_vis]
        projected_queries = self.query_adapter(aligned_queries)

        # 2. Extract a Global Semantic Gating vector (mean pool over prompt sequence)
        # [B, N_T, C_vis] -> [B, C_vis]
        global_text = projected_queries.mean(dim=1)

        # 3. Channel-wise Modulation (Sigmoid Gating)
        # Reshape scale vector to [B, C_vis, 1, 1, 1] to allow broadcasting over 3D dimensions
        scale = torch.sigmoid(global_text).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        modulated_vis = visual_features * scale # Shape: [B, C_vis, D, H, W]

        # 4. Mask Generation via Batch Matrix Multiplication (Dot Product)
        # Flatten spatial dimensions to shape: [B, C_vis, D * H * W]
        flat_vis = modulated_vis.view(B, C_vis, D * H * W)

        # Compute dot product between each query token and each spatial voxel:
        # projected_queries: [B, N_T, C_vis]
        # flat_vis:          [B, C_vis, D * H * W]
        # [B, N_T, C_vis] x [B, C_vis, D * H * W] -> [B, N_T, D * H * W]
        mask_logits_flat = torch.bmm(projected_queries, flat_vis)

        # Reshape the flat masks back into a 3D grid per prompt token
        mask_logits = mask_logits_flat.view(B, N_T, D, H, W) # Shape: [B, N_T, D, H, W]

        return modulated_vis, mask_logits


class Decoder(nn.Module):
    """
    3D Upsampling Decoder with Hierarchical Vision-Language Fusion.
    Collects masks at each resolution scale to support Deep Supervision training.
    """
    def __init__(self, channels=[16, 32, 64, 128], query_dim=128):
        super().__init__()
        
        # List of skip channels from your T1 Encoder: [16, 32, 64]
        skip_channels = channels[:-1] # [16, 32, 64]
        
        # Re-verify channels backwards
        self.up_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        
        # We build the stages going from Bottleneck (128 channels) up to Stem (16 channels)
        # Stage 1: Bottleneck (128 channels) -> Out: 64 channels
        # Stage 2: 64 channels -> Out: 32 channels
        # Stage 3: 32 channels -> Out: 16 channels
        in_ch = channels[-1] # 128
        
        for skip_ch in reversed(skip_channels): # [64, 32, 16]
            # Standard 3D Convolution to merge upsampled features and skip connections
            conv_block = nn.Sequential(
                nn.Conv3d(in_ch + skip_ch, skip_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(skip_ch),
                nn.ReLU()
            )
            self.up_blocks.append(conv_block)
            
            # Stage-specific vision-language fusion block
            self.fusion_blocks.append(StageVLFusionBlock(visual_channels=skip_ch, query_dim=query_dim))
            
            in_ch = skip_ch # Next input is current output

    def forward(self, fused_bottleneck, skips, aligned_queries):
        # fused_bottleneck: Shape [B, 128, D_b, H_b, W_b] -> (Stage 3 Bottleneck output)
        # skips:            List of T1 skip connections: [Skip1 (16), Skip2 (32), Skip3 (64)]
        # aligned_queries:  Shape [B, N_T, 128]
        
        dec_features = fused_bottleneck
        stage_predictions = [] # List to hold intermediate masks for Deep Supervision

        # We reverse the skip list: [Skip3 (64), Skip2 (32), Skip1 (16)]
        skips_reversed = list(reversed(skips))

        for s in range(len(self.up_blocks)):
            skip = skips_reversed[s]
            
            # 1. Trilinear upsample previous resolution features to match the skip connection shape
            dec_features = F.interpolate(
                dec_features, 
                size=skip.shape[2:], 
                mode='trilinear', 
                align_corners=True
            )
            
            # 2. Concatenate upsampled features with T1 skip connection
            # dec_features shape: [B, in_ch + skip_ch, D, H, W]
            dec_features = torch.cat([dec_features, skip], dim=1)
            
            # 3. Blend channels using 3D Convolutions
            dec_features = self.up_blocks[s](dec_features) # Shape: [B, skip_ch, D, H, W]
            
            # 4. Apply Channel Modulation and project the intermediate 3D mask
            dec_features, mask_logits = self.fusion_blocks[s](dec_features, aligned_queries)
            
            # 5. Save intermediate logits for training with Deep Supervision Loss
            stage_predictions.append(mask_logits)

        # Returns a list of intermediate 3D probability volumes:
        # stage_predictions[0]: Shape [B, N_T, D//4, H//4, W//4] (Lowest res)
        # stage_predictions[1]: Shape [B, N_T, D//2, H//2, W//2] (Middle res)
        # stage_predictions[2]: Shape [B, N_T, D, H, W]         (Original resolution)
        return stage_predictions