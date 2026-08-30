import torch
import torch.nn as nn

# 1. Decoupled, reusable Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Define layers with weights here
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # Apply the layers here
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Residual addition using native addition (replaces nn.add)
        out = out + x 
        out = self.relu2(out)
        return out


# 2. Stage Block (Downsamples first, then repeats Residual Blocks)
class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_resblocks):
        super().__init__()
        # Transition/Downsampling block
        self.transition = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU()
        )
        # Dynamic stacking of N residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(out_channels) for _ in range(num_resblocks)
        ])

    def forward(self, x):
        out = self.transition(x)
        out = self.res_blocks(out)
        return out


# 3. Dynamic Encoder
class Encoder(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 channels=[16, 32, 64, 128],  # Channels across stages
                 strides=[2, 2, 2],            # Strides per stage
                 kernel_sizes=[3, 3, 3],       # Kernels per stage
                 paddings=[1, 1, 1],           # Paddings per stage
                 num_resblocks=[1, 1, 1]):     # Number of resblocks per stage
        
        super().__init__()
        
        assert len(strides) == len(channels) - 1, "Verify your channel and stride list lengths match"

        # Stem (Stage 0 - preserves initial high resolution)
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(channels[0]),
            nn.ReLU(),
            ResidualBlock(channels[0])
        )

        # We use nn.ModuleList so PyTorch tracks the weights of our dynamic stages
        self.stages = nn.ModuleList()
        
        for i in range(len(strides)):
            stage = EncoderStage(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                num_resblocks=num_resblocks[i]
            )
            self.stages.append(stage)

    def forward(self, x):
        # 1. Run Stem (highest resolution)
        skip1 = self.stem(x)
        
        # 2. Run sequential stages while capturing skip connections
        skip2 = self.stages[0](skip1)
        skip3 = self.stages[1](skip2)
        
        # 3. Final bottleneck output
        F_t1 = self.stages[2](skip3)
        
        return F_t1, skip3, skip2, skip1