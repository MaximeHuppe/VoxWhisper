"""3D convolutional encoder for VoxWhisper.

Architecture
------------
Stem (stride-1) → N downsampling EncoderStages → bottleneck.

Each ``EncoderStage`` applies one strided convolution (the transition) followed
by ``num_resblocks`` identity ``ResidualBlock`` layers. The stem and every
intermediate stage output are returned as skip connections for the decoder.

Forward output
--------------
  bottleneck : Tensor [B, channels[-1], D_b, H_b, W_b]
  skips       : list of Tensors, one per stage *including* the stem output,
                ordered from shallowest (stem) to deepest (last down-stage).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two-layer 3×3×3 residual block with InstanceNorm (post-activation).

    Both convolutions preserve spatial resolution and channel count.  The
    identity shortcut is added *before* the final ReLU, following the original
    ResNet convention (He et al., 2016).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu2(out + x)


class EncoderStage(nn.Module):
    """One downsampling stage: strided transition conv → N residual blocks.

    Parameters
    ----------
    in_channels, out_channels:
        Channel widths before and after the strided convolution.
    kernel_size, stride, padding:
        Geometry of the transition (downsampling) convolution.
    num_resblocks:
        How many identity ``ResidualBlock`` layers follow the transition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        num_resblocks: int,
    ) -> None:
        super().__init__()
        self.transition = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(out_channels) for _ in range(num_resblocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_blocks(self.transition(x))


class Encoder(nn.Module):
    """Dynamic 3D encoder with configurable depth and channel widths.

    Parameters
    ----------
    input_channels:
        Number of input image channels (typically 1 for a single MRI modality).
    channels:
        Channel counts at each stage.  ``channels[0]`` is the stem output;
        ``channels[-1]`` is the bottleneck.  Must have length ``len(strides) + 1``.
    strides, kernel_sizes, paddings, num_resblocks:
        Per-stage geometry lists.  All must have length ``len(channels) - 1``.

    Forward outputs
    ---------------
    bottleneck : Tensor  [B, channels[-1], D_b, H_b, W_b]
    skips      : list of Tensors ordered shallowest → deepest
                 [stem_out, stage0_out, stage1_out, ...] (excludes bottleneck).
    """

    def __init__(
        self,
        input_channels: int = 1,
        channels: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        paddings: Optional[List[int]] = None,
        num_resblocks: Optional[List[int]] = None,
    ) -> None:
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

        n_stages = len(strides)
        if len(channels) != n_stages + 1:
            raise ValueError(
                f"len(channels)={len(channels)} must equal len(strides)+1={n_stages + 1}"
            )

        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            ResidualBlock(channels[0]),
        )

        self.stages = nn.ModuleList(
            EncoderStage(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                num_resblocks=num_resblocks[i],
            )
            for i in range(n_stages)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []

        x = self.stem(x)
        skips.append(x)

        *down_stages, bottleneck_stage = self.stages
        for stage in down_stages:
            x = stage(x)
            skips.append(x)

        bottleneck = bottleneck_stage(x)
        return bottleneck, skips
