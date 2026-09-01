"""Model contracts: dynamic encoder depth, PE grid, secondary channels, legacy ckpts."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.attention import PositionalEncoding3D, bottleneck_spatial_size
from src.models.encoder import Encoder
from src.models.vox_whisper import VoxWhisper
from src.utils.checkpoint import remap_legacy_state_dict
from src.utils.config import load_config


def test_bottleneck_spatial_size_from_patch_and_strides():
    assert bottleneck_spatial_size((128, 128, 128), [2, 2, 2]) == (16, 16, 16)
    assert bottleneck_spatial_size((32, 32, 32), [2, 2, 2]) == (4, 4, 4)
    assert bottleneck_spatial_size((16, 16, 16), [2]) == (8, 8, 8)


def test_encoder_returns_bottleneck_and_skip_list():
    encoder = Encoder(
        input_channels=1,
        channels=[8, 16, 24, 32],
        strides=[2, 2, 2],
        kernel_sizes=[3, 3, 3],
        paddings=[1, 1, 1],
        num_resblocks=[1, 1, 1],
    )
    bottleneck, skips = encoder(torch.zeros(1, 1, 16, 16, 16))
    assert bottleneck.shape == (1, 32, 2, 2, 2)
    assert len(skips) == 3
    assert skips[0].shape == (1, 8, 16, 16, 16)
    assert skips[1].shape == (1, 16, 8, 8, 8)
    assert skips[2].shape == (1, 24, 4, 4, 4)


def test_from_config_pe_matches_patch_and_strides():
    cfg = load_config()
    model = VoxWhisper.from_config(cfg)
    expected = bottleneck_spatial_size(
        cfg["data"]["patch"]["size"], cfg["model"]["encoder"]["strides"]
    )
    pe = model.cross_volume_attention.pos_primary
    assert (pe.d_size, pe.h_size, pe.w_size) == expected


def test_secondary_encoder_can_use_different_input_channels():
    model = VoxWhisper(
        input_channels=1,
        secondary_input_channels=3,
        channels=[8, 16],
        strides=[2],
        kernel_sizes=[3],
        paddings=[1],
        num_resblocks=[1],
        pe_size=(8, 8, 8),
        text_dim=8,
        embed_dim=16,
        num_heads=2,
    )
    assert model.primary_encoder.stem[0].in_channels == 1
    assert model.secondary_encoder.stem[0].in_channels == 3

    primary = torch.zeros(1, 1, 16, 16, 16)
    secondary = torch.zeros(1, 3, 16, 16, 16)
    text = torch.zeros(1, 2, 8)
    preds = model(primary, secondary, text)
    assert len(preds) == 1
    assert preds[0].shape == (1, 2, 16, 16, 16)


def test_positional_encoding_interpolates_when_grid_differs():
    pe = PositionalEncoding3D(embed_dim=4, d_size=4, h_size=4, w_size=4)
    tokens = torch.zeros(2, 8, 4)
    out = pe(tokens, spatial_size=(2, 2, 2))
    assert out.shape == tokens.shape


def test_remap_legacy_encoder_and_pe_keys():
    legacy = {
        "t1_encoder.stem.0.weight": torch.zeros(1),
        "t2_encoder.stem.0.weight": torch.zeros(1),
        "cross_volume_attention.pos_t1.pos_d": torch.zeros(1),
        "prompt_decoder.text_projection.weight": torch.zeros(1),
    }
    remapped = remap_legacy_state_dict(legacy)
    assert "primary_encoder.stem.0.weight" in remapped
    assert "secondary_encoder.stem.0.weight" in remapped
    assert "cross_volume_attention.pos_primary.pos_d" in remapped
    assert "prompt_decoder.text_projection.weight" in remapped
    assert "t1_encoder.stem.0.weight" not in remapped


if __name__ == "__main__":
    tests = [
        test_bottleneck_spatial_size_from_patch_and_strides,
        test_encoder_returns_bottleneck_and_skip_list,
        test_from_config_pe_matches_patch_and_strides,
        test_secondary_encoder_can_use_different_input_channels,
        test_positional_encoding_interpolates_when_grid_differs,
        test_remap_legacy_encoder_and_pe_keys,
    ]
    for test_fn in tests:
        test_fn()
        print(f"ok {test_fn.__name__}")
    print(f"All {len(tests)} tests passed.")
