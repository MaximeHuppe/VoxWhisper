"""Model contracts: encoder depth, PE grid, VoxDense and VoxWhisper forwards."""
from __future__ import annotations

import torch

from voxwhisper.models.attention import PositionalEncoding3D, bottleneck_spatial_size
from voxwhisper.models.encoder import Encoder
from voxwhisper.models.vox_dense import VoxDense
from voxwhisper.models.vox_whisper import VoxWhisper


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


def test_voxdense_from_config_pe_matches_patch_and_strides(tmp_config):
    tmp_config["data"]["patch"]["size"] = [16, 16, 16]
    model = VoxDense.from_config(tmp_config)
    expected = bottleneck_spatial_size(
        tmp_config["data"]["patch"]["size"], tmp_config["model"]["encoder"]["strides"]
    )
    pe = model.prompt_decoder.pos_encoder
    assert (pe.d_size, pe.h_size, pe.w_size) == expected


def test_voxdense_has_no_fa_modules(tmp_config):
    model = VoxDense.from_config(tmp_config)
    names = {name for name, _ in model.named_children()}
    assert "encoder" in names
    assert "secondary_encoder" not in names
    assert "cross_volume_attention" not in names
    assert "primary_encoder" not in names


def test_voxdense_forward_shape():
    model = VoxDense(
        channels=[8, 16],
        strides=[2],
        kernel_sizes=[3],
        paddings=[1],
        num_resblocks=[1],
        embed_dim=16,
        num_heads=2,
        text_dim=8,
        pe_size=(8, 8, 8),
    )
    n_prompts = 4
    volume = torch.zeros(1, 1, 16, 16, 16)
    text = torch.zeros(1, n_prompts, 8)
    preds = model(volume, text)
    assert len(preds) == 1
    assert preds[0].shape == (1, n_prompts, 16, 16, 16)


def test_stage_vl_fusion_logits_invariant_to_extra_prompts():
    """Shared mean-pool gating made logits depend on N_T; per-prompt gate must not."""
    from voxwhisper.models.decoder import StageVLFusionBlock

    block = StageVLFusionBlock(visual_channels=8, query_dim=8)
    torch.manual_seed(0)
    visual = torch.randn(1, 8, 2, 2, 2)
    q_shared = torch.randn(1, 2, 8)
    q_extra = torch.cat([q_shared, torch.randn(1, 3, 8)], dim=1)

    _, logits_small = block(visual, q_shared)
    _, logits_large = block(visual, q_extra)
    torch.testing.assert_close(logits_small, logits_large[:, :2], atol=1e-5, rtol=1e-5)


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


def test_voxwhisper_forward_shape():
    model = VoxWhisper(
        channels=[8, 16],
        strides=[2],
        kernel_sizes=[3],
        paddings=[1],
        num_resblocks=[1],
        embed_dim=16,
        num_heads=2,
        text_dim=8,
        pe_size=(8, 8, 8),
    )
    n_prompts = 4
    primary = torch.zeros(1, 1, 16, 16, 16)
    secondary = torch.zeros(1, 1, 16, 16, 16)
    text = torch.zeros(1, n_prompts, 8)
    preds = model(primary, secondary, text)
    assert len(preds) == 1
    assert preds[0].shape == (1, n_prompts, 16, 16, 16)


def test_positional_encoding_shape():
    pe = PositionalEncoding3D(embed_dim=4, d_size=4, h_size=4, w_size=4)
    tokens = torch.zeros(2, 64, 4)
    out = pe(tokens, spatial_size=(4, 4, 4))
    assert out.shape == tokens.shape
