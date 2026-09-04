"""Phase 1 / Phase 2 dispatch in ``voxwhisper.util.stage``."""
from __future__ import annotations

from voxwhisper.models import VoxDense, VoxWhisper
from voxwhisper.util.config import get_project_root
from voxwhisper.util.stage import (
    build_model,
    cohort_name,
    mask_kind,
    stage_id,
    unpack_batch,
    uses_secondary,
)


def test_get_project_root_is_repo():
    root = get_project_root()
    assert (root / "voxwhisper" / "util" / "config.py").is_file()
    assert (root / "pyproject.toml").is_file()


def test_dense_stage_from_voxdense_config(tmp_config):
    assert stage_id(tmp_config) == "dense"
    assert uses_secondary(tmp_config) is False
    assert mask_kind(tmp_config) == "dense"
    assert cohort_name(tmp_config) == "pretrain"
    assert isinstance(build_model(tmp_config), VoxDense)


def test_nerve_stage_from_whisper_config(tmp_whisper_config):
    assert stage_id(tmp_whisper_config) == "nerve"
    assert uses_secondary(tmp_whisper_config) is True
    assert mask_kind(tmp_whisper_config) == "named"
    assert cohort_name(tmp_whisper_config) == "nerve"
    assert isinstance(build_model(tmp_whisper_config), VoxWhisper)


def test_unpack_batch_3_and_4_tuples():
    vol, text, gt = object(), object(), object()
    primary, secondary, t, g = unpack_batch((vol, text, gt))
    assert primary is vol and secondary is None and t is text and g is gt

    fa = object()
    primary, secondary, t, g = unpack_batch((vol, fa, text, gt))
    assert primary is vol and secondary is fa


def test_mask_kind_prefers_named_source_over_wmparc(tmp_whisper_config):
    assert "wmparc" in tmp_whisper_config["data"]["volumes"]
    assert mask_kind(tmp_whisper_config) == "named"
