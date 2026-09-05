"""Stage dispatch: Phase 1 (VoxDense / T1 dense) vs Phase 2 (VoxWhisper / T1+FA nerves).

The same CLIs (``scripts/preprocess.py``, ``scripts/train.py``, ``scripts/evaluate.py``)
and the same training loop read ``model.name`` (and the volume/mask keys) to pick
the cohort, dataset, and forward signature.
"""
from __future__ import annotations

from typing import Any, Mapping

from voxwhisper.util.config import SECONDARY_MODALITY

STAGE_DENSE = "dense"
STAGE_NERVE = "nerve"


def stage_id(config: Mapping[str, Any]) -> str:
    """``dense`` (Phase 1) or ``nerve`` (Phase 2)."""
    name = str(config.get("model", {}).get("name", "VoxDense")).lower()
    if name in {"voxwhisper", "nerve"}:
        return STAGE_NERVE
    return STAGE_DENSE


def uses_secondary(config: Mapping[str, Any]) -> bool:
    """True when this stage trains a dual T1+FA model."""
    volumes = config.get("data", {}).get("volumes", {})
    if SECONDARY_MODALITY in volumes:
        return True
    return stage_id(config) == STAGE_NERVE


def mask_kind(config: Mapping[str, Any]) -> str:
    """``named`` = per-structure NIfTIs (nerves); ``dense`` = wmparc collapse."""
    masks = config.get("data", {}).get("masks", {})
    volumes = config.get("data", {}).get("volumes", {})
    if masks.get("source"):
        return "named"
    if volumes.get("wmparc"):
        return "dense"
    raise ValueError(
        "Config must set data.masks.source (nerve NIfTIs) or data.volumes.wmparc "
        "(FreeSurfer dense labels)"
    )


def cohort_name(config: Mapping[str, Any]) -> str:
    """Which ``subject_split`` list this stage processes and trains on."""
    return "nerve" if stage_id(config) == STAGE_NERVE else "pretrain"


def build_model(config: Mapping[str, Any]):
    if stage_id(config) == STAGE_NERVE:
        from voxwhisper.models import VoxWhisper
        return VoxWhisper.from_config(config)
    from voxwhisper.models import VoxDense
    return VoxDense.from_config(config)


def build_dataset(config: Mapping[str, Any], **kwargs):
    if uses_secondary(config):
        from voxwhisper.data.dataset import VoxWhisperDataset
        return VoxWhisperDataset(config, **kwargs)
    from voxwhisper.data.dataset import VoxDenseDataset
    return VoxDenseDataset(config, **kwargs)


def unpack_batch(batch):
    """Normalise a loader batch to ``(primary, secondary_or_none, text, gt)``."""
    if len(batch) == 3:
        volume, text, gt = batch
        return volume, None, text, gt
    if len(batch) == 4:
        primary, secondary, text, gt = batch
        return primary, secondary, text, gt
    raise ValueError(f"Expected 3- or 4-tuple batch, got len={len(batch)}")


def forward_model(model, primary, secondary, text):
    if secondary is None:
        return model(primary, text)
    return model(primary, secondary, text)
