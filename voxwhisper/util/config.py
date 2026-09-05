"""Shared configuration loader for VoxWhisper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

PathLike = Union[str, Path]

# T1 is the Phase 1 (VoxDense) volume. FA is only used by Phase 2 VoxWhisper.
PRIMARY_MODALITY = "t1"
SECONDARY_MODALITY = "fa"
DEFAULT_CONFIG = "config/voxdense.yaml"


def get_project_root() -> Path:
    """Return the repository root (parent of the ``voxwhisper`` package)."""
    # voxwhisper/util/config.py → util → voxwhisper → repo
    return Path(__file__).resolve().parents[2]


def resolve_path(config: Mapping[str, Any], key_path: str) -> Path:
    """Resolve a config path key (relative to project root) to an absolute Path.

    Example: ``resolve_path(cfg, "data.paths.processed")``
    """
    node: Any = config
    for key in key_path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            raise KeyError(f"Missing config key: {key_path}")
        node = node[key]
    path = Path(node)
    if path.is_absolute():
        return path
    return get_project_root() / path


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Optional[PathLike] = None) -> dict:
    """Load a YAML config file. Defaults to config/voxdense.yaml."""
    if path is None:
        config_path = get_project_root() / DEFAULT_CONFIG
    else:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = get_project_root() / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config)}")

    _apply_structures(config)
    return config


def load_structures(config: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Load structure manifest; returns None when not configured."""
    rel = config.get("data", {}).get("masks", {}).get("structures")
    if not rel:
        return None

    path = Path(rel)
    if not path.is_absolute():
        path = get_project_root() / path

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items = sorted(raw.items(), key=lambda kv: kv[1]["label"])
    foreground = [(name, v["label"]) for name, v in items if v["label"] != 0]
    return {
        "prompts": [v["prompt"] for _, v in items],
        "structure_names": [name for name, _ in items],
        "positive_labels": [label for _, label in foreground],
        "foreground": foreground,
    }


def _apply_structures(config: dict[str, Any]) -> None:
    structs = load_structures(config)
    if not structs:
        return
    config.setdefault("data", {})["prompts"] = structs["prompts"]
    config["data"]["structure_names"] = structs["structure_names"]
    config["data"].setdefault("patch", {})["positive_labels"] = structs["positive_labels"]
