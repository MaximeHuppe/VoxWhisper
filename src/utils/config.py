"""Shared configuration loader for VoxWhisper."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

PathLike = Union[str, Path]


def get_project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: Optional[PathLike] = None) -> dict:
    """
    Load a YAML config file.

    Paths in the config are left as relative strings; use ``resolve_path``
    to convert them to absolute paths under the project root.
    """
    if path is None:
        config_path = get_project_root() / "config" / "default.yaml"
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

    return config


def get_nested(config: Mapping[str, Any], key_path: str) -> Any:
    """Fetch a nested value using dot-separated keys, e.g. ``data.paths.raw``."""
    node: Any = config
    for key in key_path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            raise KeyError(f"Missing config key: {key_path}")
        node = node[key]
    return node


def resolve_path(config: Mapping[str, Any], key_path: str) -> Path:
    """
    Resolve a config path key (relative to project root) to an absolute Path.

    Example: ``resolve_path(cfg, "data.paths.processed")``
    """
    value = get_nested(config, key_path)
    path = Path(value)
    if path.is_absolute():
        return path
    return get_project_root() / path


def parse_config_args(
    argv: Optional[Sequence[str]] = None,
    description: str = "VoxWhisper",
) -> argparse.Namespace:
    """Parse a standard ``--config`` CLI argument and return the namespace."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config (relative to project root or absolute)",
    )
    return parser.parse_args(argv)


def active_modality_keys(config: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(primary, secondary)`` modality keys from config."""
    modalities = get_nested(config, "data.modalities")
    return modalities["primary"], modalities["secondary"]


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
