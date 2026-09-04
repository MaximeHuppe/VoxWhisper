"""Train VoxWhisper — thin CLI wrapping voxwhisper.training.loop."""
from __future__ import annotations

import argparse

from voxwhisper.config import load_config
from voxwhisper.training.loop import train_model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train VoxWhisper")
    parser.add_argument(
        "--config",
        default="config/best_config.yaml",
        help="Path to YAML config (relative to project root or absolute)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Use an existing run directory (with or without --resume)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from vox_whisper_latest.pt in the newest run",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    train_model(cfg, resume=args.resume, run_dir=args.run_dir, config_path=args.config)


if __name__ == "__main__":
    main()
