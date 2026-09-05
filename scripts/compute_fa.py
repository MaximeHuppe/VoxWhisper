"""Compute DTI FA maps — thin CLI around voxwhisper.data.preprocess.fa."""
from __future__ import annotations

import argparse
import logging

from voxwhisper.util.config import load_config
from voxwhisper.data.preprocess.fa import compute_fa_cohort


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Compute DTI FA maps from HCP diffusion data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/voxdense.yaml")
    parser.add_argument("--subject", default=None, metavar="ID",
                        help="Process a single subject instead of the full cohort")
    parser.add_argument("--delete-raw", action="store_true", default=False,
                        help="Delete data.nii.gz after a successful FA write")
    parser.add_argument("--workers", type=int, default=4, metavar="N",
                        help="Number of parallel worker threads")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    delete_raw = args.delete_raw or bool(
        cfg.get("data", {}).get("download", {}).get("delete_raw_4d", False)
    )
    compute_fa_cohort(
        config=cfg,
        subject_filter=args.subject,
        delete_raw=delete_raw,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
