# preprocess/extract_hcp.py
"""Download HCP structural and diffusion volumes for subjects already in the raw folder."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import (  # noqa: E402
    ensure_dir,
    load_config,
    parse_config_args,
    resolve_path,
)

S3_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=1024 * 1024 * 15,
    max_concurrency=10,
    multipart_chunksize=1024 * 1024 * 15,
    use_threads=True,
)

DIFFUSION_FILES = ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz")

_print_lock = Lock()


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def get_subjects_from_raw(raw_dir: str | Path) -> list[str]:
    """Return 6-digit subject IDs that already have a folder under raw_dir."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    return sorted(
        d for d in os.listdir(raw_dir)
        if (raw_dir / d).is_dir() and d.isdigit() and len(d) == 6
    )


# ---------------------------------------------------------------------------
# Per-file download
# ---------------------------------------------------------------------------

def _make_s3_client():
    return boto3.client("s3", region_name="us-east-1")


def download_file(s3_client, bucket: str, key: str, local_path: str) -> bool:
    """Download one S3 object; skip silently if the local file already exists."""
    if Path(local_path).exists():
        return True

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key, RequestPayer="requester")
        size = head["ContentLength"]

        pbar = tqdm(
            total=size, unit="B", unit_scale=True,
            desc=os.path.basename(local_path), leave=False,
        )
        s3_client.download_file(
            Bucket=bucket, Key=key, Filename=local_path,
            ExtraArgs={"RequestPayer": "requester"},
            Config=S3_TRANSFER_CONFIG,
            Callback=lambda n: pbar.update(n),
        )
        pbar.close()
        return True
    except ClientError as exc:
        with _print_lock:
            tqdm.write(f"  [WARN] {key}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Per-subject download (runs in worker thread — creates its own boto3 client)
# ---------------------------------------------------------------------------

def download_subject(
    bucket: str,
    prefix: str,
    sub: str,
    target_dir: str,
    modalities: list[str],
    volumes_cfg: dict,
) -> tuple[str, list[str]]:
    """Download all requested modalities for one subject. Returns (sub, failed_files)."""
    s3 = _make_s3_client()
    failed: list[str] = []

    for modality in modalities:
        if modality == "diffusion":
            for fname in DIFFUSION_FILES:
                key = f"{prefix}/{sub}/T1w/Diffusion/{fname}"
                local = os.path.join(target_dir, sub, "Diffusion", fname)
                if not download_file(s3, bucket, key, local):
                    failed.append(fname)
        else:
            if modality not in volumes_cfg:
                with _print_lock:
                    tqdm.write(f"  [WARN] modality '{modality}' not in config; skipping.")
                continue
            fname = volumes_cfg[modality]["filename"]
            key = f"{prefix}/{sub}/T1w/{fname}"
            local = os.path.join(target_dir, sub, fname)
            if not download_file(s3, bucket, key, local):
                failed.append(fname)

    return sub, failed


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def download_hcp_dataset(config):
    target_dir = str(ensure_dir(resolve_path(config, "data.paths.raw")))

    download_cfg = config["data"]["download"]
    volumes_cfg  = config["data"].get("volumes", {})
    modalities   = download_cfg.get("modalities", ["diffusion"])
    bucket       = download_cfg["bucket"]
    prefix       = download_cfg["dataset_prefix"]
    max_workers  = int(download_cfg.get("max_workers", 8))

    subjects = get_subjects_from_raw(target_dir)
    if not subjects:
        print(f"No subject folders found in {target_dir}. Nothing to download.")
        sys.exit(0)

    if download_cfg.get("limit_subjects"):
        subjects = subjects[: int(download_cfg.get("limit_count", 10))]

    print(f"Subjects to process : {len(subjects)}")
    print(f"Modalities          : {modalities}")
    print(f"Parallel workers    : {max_workers}")

    # Validate AWS credentials once before spawning workers
    try:
        boto3.client("s3", region_name="us-east-1").list_buckets()
    except NoCredentialsError:
        print(
            "Error: AWS credentials not found.\n"
            "Configure them via ~/.aws/credentials or the "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables."
        )
        sys.exit(1)
    except ClientError:
        pass  # Restricted key — still valid, proceed

    failed_subjects: dict[str, list[str]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                download_subject,
                bucket, prefix, sub, target_dir, modalities, volumes_cfg,
            ): sub
            for sub in subjects
        }
        with tqdm(total=len(subjects), desc="Subjects", unit="sub") as pbar:
            for future in as_completed(futures):
                sub, failed = future.result()
                if failed:
                    failed_subjects[sub] = failed
                pbar.update(1)
                pbar.set_postfix(failed=len(failed_subjects))

    if failed_subjects:
        print(f"\nWarning: {len(failed_subjects)} subject(s) had download errors:")
        for sub, files in failed_subjects.items():
            print(f"  {sub}: {files}")
    else:
        print("\nAll downloads completed successfully.")


if __name__ == "__main__":
    args = parse_config_args(description="Download HCP structural/diffusion volumes")
    cfg = load_config(args.config)
    download_hcp_dataset(cfg)
