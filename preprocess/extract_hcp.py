# preprocess/extract_hcp.py
"""Download HCP structural volumes listed in config (T1/T2 by default)."""
from __future__ import annotations

import glob
import os
import sys

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm

# Allow imports from project root when run as a script
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


class S3TransferProgressBar:
    """Dynamic progress bar hook for boto3 s3 transfers."""

    def __init__(self, filename, size_in_bytes):
        self._filename = filename
        self._size = size_in_bytes
        self._seen_so_far = 0
        self._pbar = tqdm(
            total=self._size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {os.path.basename(filename)}",
            leave=True,
        )

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        self._pbar.update(bytes_amount)
        if self._seen_so_far >= self._size:
            self._pbar.close()


def get_subjects_from_masks(mask_dir):
    subjects = []
    if not os.path.exists(mask_dir):
        return []

    for item in os.listdir(mask_dir):
        item_path = os.path.join(mask_dir, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 6:
            subjects.append(item)

    recursive_pattern = os.path.join(mask_dir, "**/*.nii.gz")
    mask_files = glob.glob(recursive_pattern, recursive=True)

    for filepath in mask_files:
        path_segments = filepath.split(os.sep)
        for segment in path_segments:
            if segment.isdigit() and len(segment) == 6:
                subjects.append(segment)

        filename = os.path.basename(filepath)
        potential_id = filename.split(".")[0].split("_")[0]
        if potential_id.isdigit() and len(potential_id) == 6:
            subjects.append(potential_id)

    return sorted(list(set(subjects)))


def download_file_with_progress(s3_client, bucket, key, local_path):
    """Fetch file size, initialize a progress bar, and download in parallel."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        response = s3_client.head_object(
            Bucket=bucket,
            Key=key,
            RequestPayer="requester",
        )
        file_size = response["ContentLength"]
        progress_callback = S3TransferProgressBar(local_path, file_size)

        s3_client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=local_path,
            ExtraArgs={"RequestPayer": "requester"},
            Config=S3_TRANSFER_CONFIG,
            Callback=progress_callback,
        )
    except ClientError as e:
        print(f"Error downloading {key}: {e}")


def download_hcp_dataset(config):
    mask_dir = resolve_path(config, "data.paths.raw_masks")
    target_dir = resolve_path(config, "data.paths.raw")
    ensure_dir(target_dir)

    download_cfg = config["data"]["download"]
    volumes_cfg = config["data"]["volumes"]
    modalities = download_cfg.get("modalities", ["t1", "t2"])
    bucket_name = download_cfg["bucket"]
    dataset_prefix = download_cfg["dataset_prefix"]

    subjects = get_subjects_from_masks(str(mask_dir))
    if not subjects:
        print(f"Error: No valid subject masks found in {mask_dir}")
        print("Please download and extract the OpticNerveSeg masks to that directory first.")
        sys.exit(1)

    print(f"Detected {len(subjects)} subjects with valid masks in raw_masks.")

    if download_cfg.get("limit_subjects", False):
        limit_count = int(download_cfg.get("limit_count", 10))
        subjects = subjects[:limit_count]
        print(f"Limiting download to the first {len(subjects)} subjects.")

    try:
        s3 = boto3.client("s3", region_name="us-east-1")
        # Validate credentials early
        s3.list_buckets()
    except NoCredentialsError:
        print(
            "Error: AWS credentials not found.\n"
            "Configure them via ~/.aws/credentials or AWS_ACCESS_KEY_ID / "
            "AWS_SECRET_ACCESS_KEY environment variables."
        )
        sys.exit(1)
    except ClientError:
        # list_buckets may fail for restricted keys; still attempt downloads
        s3 = boto3.client("s3", region_name="us-east-1")

    for sub in subjects:
        print("==========================================")
        print(f"Downloading Subject: {sub}")
        print("==========================================")

        for modality in modalities:
            if modality not in volumes_cfg:
                print(f"Warning: modality '{modality}' missing from data.volumes; skipping.")
                continue

            filename = volumes_cfg[modality]["filename"]
            s3_key = f"{dataset_prefix}/{sub}/T1w/{filename}"
            local_path = os.path.join(str(target_dir), sub, "T1w", filename)
            download_file_with_progress(s3, bucket_name, s3_key, local_path)

    print("\nMultimodal download process completed.")


if __name__ == "__main__":
    args = parse_config_args(description="Download HCP T1/T2 volumes")
    cfg = load_config(args.config)
    download_hcp_dataset(cfg)
