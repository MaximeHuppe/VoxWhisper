#!/bin/bash
# preprocess/extract_hcp.sh
# Downloads specific HCP subject datasets (T1 and Diffusion) via AWS CLI

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: aws-cli is not installed. Run 'sudo apt install awscli' first."
    exit 1
fi

# Subject IDs to download (modify this list based on your study needs)
SUBJECTS=("100307" "100408")
TARGET_DIR="../data/raw"

echo "Starting download of HCP subjects to: ${TARGET_DIR}"

for sub in "${SUBJECTS[@]}"; do
    echo "=========================================="
    echo "Downloading Subject: ${sub}"
    echo "=========================================="

    # Create local directories
    mkdir -p "${TARGET_DIR}/${sub}/T1w"
    mkdir -p "${TARGET_DIR}/${sub}/T1w/Diffusion"

    # 1. Download Preprocessed High-Res T1 MRI
    echo "Downloading T1 structural volume..."
    aws s3 cp \
      s3://hcp-openaccess/HCP_1200/${sub}/T1w/T1w_acpc_dc_restore_1.25.nii.gz \
      "${TARGET_DIR}/${sub}/T1w/" \
      --region us-east-1

    # 2. Download Preprocessed Diffusion MRI directory
    echo "Downloading Diffusion dataset..."
    aws s3 cp \
      s3://hcp-openaccess/HCP_1200/${sub}/T1w/Diffusion \
      "${TARGET_DIR}/${sub}/T1w/Diffusion/" \
      --recursive \
      --region us-east-1
done

echo "Download process completed."