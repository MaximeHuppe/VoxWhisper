# preprocess/preprocess_volumes.py
import os
import glob
import numpy as np
import nibabel as nib

def normalize_intensity(volume):
    """Min-Max normalization to scale intensities to [0, 1]."""
    vol_min = np.min(volume)
    vol_max = np.max(volume)
    if vol_max - vol_min > 0:
        return (volume - vol_min) / (vol_max - vol_min)
    return volume

def center_crop_or_pad_3d(volume, target_shape=(96, 96, 96)):
    """Center crops or pads a 3D numpy array to match target dimensions."""
    spatial_shape = volume.shape
    output = np.zeros(target_shape, dtype=volume.dtype)
    
    slices_in = []
    slices_out = []
    
    for i in range(3):
        if spatial_shape[i] >= target_shape[i]:
            start = (spatial_shape[i] - target_shape[i]) // 2
            slices_in.append(slice(start, start + target_shape[i]))
            slices_out.append(slice(0, target_shape[i]))
        else:
            start = (target_shape[i] - spatial_shape[i]) // 2
            slices_in.append(slice(0, spatial_shape[i]))
            slices_out.append(slice(start, start + spatial_shape[i]))
            
    output[tuple(slices_out)] = volume[tuple(slices_in)]
    return output

def process_subject(subject_id, raw_dir, output_dir):
    print(f"Processing Subject: {subject_id}")
    
    # Define file paths
    t1_path = os.path.join(raw_dir, subject_id, "T1w", "T1w_acpc_dc_restore_1.25.nii.gz")
    diff_path = os.path.join(raw_dir, subject_id, "T1w", "Diffusion", "data.nii.gz") # 4D DTI
    
    if not os.path.exists(t1_path):
        print(f"Warning: T1 file missing for {subject_id}. Skipping.")
        return

    # 1. Load and process structural T1
    t1_img = nib.load(t1_path)
    t1_data = t1_img.get_fdata().astype(np.float32)
    t1_norm = normalize_intensity(t1_data)
    t1_cropped = center_crop_or_pad_3d(t1_norm, target_shape=(128, 128, 128))
    
    # 2. Load and process Diffusion (extract first b=0 volume or mean FA for simplicity)
    # Here we load the first volume of the DTI series as our structural diffusion guide
    diff_cropped = np.zeros((64, 64, 64), dtype=np.float32)
    if os.path.exists(diff_path):
        diff_img = nib.load(diff_path)
        # Squeeze 4D to 3D by taking the first volume (index 0)
        diff_data = diff_img.slicer[..., 0].get_fdata().astype(np.float32)
        diff_norm = normalize_intensity(diff_data)
        diff_cropped = center_crop_or_pad_3d(diff_norm, target_shape=(64, 64, 64))

    # 3. Save preprocessed arrays
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{subject_id}_preprocessed.npz")
    np.savez_compressed(
        out_file,
        t1=t1_cropped[np.newaxis, ...],      # Shape: [1, 128, 128, 128]
        diff=diff_cropped[np.newaxis, ...],  # Shape: [1, 64, 64, 64]
    )
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    raw_data_dir = "../data/raw"
    processed_data_dir = "../data/processed"
    
    subjects = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
    for subject in subjects:
        process_subject(subject, raw_data_dir, processed_data_dir)