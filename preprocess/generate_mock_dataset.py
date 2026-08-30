# preprocess/generate_mock_dataset.py
import os
import numpy as np

def make_mock_cohort(num_subjects=10):
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_subjects} mock subject volumes...")
    for i in range(num_subjects):
        subject_id = f"SUBJ{i:03d}"
        
        # 1. High-Res T1 [1, 128, 128, 128]
        t1 = np.random.uniform(0.0, 1.0, size=(1, 128, 128, 128)).astype(np.float32)
        # 2. Unregistered Diffusion [1, 64, 64, 64]
        diff = np.random.uniform(0.0, 1.0, size=(1, 64, 64, 64)).astype(np.float32)
        # 3. Three Ground Truth Segmentation Masks [3, 128, 128, 128]
        masks = np.random.randint(0, 2, size=(3, 128, 128, 128)).astype(np.float32)

        out_file = os.path.join(output_dir, f"{subject_id}_preprocessed.npz")
        np.savez_compressed(out_file, t1=t1, diff=diff)
        
        # Save companion mask file for deep supervision target extraction
        mask_file = os.path.join(output_dir, f"{subject_id}_mask.npz")
        np.savez_compressed(mask_file, masks=masks)
        
    print("Mock cohort generated successfully.")

if __name__ == "__main__":
    # If running from project root, adjust relative import directory
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    make_mock_cohort()