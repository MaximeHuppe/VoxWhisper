# src/dataset.py
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

class VoxWhisperDataset(Dataset):
    """
    Custom Dataset class loading T1, Diffusion, cached prompt embeddings,
    and target segmentation masks for training.
    """
    def __init__(self, processed_dir, cache_path, mask_dir=None):
        self.processed_dir = processed_dir
        self.mask_dir = mask_dir
        
        # Locate preprocessed visual files
        self.subject_files = sorted(glob.glob(os.path.join(processed_dir, "*_preprocessed.npz")))
        
        # Load the pre-computed text embeddings once to conserve memory
        # Shape: [1, N_T, 768] (We squeeze batch dimension to [N_T, 768])
        self.text_embeddings = torch.load(cache_path).squeeze(0)

    def __len__(self):
        return len(self.subject_files)

    def __getitem__(self, idx):
        file_path = self.subject_files[idx]
        subject_id = os.path.basename(file_path).split("_")[0]
        
        # 1. Load visual features
        data = np.load(file_path)
        t1_vol = torch.from_numpy(data["t1"]).float()      # Shape: [1, 128, 128, 128]
        diff_vol = torch.from_numpy(data["diff"]).float()  # Shape: [1, 64, 64, 64]

        # 2. Get precomputed text embeddings
        text_emb = self.text_embeddings  # Shape: [N_T, 768]

        # 3. Load or generate a mock ground truth mask for validation
        if self.mask_dir and os.path.exists(os.path.join(self.mask_dir, f"{subject_id}_mask.npz")):
            mask_data = np.load(os.path.join(self.mask_dir, f"{subject_id}_mask.npz"))
            gt_mask = torch.from_numpy(mask_data["masks"]).float() # Shape: [N_T, 128, 128, 128]
        else:
            # Fallback to a zero-filled tensor matching dimensions if no masks are present
            N_T = text_emb.shape[0]
            gt_mask = torch.zeros((N_T, 128, 128, 128), dtype=torch.float32)

        return t1_vol, diff_vol, text_emb, gt_mask