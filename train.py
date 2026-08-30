# train.py (Updated for real training)
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.vox_whisper import VoxWhisper
from src.dataset import VoxWhisperDataset
from src.utils.metrics import DiceBCELoss

def train_model():
    # 1. Device and Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    epochs = 150
    batch_size = 2  # Keep low to prevent VRAM OOM errors
    learning_rate = 5e-5
    deep_sup_weights = [0.1, 0.3, 0.6]

    # 2. Instantiate Dataset and DataLoader
    print("Initializing dataset and dataloader...")
    train_dataset = VoxWhisperDataset(
        processed_dir="data/processed",
        cache_path="cache/prompts_cn2.pt",
        mask_dir="data/processed"  # Assuming masks are in the same folder
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. Initialize Model, Loss, Optimizer
    model = VoxWhisper(input_channels=1, text_dim=768, embed_dim=128).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Cosine learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training on device: {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (t1_vol, diff_vol, text_emb, gt_mask) in enumerate(train_loader):
            # Move inputs to GPU/CPU
            t1_vol = t1_vol.to(device)       # [B, 1, 128, 128, 128]
            diff_vol = diff_vol.to(device)   # [B, 1, 64, 64, 64]
            text_emb = text_emb.to(device)   # [B, N_T, 768]
            gt_mask = gt_mask.to(device)     # [B, N_T, 128, 128, 128]

            optimizer.zero_grad()

            # Forward pass: yields intermediate predictions at 3 scales
            predictions = model(t1_vol, diff_vol, text_emb)

            # Calculate Multi-Scale Loss (Deep Supervision)
            batch_loss = 0.0
            for idx, pred in enumerate(predictions):
                target_res = pred.shape[2:] # [32^3], [64^3], [128^3]
                
                # Downsample ground truth mask to match the current stage's resolution
                downsampled_target = F.interpolate(
                    gt_mask, 
                    size=target_res, 
                    mode='trilinear', 
                    align_corners=True
                )
                
                stage_loss = criterion(pred, downsampled_target)
                batch_loss += deep_sup_weights[idx] * stage_loss

            # Backward pass and weight update
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

        # Step the learning rate scheduler
        scheduler.step()

        # Calculate epoch average loss
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else epoch_loss
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"cache/vox_whisper_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to: {checkpoint_path}")

if __name__ == "__main__":
    train_model()