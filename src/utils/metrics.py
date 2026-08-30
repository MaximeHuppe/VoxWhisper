# src/utils/metrics.py
import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, target_mask):
        # pred_logits: [B, N_T, D, H, W]
        # target_mask: [B, N_T, D, H, W]
        
        # 1. Compute BCE Loss
        bce_loss = self.bce(pred_logits, target_mask)
        
        # 2. Compute Dice Loss
        pred_probs = torch.sigmoid(pred_logits)
        
        # Flatten spatial dimensions
        pred_flat = pred_probs.view(-1)
        target_flat = target_mask.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
        
        return bce_loss + dice_loss