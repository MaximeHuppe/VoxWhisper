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


def per_class_dice(pred_labels, gt_labels, n_classes, eps=1e-5):
    """
    Integer label maps → one Dice score per class.

    Empty class in both prediction and target scores 1.0 (perfect agreement
    on absence). Returns a Python list of floats, length ``n_classes``.
    """
    pred_labels = pred_labels.long()
    gt_labels = gt_labels.long()
    scores = []
    for class_id in range(n_classes):
        pred_c = pred_labels == class_id
        gt_c = gt_labels == class_id
        intersection = (pred_c & gt_c).sum().float()
        denom = pred_c.sum() + gt_c.sum()
        if denom == 0:
            scores.append(1.0)
        else:
            scores.append(float((2.0 * intersection + eps) / (denom + eps)))
    return scores


def channel_dice_from_logits(logits, target, threshold=0.5, eps=1e-5):
    """
    Independent per-prompt Dice after sigmoid (matches the multi-label loss).

    ``logits`` and ``target`` are ``[B, N_T, D, H, W]`` (or without batch).
    """
    if logits.ndim == 4:
        logits = logits.unsqueeze(0)
        target = target.unsqueeze(0)
    pred = torch.sigmoid(logits) > threshold
    gt = target > 0.5
    n_classes = logits.shape[1]
    scores = []
    for class_id in range(n_classes):
        pred_c = pred[:, class_id]
        gt_c = gt[:, class_id]
        intersection = (pred_c & gt_c).sum().float()
        denom = pred_c.sum() + gt_c.sum()
        if denom == 0:
            scores.append(1.0)
        else:
            scores.append(float((2.0 * intersection + eps) / (denom + eps)))
    return scores
