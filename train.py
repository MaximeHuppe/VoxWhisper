# train.py — config-driven VoxWhisper training
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import VoxWhisperDataset
from src.models.vox_whisper import VoxWhisper
from src.utils.config import (
    ensure_dir,
    load_config,
    parse_config_args,
    resolve_path,
)
from src.utils.metrics import DiceBCELoss
from src.utils.splits import create_or_load_splits


@torch.no_grad()
def evaluate(model, loader, criterion, deep_sup_weights, device):
    """Run a validation pass and return mean deep-supervision loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for t1_vol, t2_vol, text_emb, gt_mask in loader:
        t1_vol = t1_vol.to(device)
        t2_vol = t2_vol.to(device)
        text_emb = text_emb.to(device)
        gt_mask = gt_mask.to(device)

        predictions = model(t1_vol, t2_vol, text_emb)
        batch_loss = 0.0
        for idx, pred in enumerate(predictions):
            downsampled_target = F.interpolate(
                gt_mask,
                size=pred.shape[2:],
                mode="trilinear",
                align_corners=True,
            )
            batch_loss = batch_loss + deep_sup_weights[idx] * criterion(
                pred, downsampled_target
            )

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def build_loaders(config):
    """Create train (and optional val) DataLoaders from config."""
    train_cfg = config["training"]
    dl_cfg = train_cfg["dataloader"]
    splits_cfg = config["splits"]

    common_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=dl_cfg.get("num_workers", 0),
        pin_memory=dl_cfg.get("pin_memory", False),
    )

    if splits_cfg.get("enabled", False):
        splits = create_or_load_splits(config)
        train_dataset = VoxWhisperDataset(
            config, subject_ids=splits["train"], training=True
        )
        val_dataset = VoxWhisperDataset(
            config, subject_ids=splits["val"], training=False
        )
        train_loader = DataLoader(
            train_dataset,
            shuffle=dl_cfg.get("shuffle", True),
            drop_last=dl_cfg.get("drop_last", True),
            **common_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )
        return train_loader, val_loader

    train_dataset = VoxWhisperDataset(config, training=True)
    train_loader = DataLoader(
        train_dataset,
        shuffle=dl_cfg.get("shuffle", True),
        drop_last=dl_cfg.get("drop_last", True),
        **common_kwargs,
    )
    return train_loader, None


def train_model(config):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    train_cfg = config["training"]
    epochs = train_cfg["epochs"]
    learning_rate = train_cfg["learning_rate"]
    deep_sup_weights = train_cfg["deep_supervision_weights"]
    checkpoint_every = train_cfg.get("checkpoint_every", 10)

    cache_dir = resolve_path(config, "data.paths.cache")
    ensure_dir(cache_dir)

    print("Initializing dataset and dataloader...")
    train_loader, val_loader = build_loaders(config)

    model = VoxWhisper.from_config(config).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Starting training on device: {device}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for t1_vol, t2_vol, text_emb, gt_mask in train_loader:
            t1_vol = t1_vol.to(device)
            t2_vol = t2_vol.to(device)
            text_emb = text_emb.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()
            predictions = model(t1_vol, t2_vol, text_emb)

            batch_loss = 0.0
            for idx, pred in enumerate(predictions):
                downsampled_target = F.interpolate(
                    gt_mask,
                    size=pred.shape[2:],
                    mode="trilinear",
                    align_corners=True,
                )
                stage_loss = criterion(pred, downsampled_target)
                batch_loss = batch_loss + deep_sup_weights[idx] * stage_loss

            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else epoch_loss

        log_msg = (
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {avg_loss:.4f} - "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, deep_sup_weights, device)
            log_msg += f" - Val Loss: {val_loss:.4f}"

        print(log_msg)

        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = cache_dir / f"vox_whisper_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    args = parse_config_args(description="Train VoxWhisper")
    cfg = load_config(args.config)
    train_model(cfg)
