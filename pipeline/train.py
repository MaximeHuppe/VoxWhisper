# train.py — config-driven VoxWhisper training
from __future__ import annotations
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.dataset import VoxWhisperDataset
from src.models.vox_whisper import VoxWhisper
from src.utils.checkpoint import (
    TopKCheckpoints,
    checkpoint_config,
    describe_monitor,
    monitor_score,
    save_checkpoint,
    should_eval_volume,
    should_save_periodic,
)
from src.utils.config import (
    ensure_dir,
    load_config,
    parse_config_args,
    resolve_path,
)
from src.utils.metrics import DiceBCELoss, deep_supervision_loss
from src.utils.seed import get_training_seed, set_global_seed, worker_init_fn
from src.utils.splits import create_or_load_splits
from src.utils.validate import evaluate_patches, evaluate_volume_dice


def build_loaders(config, seed: int):
    """Create train (and optional val) DataLoaders from config."""
    train_cfg = config["training"]
    dl_cfg = train_cfg["dataloader"]
    splits_cfg = config["splits"]
    num_workers = dl_cfg.get("num_workers", 0)

    common_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=dl_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["worker_init_fn"] = lambda worker_id: worker_init_fn(
            worker_id, seed
        )

    shuffle = dl_cfg.get("shuffle", True)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

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
            shuffle=shuffle,
            drop_last=dl_cfg.get("drop_last", True),
            generator=train_generator if shuffle else None,
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
        shuffle=shuffle,
        drop_last=dl_cfg.get("drop_last", True),
        generator=train_generator if shuffle else None,
        **common_kwargs,
    )
    return train_loader, None


def train_model(config):
    train_cfg = config["training"]
    seed = get_training_seed(config)
    set_global_seed(seed)
    ckpt_cfg = checkpoint_config(train_cfg)
    inf_threshold = float(config.get("inference", {}).get("threshold", 0.5))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    epochs = train_cfg["epochs"]
    learning_rate = train_cfg["learning_rate"]
    deep_sup_weights = train_cfg["deep_supervision_weights"]

    cache_dir = resolve_path(config, "data.paths.cache")
    ensure_dir(cache_dir)

    print(f"Training seed: {seed}")
    print(f"Checkpoint monitor: {describe_monitor(ckpt_cfg)}")
    print("Initializing dataset and dataloader...")
    train_loader, val_loader = build_loaders(config, seed)

    if val_loader is None:
        print(
            "Warning: no val split; metric-based best checkpoint will be skipped"
        )

    model = VoxWhisper.from_config(config).to(device)
    n_decoder_stages = len(model.decoder.up_blocks)
    if len(deep_sup_weights) != n_decoder_stages:
        raise ValueError(
            f"training.deep_supervision_weights has {len(deep_sup_weights)} values, "
            f"but the decoder has {n_decoder_stages} stages"
        )
    model.print_summary(config)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    topk = TopKCheckpoints(k=ckpt_cfg["keep"], cache_dir=cache_dir)

    print(f"Starting training on device: {device}...")
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0

        for primary_vol, secondary_vol, text_emb, gt_mask in train_loader:
            primary_vol = primary_vol.to(device)
            secondary_vol = secondary_vol.to(device)
            text_emb = text_emb.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()
            predictions = model(primary_vol, secondary_vol, text_emb)
            batch_loss = deep_supervision_loss(
                predictions, gt_mask, criterion, deep_sup_weights
            )
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else epoch_loss
        metrics = {"train_loss": avg_loss}

        log_msg = (
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {avg_loss:.4f} - "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_loader is not None:
            val_metrics = evaluate_patches(
                model,
                val_loader,
                criterion,
                deep_sup_weights,
                device,
                threshold=inf_threshold,
            )
            metrics.update(val_metrics)
            log_msg += (
                f" - Val Loss: {val_metrics['val_loss']:.4f}"
                f" - Val Dice (patch): {val_metrics['dice_patch']:.4f}"
            )

            want_volume = (
                ckpt_cfg["monitor"] == "dice"
                and ckpt_cfg["dice_scope"] == "volume"
                and should_eval_volume(epoch, ckpt_cfg["volume_every"])
            )
            if want_volume:
                volume_dice = evaluate_volume_dice(
                    model, config, val_loader.dataset.subject_ids, device
                )
                if volume_dice is None:
                    print("Warning: volume Dice skipped (no val subjects with GT)")
                else:
                    metrics["dice_volume"] = volume_dice
                    log_msg += f" - Val Dice (volume): {volume_dice:.4f}"

        print(log_msg)

        score, higher_is_better, score_name = monitor_score(metrics, ckpt_cfg)
        if score is not None:
            rank = topk.update(
                score,
                higher_is_better,
                score_name,
                epoch + 1,
                lambda path: save_checkpoint(
                    path,
                    epoch + 1,
                    model,
                    optimizer,
                    metrics,
                    config,
                    extra={
                        "monitor_score": score,
                        "monitor_metric": score_name,
                    },
                ),
            )
            if rank is not None:
                print(
                    f"Saved top-{rank}/{ckpt_cfg['keep']} checkpoint "
                    f"({score_name}={score:.4f})"
                )

        latest_extra = {}
        if score is not None:
            latest_extra = {
                "monitor_score": score,
                "monitor_metric": score_name,
            }
        save_checkpoint(
            cache_dir / "vox_whisper_latest.pt",
            epoch + 1,
            model,
            optimizer,
            metrics,
            config,
            extra=latest_extra or None,
        )

        if should_save_periodic(epoch, ckpt_cfg):
            checkpoint_path = cache_dir / f"vox_whisper_epoch_{epoch + 1}.pt"
            save_checkpoint(
                checkpoint_path, epoch + 1, model, optimizer, metrics, config
            )
            print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    args = parse_config_args(description="Train VoxWhisper")
    cfg = load_config(args.config)
    train_model(cfg)
