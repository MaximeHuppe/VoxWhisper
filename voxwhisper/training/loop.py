"""Training loop for VoxDense (Phase 1) and VoxWhisper (Phase 2).

The model, dataset, and forward signature come from ``voxwhisper.util.stage``.
All tunable knobs come from YAML.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from voxwhisper.data.splits import create_or_load_splits
from voxwhisper.training.checkpoint import (
    TopKCheckpoints,
    load_model_state,
    monitor_score,
    save_checkpoint,
)
from voxwhisper.training.logger import TrainingLogger
from voxwhisper.training.loss import DiceBCELoss, deep_supervision_loss
from voxwhisper.training.metrics import channel_dice_from_logits, named_foreground_dice
from voxwhisper.util.run import create_or_resume_run
from voxwhisper.util.seed import get_training_seed, set_global_seed, worker_init_fn
from voxwhisper.util.stage import (
    build_dataset,
    build_model,
    forward_model,
    stage_id,
    unpack_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

_GRAD_CLIP_NORM = 1.0


# ---------------------------------------------------------------------------
# Patch validation (absorbed from the deleted validate.py)
# ---------------------------------------------------------------------------

def _mean_channel_scores(score_lists: list[list[float]]) -> list[float]:
    n = len(score_lists)
    n_ch = len(score_lists[0])
    return [sum(row[i] for row in score_lists) / n for i in range(n_ch)]


@torch.no_grad()
def evaluate_patches(
    model,
    loader,
    criterion,
    deep_sup_weights: Sequence[float],
    device,
    threshold: float = 0.5,
    class_names: Optional[Sequence[str]] = None,
) -> dict:
    """Patch-level validation.

    Returns ``val_loss``, ``dice_patch`` (mean foreground), and
    ``dice_patch_classes`` (per-tract dict).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    per_patch_scores: list[list[float]] = []

    for batch in loader:
        primary, secondary, text_emb, gt_mask = unpack_batch(batch)
        primary = primary.to(device)
        text_emb = text_emb.to(device)
        gt_mask = gt_mask.to(device)
        if secondary is not None:
            secondary = secondary.to(device)

        predictions = forward_model(model, primary, secondary, text_emb)
        total_loss += deep_supervision_loss(
            predictions, gt_mask, criterion, deep_sup_weights
        ).item()
        n_batches += 1

        logits = predictions[-1]
        for b in range(logits.shape[0]):
            per_patch_scores.append(
                channel_dice_from_logits(logits[b], gt_mask[b], threshold=threshold)
            )

    if not per_patch_scores:
        return {"val_loss": 0.0, "dice_patch": 0.0, "dice_patch_classes": {}}

    mean_scores = _mean_channel_scores(per_patch_scores)
    foreground = mean_scores[1:] if len(mean_scores) > 1 else mean_scores
    return {
        "val_loss": total_loss / n_batches if n_batches > 0 else 0.0,
        "dice_patch": sum(foreground) / len(foreground) if foreground else 1.0,
        "dice_patch_classes": named_foreground_dice(mean_scores, class_names),
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_loaders(config: dict, seed: int):
    """Create train and val DataLoaders. Splits are always enabled."""
    train_cfg = config["training"]
    dl_cfg = train_cfg["dataloader"]
    num_workers = dl_cfg.get("num_workers", 0)

    common_kwargs = dict(
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=dl_cfg.get("pin_memory", False),
    )
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True
        common_kwargs["worker_init_fn"] = lambda wid: worker_init_fn(wid, seed)

    shuffle = dl_cfg.get("shuffle", True)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    splits = create_or_load_splits(config)
    train_dataset = build_dataset(config, subject_ids=splits["train"], training=True)
    val_dataset = build_dataset(config, subject_ids=splits["val"], training=False)

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        drop_last=dl_cfg.get("drop_last", True),
        generator=train_generator if shuffle else None,
        **common_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_resume_state(run_dir: Path, model, optimizer, scheduler) -> int:
    """Load latest checkpoint; fast-forward scheduler. Returns start epoch."""
    latest = run_dir / "vox_whisper_latest.pt"
    if not latest.exists():
        print("Warning: --resume requested but no checkpoint found — starting fresh")
        return 0

    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    load_model_state(model, ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0))
    for _ in range(start_epoch):
        scheduler.step()

    print(f"Resumed from epoch {start_epoch}: {latest}")
    return start_epoch


# ---------------------------------------------------------------------------
# Run header
# ---------------------------------------------------------------------------

def _print_run_header(config, run_path, device, seed, train_loader, val_loader, model=None):
    train_cfg = config["training"]
    structures = config["data"].get("structure_names") or config["data"].get("prompts", [])
    n_structs = len(structures)
    struct_preview = ", ".join(structures[:4])
    if n_structs > 4:
        struct_preview += f", … (+{n_structs - 4})"

    n_train_subj = len(getattr(train_loader.dataset, "subject_ids", []))
    n_train_patches = len(train_loader.dataset)
    n_val_subj = len(getattr(val_loader.dataset, "subject_ids", []))
    n_val_patches = len(val_loader.dataset)

    sep = "─" * 66
    print(sep)
    print(f"  {stage_id(config):<10}  ·  {train_cfg.get('run_name', '?')}")
    print(sep)
    print(f"  Run        {run_path}")
    print(f"  Device     {device:<20}  Seed  {seed}")
    warmup = train_cfg.get("warmup_epochs", 0)
    print(
        f"  Epochs     {train_cfg['epochs']:<8}  LR  {train_cfg['learning_rate']:.2e}"
        f"  Warmup  {warmup} ep"
    )
    print(f"  Batch      {train_cfg['batch_size']:<8}  Structures  {n_structs}  [{struct_preview}]")
    if model is not None:
        model.print_param_counts()
    print(f"  Train      {n_train_subj} subjects · {n_train_patches} patches")
    print(f"  Val        {n_val_subj} subjects · {n_val_patches} patches")
    print(sep)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    config: dict,
    resume: bool = False,
    *,
    run_dir: Optional[str] = None,
    config_path: Optional[str] = None,
) -> None:
    """Run the full training loop."""
    train_cfg = config["training"]
    seed = get_training_seed(config)
    set_global_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    epochs = train_cfg["epochs"]
    learning_rate = train_cfg["learning_rate"]
    deep_sup_weights = train_cfg["deep_supervision_weights"]
    inf_threshold = float(config.get("inference", {}).get("threshold", 0.5))

    run_path = create_or_resume_run(
        config, resume=resume, run_dir=run_dir, config_path=config_path, seed=seed
    )
    train_loader, val_loader = build_loaders(config, seed)

    model = build_model(config).to(device)
    n_decoder_stages = len(model.decoder.up_blocks)
    if len(deep_sup_weights) != n_decoder_stages:
        raise ValueError(
            f"training.deep_supervision_weights has {len(deep_sup_weights)} values, "
            f"but the decoder has {n_decoder_stages} stages"
        )

    _print_run_header(config, run_path, device, seed, train_loader, val_loader, model=model)

    class_names = config["data"].get("structure_names") or config["data"]["prompts"]
    criterion = DiceBCELoss.from_config(config)

    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - warmup_epochs
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_cfg = train_cfg.get("checkpoint", {})
    topk = TopKCheckpoints(k=int(ckpt_cfg.get("keep", 3)), cache_dir=run_path)

    start_epoch = 0
    if resume:
        start_epoch = _load_resume_state(run_path, model, optimizer, scheduler)
        restored = topk.restore_from_manifest()
        if restored:
            print(f"Restored {restored} top-k checkpoint entries from manifest")

    with TrainingLogger(
        run_path,
        total_epochs=epochs,
        resume=resume,
        log_cfg=config.get("logging", {}),
        run_name=train_cfg.get("run_name"),
        full_config=config,
    ) as logger:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for batch in train_loader:
                primary, secondary, text_emb, gt_mask = unpack_batch(batch)
                primary = primary.to(device, non_blocking=True)
                text_emb = text_emb.to(device, non_blocking=True)
                gt_mask = gt_mask.to(device, non_blocking=True)
                if secondary is not None:
                    secondary = secondary.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    predictions = forward_model(model, primary, secondary, text_emb)
                    batch_loss = deep_supervision_loss(
                        predictions, gt_mask, criterion, deep_sup_weights
                    )

                batch_loss.backward()
                epoch_loss += batch_loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else epoch_loss
            metrics: dict = {"train_loss": avg_loss}

            val_metrics = evaluate_patches(
                model, val_loader, criterion, deep_sup_weights, device,
                threshold=inf_threshold,
                class_names=class_names,
            )
            metrics.update(val_metrics)

            current_lr = scheduler.get_last_lr()[0]
            score, higher_is_better, score_name = monitor_score(metrics, ckpt_cfg)

            rank = None
            if score is not None:
                rank = topk.update(
                    score,
                    higher_is_better,
                    score_name,
                    epoch + 1,
                    lambda path: save_checkpoint(
                        path, epoch + 1, model, optimizer, metrics, config,
                        extra={"monitor_score": score, "monitor_metric": score_name},
                    ),
                )

            logger.log_epoch(epoch + 1, metrics, current_lr, rank)

            save_checkpoint(
                run_path / "vox_whisper_latest.pt",
                epoch + 1, model, optimizer, metrics, config,
                extra={"monitor_score": score, "monitor_metric": score_name}
                if score is not None else None,
            )

        logger.print_summary()
