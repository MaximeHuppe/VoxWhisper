# pipeline/train.py — config-driven VoxWhisper training
from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

import torch
from torch.utils.data import DataLoader

from src.data.dataset import VoxWhisperDataset
from src.models.vox_whisper import VoxWhisper
from src.training.checkpoint import (
    TopKCheckpoints,
    checkpoint_config,
    describe_monitor,
    load_model_state,
    monitor_score,
    save_checkpoint,
    should_eval_volume,
    should_save_periodic,
)
from src.utils.config import load_config
from src.training.logger import TrainingLogger
from src.training.metrics import DiceBCELoss, deep_supervision_loss
from src.utils.run import create_or_resume_run
from src.utils.seed import get_training_seed, set_global_seed, worker_init_fn
from src.data.splits import create_or_load_splits
from src.training.validate import evaluate_patches, evaluate_volume_dice

# Maximum gradient norm for clipping.  MHA-heavy architectures can produce
# large gradient spikes early in training; clipping improves stability without
# significantly slowing convergence.
_GRAD_CLIP_NORM = 1.0


def _print_run_header(
    config: dict,
    run_path,
    device,
    seed: int,
    ckpt_cfg: dict,
    train_loader,
    val_loader,
) -> None:
    """Print a compact one-time summary at training start."""
    train_cfg = config["training"]
    modalities = config["data"].get("modalities", {})
    structures = config["data"].get("structure_names") or config["data"].get("prompts", [])
    n_structs = len(structures)
    struct_preview = ", ".join(structures[:4])
    if n_structs > 4:
        struct_preview += f", … (+{n_structs - 4})"

    n_train_subj = len(getattr(train_loader.dataset, "subject_ids", []))
    n_train_patches = len(train_loader.dataset)
    n_train_steps = len(train_loader)

    val_line = "—"
    if val_loader is not None:
        n_val_subj = len(getattr(val_loader.dataset, "subject_ids", []))
        n_val_patches = len(val_loader.dataset)
        val_line = f"{n_val_subj} subjects · {n_val_patches} patches"

    run_name = config["training"].get("run_name", "?")
    dataset = str(run_path).split("/")[-3] if len(str(run_path).split("/")) >= 3 else "?"

    sep = "─" * 66
    print(sep)
    print(f"  VoxWhisper  ·  {run_name}  ·  {dataset}")
    print(sep)
    print(f"  Run        {run_path}")
    print(f"  Device     {device:<20}  Seed     {seed}")
    print(f"  Epochs     {train_cfg['epochs']:<8}  LR  {train_cfg['learning_rate']:.2e}"
          f"    Batch  {train_cfg['batch_size']}")
    print(f"  Monitor    {describe_monitor(ckpt_cfg)}")
    print(f"  Structures {n_structs}  [{struct_preview}]")
    print(sep)
    print(f"  Modalities primary={modalities.get('primary','?')}  "
          f"secondary={modalities.get('secondary','?')}")
    print(f"  Train      {n_train_subj} subjects · {n_train_patches} patches"
          f" · {n_train_steps} steps/epoch")
    print(f"  Val        {val_line}")
    print(sep)


def build_loaders(config: dict, seed: int):
    """Create train (and optional val) DataLoaders from config.

    Returns ``(train_loader, val_loader)`` where ``val_loader`` is ``None``
    when no split is configured.
    """
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
        common_kwargs["worker_init_fn"] = lambda worker_id: worker_init_fn(worker_id, seed)

    shuffle = dl_cfg.get("shuffle", True)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    if splits_cfg.get("enabled", False):
        splits = create_or_load_splits(config)
        train_dataset = VoxWhisperDataset(config, subject_ids=splits["train"], training=True)
        val_dataset = VoxWhisperDataset(config, subject_ids=splits["val"], training=False)
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


def _load_resume_state(run_dir, model, optimizer, scheduler):
    """Load the latest checkpoint and fast-forward the scheduler.

    Returns the epoch to resume from (1-based, i.e. training continues at
    epoch ``start_epoch + 1``).  Returns 0 when no checkpoint is found.
    """
    latest = run_dir / "vox_whisper_latest.pt"
    if not latest.exists():
        print("Warning: --resume requested but no checkpoint found — starting fresh")
        return 0

    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    load_model_state(model, ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0))

    # Advance the scheduler to match the saved state.  CosineAnnealingLR is
    # stateless apart from ``last_epoch``, so replaying ``step()`` is correct.
    for _ in range(start_epoch):
        scheduler.step()

    print(f"Resumed from epoch {start_epoch}: {latest}")
    return start_epoch


def train_model(
    config: dict,
    resume: bool = False,
    *,
    name_override: str | None = None,
    run_dir: str | None = None,
    config_path: str | None = None,
    verbose: bool = False,
) -> None:
    train_cfg = config["training"]
    seed = get_training_seed(config)
    set_global_seed(seed)
    ckpt_cfg = checkpoint_config(train_cfg)
    inf_threshold = float(config.get("inference", {}).get("threshold", 0.5))

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    epochs = train_cfg["epochs"]
    deep_sup_weights = train_cfg["deep_supervision_weights"]

    run_path = create_or_resume_run(
        config,
        resume=resume,
        name_override=name_override,
        run_dir=run_dir,
        config_path=config_path,
        seed=seed,
    )
    train_loader, val_loader = build_loaders(config, seed)

    if val_loader is None:
        print("Warning: no val split — metric-based best checkpoint will be skipped")

    model = VoxWhisper.from_config(config).to(device)
    n_decoder_stages = len(model.decoder.up_blocks)
    if len(deep_sup_weights) != n_decoder_stages:
        raise ValueError(
            f"training.deep_supervision_weights has {len(deep_sup_weights)} values, "
            f"but the decoder has {n_decoder_stages} stages"
        )

    if verbose:
        model.print_summary(config)

    _print_run_header(config, run_path, device, seed, ckpt_cfg, train_loader, val_loader)

    n_prompts = len(config["data"]["prompts"])
    class_names = config["data"].get("structure_names") or config["data"]["prompts"]
    criterion = DiceBCELoss.from_config(config, n_prompts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    topk = TopKCheckpoints(k=ckpt_cfg["keep"], cache_dir=run_path)

    start_epoch = 0
    if resume:
        start_epoch = _load_resume_state(run_path, model, optimizer, scheduler)
        restored = topk.restore_from_manifest()
        if restored:
            print(f"Restored {restored} top-k checkpoint entries from manifest")

    with TrainingLogger(run_path, total_epochs=epochs, resume=resume) as logger:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0

            for primary_vol, secondary_vol, text_emb, gt_mask in train_loader:
                primary_vol = primary_vol.to(device)
                secondary_vol = secondary_vol.to(device)
                text_emb = text_emb.to(device)
                gt_mask = gt_mask.to(device)

                optimizer.zero_grad()
                predictions = model(primary_vol, secondary_vol, text_emb)
                batch_loss = deep_supervision_loss(predictions, gt_mask, criterion, deep_sup_weights)
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)

                optimizer.step()
                epoch_loss += batch_loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else epoch_loss
            metrics = {"train_loss": avg_loss}

            if val_loader is not None:
                val_metrics = evaluate_patches(
                    model, val_loader, criterion, deep_sup_weights, device,
                    threshold=inf_threshold,
                    class_names=class_names,
                )
                metrics.update(val_metrics)

                want_volume = (
                    ckpt_cfg["monitor"] == "dice"
                    and ckpt_cfg["dice_scope"] == "volume"
                    and should_eval_volume(epoch, ckpt_cfg["volume_every"])
                )
                if want_volume:
                    volume_metrics = evaluate_volume_dice(
                        model, config, val_loader.dataset.subject_ids, device
                    )
                    if volume_metrics is None:
                        print("Warning: volume Dice skipped (no val subjects with GT)")
                    else:
                        metrics.update(volume_metrics)

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

            latest_extra = (
                {"monitor_score": score, "monitor_metric": score_name}
                if score is not None else None
            )
            save_checkpoint(
                run_path / "vox_whisper_latest.pt",
                epoch + 1,
                model,
                optimizer,
                metrics,
                config,
                extra=latest_extra,
            )

            if should_save_periodic(epoch, ckpt_cfg):
                checkpoint_path = run_path / f"vox_whisper_epoch_{epoch + 1}.pt"
                save_checkpoint(checkpoint_path, epoch + 1, model, optimizer, metrics, config)
                print(f"Saved periodic checkpoint: {checkpoint_path.name}")

        logger.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VoxWhisper")
    parser.add_argument(
        "--config",
        default="config/tracts.yaml",
        help="Path to YAML config (relative to project root or absolute)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override training.run_name for this launch (folder under runs/{dataset}/)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Use an existing timestamped run directory (with or without --resume)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from vox_whisper_latest.pt in the newest run under "
            "runs/{dataset}/{run_name}/ (or --run-dir). "
            "If no checkpoint exists, training starts a new run with a warning."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full model architecture summary at startup",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_model(
        cfg,
        resume=args.resume,
        name_override=args.name,
        run_dir=args.run_dir,
        config_path=args.config,
        verbose=args.verbose,
    )
