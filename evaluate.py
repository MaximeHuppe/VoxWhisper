# evaluate.py — full-volume sliding-window inference for VoxWhisper
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.infer import (
    load_subject_for_inference,
    load_text_embeddings,
    logits_to_label_map,
    predict_full_volume,
    volumes_to_tensors,
)
from src.models.vox_whisper import VoxWhisper
from src.utils.config import (
    ensure_dir,
    get_project_root,
    load_config,
    resolve_path,
)
from src.utils.metrics import channel_dice_from_logits, per_class_dice
from src.utils.nifti_io import label_to_multichannel, list_subject_ids, save_nifti
from src.utils.splits import create_or_load_splits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate VoxWhisper on full head volumes with sliding-window inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config (relative to project root or absolute)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a training checkpoint (.pt). Defaults to the latest in cache/",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Subject split to evaluate (default: inference.split)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for predicted NIfTIs (default: inference.output_dir)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Evaluate a single subject id instead of a split",
    )
    return parser.parse_args()


def _as_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return get_project_root() / path


def resolve_checkpoint(config, explicit: str | None) -> Path:
    inf_cfg = config.get("inference", {})
    raw = explicit or inf_cfg.get("checkpoint")
    if raw:
        path = _as_path(str(raw))
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    cache_dir = resolve_path(config, "data.paths.cache")
    ckpts = list(cache_dir.glob("vox_whisper_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoint given and none found in {cache_dir} "
            "(expected vox_whisper_epoch_*.pt). Train first or pass --checkpoint."
        )
    return max(ckpts, key=_checkpoint_epoch)


def _checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def resolve_subjects(config, split: str | None, subject: str | None) -> list[str]:
    processed_dir = resolve_path(config, "data.paths.processed")
    available = list_subject_ids(processed_dir)
    if not available:
        raise FileNotFoundError(f"No processed subjects in {processed_dir}")

    if subject is not None:
        if subject not in available:
            raise KeyError(f"Subject {subject} not found in {processed_dir}")
        return [subject]

    inf_cfg = config.get("inference", {})
    split_name = split or inf_cfg.get("split", "test")
    if config.get("splits", {}).get("enabled", False):
        splits = create_or_load_splits(config)
        if split_name not in splits:
            raise KeyError(f"Split '{split_name}' not in {list(splits)}")
        chosen = [s for s in splits[split_name] if s in available]
        missing = set(splits[split_name]) - set(chosen)
        for sid in sorted(missing):
            print(f"Warning: skipping {sid} (not in processed data)")
        if not chosen:
            raise ValueError(f"No processed subjects remain in split '{split_name}'")
        return chosen

    return available


def load_model(config, checkpoint_path: Path, device: torch.device) -> VoxWhisper:
    model = VoxWhisper.from_config(config).to(device)
    model.print_summary(config)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_subject(
    model,
    config,
    subject_id: str,
    text_embeddings: torch.Tensor,
    device: torch.device,
    output_dir: Path,
):
    inf_cfg = config.get("inference", {})
    patch_size = tuple(int(x) for x in config["data"]["patch"]["size"])
    prompts = list(config["data"]["prompts"])
    n_prompts = len(prompts)
    threshold = float(inf_cfg.get("threshold", 0.5))

    t1_np, t2_np, labels_np, affine = load_subject_for_inference(config, subject_id)
    t1, t2 = volumes_to_tensors(t1_np, t2_np)
    t1 = t1.to(device)
    t2 = t2.to(device)

    with torch.no_grad():
        logits = predict_full_volume(
            model,
            t1,
            t2,
            text_embeddings.to(device),
            roi_size=patch_size,
            sw_batch_size=int(inf_cfg.get("sw_batch_size", 2)),
            overlap=float(inf_cfg.get("overlap", 0.5)),
            mode=str(inf_cfg.get("mode", "gaussian")),
            sigma_scale=float(inf_cfg.get("sigma_scale", 0.125)),
            progress=bool(inf_cfg.get("progress", True)),
        )

        print("logit shape:", tuple(logits.shape))
        print("background logit mean:", logits[0, 0].mean().item(), "max", logits[0, 0].max().item(), "min", logits[0, 0].min().item())
        print("optic nerve logit mean:", logits[0, 1].mean().item(), "max", logits[0, 1].max().item(), "min", logits[0, 1].min().item())
        print("frac argmax==background:", (logits.argmax(dim=1) == 0).float().mean().item())

    pred_labels = logits_to_label_map(logits)[0].cpu()
    subject_dir = ensure_dir(output_dir / subject_id)
    save_nifti(
        pred_labels.numpy().astype(np.uint8),
        subject_dir / "pred_labels.nii.gz",
        affine=affine,
        dtype=np.uint8,
    )

    if inf_cfg.get("save_probabilities", False):
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        # NIfTI convention: (D, H, W, C)
        save_nifti(
            np.moveaxis(probs, 0, -1),
            subject_dir / "pred_probs.nii.gz",
            affine=affine,
            dtype=np.float32,
        )

    metrics = None
    if labels_np is not None:
        gt = torch.from_numpy(labels_np.astype(np.int64))
        argmax_dice = per_class_dice(pred_labels, gt, n_classes=n_prompts)
        gt_onehot = torch.from_numpy(label_to_multichannel(labels_np, n_prompts))
        channel_dice = channel_dice_from_logits(
            logits.cpu(), gt_onehot.unsqueeze(0), threshold=threshold
        )
        metrics = {"argmax": argmax_dice, "channel": channel_dice}

        parts = [
            f"{name}: argmax={argmax_dice[i]:.4f} ch={channel_dice[i]:.4f}"
            for i, name in enumerate(prompts)
        ]
        print(f"  {subject_id}  " + "  ".join(parts))
    else:
        print(f"  {subject_id}  saved {subject_dir / 'pred_labels.nii.gz'} (no GT mask)")

    return metrics


def evaluate(config, args):
    device = pick_device()
    inf_cfg = config.get("inference", {})
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    output_dir = _as_path(args.output_dir or inf_cfg.get("output_dir", "data/predictions"))
    ensure_dir(output_dir)

    subjects = resolve_subjects(config, args.split, args.subject)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(
        f"Sliding window: roi={tuple(config['data']['patch']['size'])} "
        f"overlap={inf_cfg.get('overlap', 0.5)} mode={inf_cfg.get('mode', 'gaussian')}"
    )

    model = load_model(config, checkpoint_path, device)
    text_embeddings = load_text_embeddings(config, map_location="cpu")

    all_argmax = []
    all_channel = []
    n_prompts = len(config["data"]["prompts"])

    for subject_id in subjects:
        metrics = evaluate_subject(
            model, config, subject_id, text_embeddings, device, output_dir
        )
        if metrics is not None:
            all_argmax.append(metrics["argmax"])
            all_channel.append(metrics["channel"])

    if all_argmax:
        mean_argmax = np.mean(np.asarray(all_argmax), axis=0)
        mean_channel = np.mean(np.asarray(all_channel), axis=0)
        prompts = config["data"]["prompts"]
        print("Mean Dice across subjects:")
        for i, name in enumerate(prompts):
            print(
                f"  {name:20s}  argmax={mean_argmax[i]:.4f}  "
                f"channel={mean_channel[i]:.4f}"
            )
        fg = list(range(1, n_prompts))
        if fg:
            print(
                f"  foreground mean     argmax={mean_argmax[fg].mean():.4f}  "
                f"channel={mean_channel[fg].mean():.4f}"
            )

    print(f"Wrote predictions to {output_dir}")


if __name__ == "__main__":
    cli = parse_args()
    cfg = load_config(cli.config)
    evaluate(cfg, cli)
