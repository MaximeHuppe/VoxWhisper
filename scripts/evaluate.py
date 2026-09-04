"""Evaluate VoxWhisper on a subject split — thin CLI."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from voxwhisper.util.config import ensure_dir, load_config, resolve_path
from voxwhisper.data.nifti_io import (
    label_to_multichannel,
    list_subject_ids,
    save_nifti,
)
from voxwhisper.data.splits import create_or_load_splits
from voxwhisper.infer import (
    load_dense_subject_for_inference,
    load_subject_for_inference,
    load_text_embeddings,
    logits_to_label_map,
    predict_dense_volume,
    predict_full_volume,
    volume_to_tensor,
    volumes_to_tensors,
)
from voxwhisper.util.run import (
    find_checkpoint_for_eval,
    predictions_dir,
    resolve_run_dir_for_eval,
)
from voxwhisper.training.checkpoint import load_model_state
from voxwhisper.training.metrics import channel_dice_from_logits, per_class_dice
from voxwhisper.util.stage import build_model, uses_secondary


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VoxDense or VoxWhisper")
    parser.add_argument("--config", default="config/voxdense.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--subject", default=None)
    return parser.parse_args(argv)


def _as_path(value: str) -> Path:
    from voxwhisper.util.config import get_project_root
    path = Path(value)
    return path if path.is_absolute() else get_project_root() / path


def _checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def resolve_checkpoint(config, explicit, *, run_dir=None) -> Path:
    inf_cfg = config.get("inference", {})
    raw = explicit if explicit is not None else inf_cfg.get("checkpoint")
    if raw in (None, "", "null"):
        raw = None
    return find_checkpoint_for_eval(
        config,
        explicit=str(raw) if raw else None,
        run_dir=run_dir,
    )


def resolve_subjects(config, split, subject) -> list[str]:
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


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(config, checkpoint_path: Path, device: torch.device):
    model = build_model(config).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    load_model_state(model, state)
    model.eval()
    return model


def evaluate_subject(model, config, subject_id, text_embeddings, device, output_dir):
    inf_cfg = config.get("inference", {})
    patch_size = tuple(int(x) for x in config["data"]["patch"]["size"])
    prompts = list(config["data"]["prompts"])
    n_prompts = len(prompts)
    threshold = float(inf_cfg.get("threshold", 0.5))
    sw_kwargs = dict(
        roi_size=patch_size,
        sw_batch_size=int(inf_cfg.get("sw_batch_size", 2)),
        overlap=float(inf_cfg.get("overlap", 0.5)),
        mode=str(inf_cfg.get("mode", "gaussian")),
        sigma_scale=float(inf_cfg.get("sigma_scale", 0.125)),
        progress=bool(inf_cfg.get("progress", True)),
    )

    if uses_secondary(config):
        primary_np, secondary_np, labels_np, affine = load_subject_for_inference(config, subject_id)
        primary, secondary = volumes_to_tensors(primary_np, secondary_np)
        with torch.no_grad():
            logits = predict_full_volume(
                model, primary.to(device), secondary.to(device),
                text_embeddings.to(device), **sw_kwargs,
            )
    else:
        volume_np, labels_np, affine = load_dense_subject_for_inference(config, subject_id)
        volume = volume_to_tensor(volume_np)
        with torch.no_grad():
            logits = predict_dense_volume(
                model, volume.to(device), text_embeddings.to(device), **sw_kwargs,
            )

    pred_labels = logits_to_label_map(logits)[0].cpu()
    subject_dir = ensure_dir(output_dir / subject_id)
    save_nifti(pred_labels.numpy().astype(np.uint8), subject_dir / "pred_labels.nii.gz",
               affine=affine, dtype=np.uint8)

    if inf_cfg.get("save_probabilities", False):
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        save_nifti(np.moveaxis(probs, 0, -1), subject_dir / "pred_probs.nii.gz",
                   affine=affine, dtype=np.float32)

    metrics = None
    if labels_np is not None:
        gt = torch.from_numpy(labels_np.astype(np.int64))
        argmax_dice = per_class_dice(pred_labels, gt, n_classes=n_prompts)
        gt_onehot = torch.from_numpy(label_to_multichannel(labels_np, n_prompts))
        channel_dice = channel_dice_from_logits(logits.cpu(), gt_onehot.unsqueeze(0), threshold=threshold)
        metrics = {"argmax": argmax_dice, "channel": channel_dice}
        parts = [
            f"{name}: argmax={argmax_dice[i]:.4f} ch={channel_dice[i]:.4f}"
            for i, name in enumerate(prompts)
        ]
        print(f"  {subject_id}  " + "  ".join(parts))
    else:
        print(f"  {subject_id}  saved {subject_dir / 'pred_labels.nii.gz'} (no GT mask)")

    return metrics


def write_dice_tables(output_dir, subjects, scores, structure_names) -> None:
    arr = np.array(scores)

    per_subject_path = output_dir / "dice_per_subject.csv"
    with open(per_subject_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id"] + structure_names)
        for sid, row in zip(subjects, arr):
            writer.writerow([sid] + [f"{v:.4f}" for v in row])
    print(f"  Saved {per_subject_path.name}")

    summary_path = output_dir / "dice_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["structure", "mean", "std", "min", "max"])
        for i, name in enumerate(structure_names):
            col = arr[:, i]
            writer.writerow([name, f"{col.mean():.4f}", f"{col.std():.4f}",
                             f"{col.min():.4f}", f"{col.max():.4f}"])
    print(f"  Saved {summary_path.name}")


def evaluate(config, args) -> None:
    device = pick_device()
    inf_cfg = config.get("inference", {})
    split_name = args.split or inf_cfg.get("split", "test")

    eval_run_dir = resolve_run_dir_for_eval(
        config, run_dir=getattr(args, "run_dir", None)
    )
    if eval_run_dir is not None:
        output_dir = predictions_dir(eval_run_dir, split_name)
    else:
        output_dir = _as_path(inf_cfg.get("output_dir", "data/predictions"))
    if getattr(args, "output_dir", None):
        output_dir = _as_path(args.output_dir)
    ensure_dir(output_dir)

    checkpoint_path = resolve_checkpoint(
        config, args.checkpoint, run_dir=getattr(args, "run_dir", None)
    )
    subjects = resolve_subjects(config, args.split, args.subject)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Subjects ({len(subjects)}): {subjects}")

    model = load_model(config, checkpoint_path, device)
    text_embeddings = load_text_embeddings(config, map_location="cpu")

    subjects_with_gt = []
    all_channel = []
    n_prompts = len(config["data"]["prompts"])

    for subject_id in subjects:
        metrics = evaluate_subject(model, config, subject_id, text_embeddings, device, output_dir)
        if metrics is not None:
            subjects_with_gt.append(subject_id)
            all_channel.append(metrics["channel"])

    if all_channel:
        arr_ch = np.asarray(all_channel)
        prompts = config["data"]["prompts"]
        structure_names = config["data"].get("structure_names") or prompts
        print("\nMean Dice across subjects:")
        fg = list(range(1, n_prompts))
        for i, name in enumerate(prompts):
            print(f"  {name:20s}  channel={arr_ch[:, i].mean():.4f}")
        if fg:
            print(f"  foreground mean     channel={arr_ch[:, fg].mean():.4f}")
        print("\nWriting result tables:")
        write_dice_tables(output_dir, subjects_with_gt, all_channel, structure_names)

    print(f"\nWrote predictions to {output_dir}")


def main(argv=None) -> None:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    evaluate(cfg, args)


if __name__ == "__main__":
    main()
