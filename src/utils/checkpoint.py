"""Parse checkpoint config, pick the monitored score, and write .pt files."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def checkpoint_config(train_cfg):
    """Parse ``training.checkpoint`` (falls back to legacy ``checkpoint_every``)."""
    raw = train_cfg.get("checkpoint") or {}
    monitor = str(raw.get("monitor", "loss")).lower()
    if monitor not in {"loss", "dice"}:
        raise ValueError(
            "training.checkpoint.monitor must be 'loss' or 'dice', "
            f"got {monitor!r}"
        )

    dice_scope = str(raw.get("dice_scope", "patch")).lower()
    if dice_scope not in {"patch", "volume"}:
        raise ValueError(
            "training.checkpoint.dice_scope must be 'patch' or 'volume', "
            f"got {dice_scope!r}"
        )

    every = int(raw.get("every", train_cfg.get("checkpoint_every", 10)))
    if every < 1:
        raise ValueError("training.checkpoint.every must be >= 1")

    volume_every = int(raw.get("volume_every", every))
    if volume_every < 1:
        raise ValueError("training.checkpoint.volume_every must be >= 1")

    keep = int(raw.get("keep", 3))
    if keep < 1:
        raise ValueError("training.checkpoint.keep must be >= 1")

    return {
        "monitor": monitor,
        "dice_scope": dice_scope,
        "every": every,
        "volume_every": volume_every,
        "keep": keep,
        "keep_periodic": bool(raw.get("keep_periodic", True)),
    }


def describe_monitor(ckpt_cfg: dict) -> str:
    keep = ckpt_cfg.get("keep", 3)
    if ckpt_cfg["monitor"] == "loss":
        base = "loss"
    else:
        base = f"dice ({ckpt_cfg['dice_scope']}"
        if ckpt_cfg["dice_scope"] == "volume":
            base += f", every {ckpt_cfg['volume_every']} epochs"
        base += ")"
    return f"{base}, keep top {keep}"


def should_eval_volume(epoch: int, volume_every: int) -> bool:
    """True on epochs ``volume_every``, ``2 * volume_every``, ... (1-based)."""
    return (epoch + 1) % volume_every == 0


def should_save_periodic(epoch: int, ckpt_cfg: dict) -> bool:
    return ckpt_cfg["keep_periodic"] and (epoch + 1) % ckpt_cfg["every"] == 0


def monitor_score(metrics: dict, ckpt_cfg: dict):
    """
    Return ``(score, higher_is_better, name)`` for the configured monitor.

    ``score`` is ``None`` when the chosen metric was not computed this epoch
    (volume Dice is periodic).
    """
    monitor = ckpt_cfg["monitor"]
    if monitor == "loss":
        return metrics.get("val_loss"), False, "val_loss"

    key = "dice_patch" if ckpt_cfg["dice_scope"] == "patch" else "dice_volume"
    return metrics.get(key), True, key


def is_better(score: float, best: float | None, higher_is_better: bool) -> bool:
    if best is None:
        return True
    return score > best if higher_is_better else score < best


def qualifies_for_topk(score, ranked_scores, k, higher_is_better) -> bool:
    """``ranked_scores`` is best-first. ``score is None`` never qualifies."""
    if score is None:
        return False
    if len(ranked_scores) < k:
        return True
    return is_better(score, ranked_scores[-1], higher_is_better)


class TopKCheckpoints:
    """Keep the best ``k`` checkpoints on disk for the configured monitor.

    State is persisted to ``vox_whisper_topk.json`` after every update so it
    can be restored when training resumes.  Call ``restore_from_manifest()``
    before the first epoch to pick up where a previous run left off.
    """

    def __init__(self, k: int, cache_dir):
        self.k = k
        self.cache_dir = Path(cache_dir)
        self.entries: list[dict] = []
        self.higher_is_better: bool | None = None
        self.metric_name: str | None = None

    def restore_from_manifest(self) -> int:
        """Repopulate state from an existing manifest written by a previous run.

        Only entries whose ``.pt`` file still exists on disk are kept.
        Returns the number of entries successfully restored (0 when the
        manifest is absent or empty).
        """
        manifest = self.cache_dir / "vox_whisper_topk.json"
        if not manifest.exists():
            return 0

        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0

        self.metric_name = payload.get("monitor")
        self.higher_is_better = payload.get("higher_is_better")

        restored = []
        for entry in payload.get("entries", []):
            pt_path = self.cache_dir / entry["file"]
            if pt_path.exists():
                restored.append({
                    "score": float(entry["score"]),
                    "epoch": int(entry["epoch"]),
                    "path": pt_path,
                })

        self.entries = sorted(
            restored,
            key=lambda e: e["score"],
            reverse=bool(self.higher_is_better),
        )
        return len(self.entries)

    def update(self, score, higher_is_better: bool, metric_name: str, epoch: int, save_fn):
        """
        Write a new checkpoint if ``score`` belongs in the top-k.

        ``save_fn(path)`` must create the ``.pt`` file. Returns 1-based rank
        or ``None`` if the score was not kept.
        """
        if self.higher_is_better is None:
            self.higher_is_better = higher_is_better
            self.metric_name = metric_name

        ranked = [entry["score"] for entry in self.entries]
        if not qualifies_for_topk(score, ranked, self.k, self.higher_is_better):
            return None

        path = self.cache_dir / f"vox_whisper_e{epoch:03d}.pt"
        save_fn(path)
        self.entries.append(
            {"score": float(score), "epoch": int(epoch), "path": path}
        )
        self.entries.sort(
            key=lambda entry: entry["score"], reverse=self.higher_is_better
        )
        while len(self.entries) > self.k:
            dropped = self.entries.pop()
            if dropped["path"] != path:
                dropped["path"].unlink(missing_ok=True)

        self._refresh_rank_links()
        self._write_manifest()
        return next(
            rank
            for rank, entry in enumerate(self.entries, start=1)
            if entry["path"] == path
        )

    def _refresh_rank_links(self) -> None:
        for rank in range(1, self.k + 1):
            link = self.cache_dir / f"vox_whisper_top{rank}.pt"
            if link.exists() or link.is_symlink():
                link.unlink()
        best = self.cache_dir / "vox_whisper_best.pt"
        if best.exists() or best.is_symlink():
            best.unlink()

        for rank, entry in enumerate(self.entries, start=1):
            link = self.cache_dir / f"vox_whisper_top{rank}.pt"
            link.symlink_to(entry["path"].name)
        if self.entries:
            (self.cache_dir / "vox_whisper_best.pt").symlink_to(
                self.entries[0]["path"].name
            )

    def _write_manifest(self) -> None:
        payload = {
            "monitor": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "keep": self.k,
            "entries": [
                {
                    "rank": rank,
                    "epoch": entry["epoch"],
                    "score": entry["score"],
                    "file": entry["path"].name,
                }
                for rank, entry in enumerate(self.entries, start=1)
            ],
        }
        path = self.cache_dir / "vox_whisper_topk.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_LEGACY_STATE_PREFIXES = (
    ("t1_encoder.", "primary_encoder."),
    ("t2_encoder.", "secondary_encoder."),
    ("cross_volume_attention.pos_t1.", "cross_volume_attention.pos_primary."),
)


def remap_legacy_state_dict(state_dict):
    """Rewrite pre-rename encoder / PE keys so old checkpoints still load."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in _LEGACY_STATE_PREFIXES:
            if new_key.startswith(old):
                new_key = new + new_key[len(old) :]
                break
        remapped[new_key] = value
    return remapped


def load_model_state(model, state_dict, strict=True):
    return model.load_state_dict(remap_legacy_state_dict(state_dict), strict=strict)


def save_checkpoint(path, epoch, model, optimizer, metrics, config, extra=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": metrics.get("train_loss"),
        "metrics": metrics,
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
