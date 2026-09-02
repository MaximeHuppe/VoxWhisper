"""Structured per-epoch training logger for VoxWhisper.

Each training run writes one JSONL file to the checkpoint directory:

    cache/baseline_T1_B0/run_20260902_114200.jsonl

One JSON object per line, one line per epoch:

    {"epoch": 1, "elapsed_s": 94.2, "lr": 5e-05,
     "train_loss": 1.042, "val_loss": 0.891, "dice_patch": 0.183,
     "dice_patch_classes": {"ATR_left": 0.84, "UF_right": 0.0},
     "monitor_score": 0.183, "monitor_metric": "dice_patch", "rank": null}

Rationale
---------
stdout output is transient — lost when a terminal closes or a job scheduler
captures only part of stderr.  A JSONL file is trivially readable with:

    import pandas as pd
    df = pd.read_json("run_....jsonl", lines=True)
    df.plot(x="epoch", y=["train_loss", "val_loss"])

Resume behaviour
----------------
When ``--resume`` is used, ``TrainingLogger.open()`` detects the *newest*
existing ``.jsonl`` in the cache directory and appends to it, so a resumed
run shares one continuous log with its predecessor.

Stdout format
-------------
A fixed-width table row is printed per epoch so progress is easy to read
while training:

    Ep   1/150  loss 1.0421  val_loss 0.8910  dice_patch 0.1832  lr 5.00e-05
    Ep   5/150  loss 0.9102  val_loss 0.8121  dice_patch 0.2210  dice_vol 0.198  lr 4.98e-05  [top-2]
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _jsonify_metric(value: Any) -> Any:
    """Round floats; recursively round dicts of floats for JSONL."""
    if isinstance(value, dict):
        return {str(k): _jsonify_metric(v) for k, v in value.items()}
    return round(float(value), 6)


def _format_class_dice(class_scores: Dict[str, float]) -> str:
    """Compact ``name=0.12`` list for stdout."""
    if not class_scores:
        return "[]"
    inner = " ".join(f"{name}={score:.3f}" for name, score in class_scores.items())
    return f"[{inner}]"


class TrainingLogger:
    """Write per-epoch metrics to JSONL and print a compact stdout table row.

    Parameters
    ----------
    cache_dir : directory where the ``.jsonl`` file is written.
    total_epochs : total number of training epochs (used for the epoch column width).
    resume : if ``True``, append to the most recent existing ``.jsonl`` in
             ``cache_dir`` rather than creating a new file.
    """

    def __init__(
        self,
        cache_dir: Path,
        total_epochs: int,
        resume: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.total_epochs = total_epochs
        self._start_time = time.monotonic()
        self._file = self._open(resume)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _open(self, resume: bool):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if resume:
            existing = sorted(self.cache_dir.glob("run_*.jsonl"))
            if existing:
                path = existing[-1]
                print(f"[Logger] Appending to existing run log: {path.name}")
                return open(path, "a", encoding="utf-8")  # noqa: SIM115

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.cache_dir / f"run_{timestamp}.jsonl"
        print(f"[Logger] Writing run log to: {path.name}")
        return open(path, "w", encoding="utf-8")  # noqa: SIM115

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        lr: float,
        rank: Optional[int] = None,
    ) -> None:
        """Write one JSONL entry and print a stdout table row.

        Parameters
        ----------
        epoch   : 1-based epoch number.
        metrics : dict of metric name → float or nested dict of floats.
        lr      : current learning rate (scalar).
        rank    : top-k rank of the checkpoint saved this epoch, or ``None``.
        """
        elapsed = time.monotonic() - self._start_time

        record: dict = {
            "epoch": epoch,
            "elapsed_s": round(elapsed, 1),
            "lr": lr,
        }
        record.update({k: _jsonify_metric(v) for k, v in metrics.items()})
        if rank is not None:
            record["rank"] = rank

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        self._print_row(epoch, metrics, lr, rank)

    def close(self) -> None:
        """Flush and close the log file."""
        self._file.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def best(self) -> Dict[str, dict]:
        """Scan the log file and return the best epoch for each metric.

        Returns a dict mapping metric name → ``{"epoch": int, "value": float}``.
        Metrics ending in ``_loss`` are minimised; all others are maximised.
        """
        self._file.flush()
        path = Path(self._file.name)
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            return {}

        metric_keys = [
            k for k in records[0]
            if k not in {"epoch", "elapsed_s", "lr", "rank"}
            and not isinstance(records[0][k], dict)
        ]
        result: Dict[str, dict] = {}
        for key in metric_keys:
            values = [(r["epoch"], r[key]) for r in records if key in r]
            if not values:
                continue
            minimise = key.endswith("_loss")
            best_ep, best_val = min(values, key=lambda x: x[1]) if minimise \
                else max(values, key=lambda x: x[1])
            result[key] = {"epoch": best_ep, "value": best_val}
        return result

    def print_summary(self) -> None:
        """Print a summary of the best value per metric seen so far."""
        bests = self.best()
        if not bests:
            return
        width = max(len(k) for k in bests) + 2
        print("\n--- Training summary (best per metric) ---")
        for metric, info in bests.items():
            print(f"  {metric:<{width}} {info['value']:.4f}  (epoch {info['epoch']})")
        print()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_row(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        lr: float,
        rank: Optional[int],
    ) -> None:
        ep_w = len(str(self.total_epochs))
        parts = [f"Ep {epoch:{ep_w}d}/{self.total_epochs}"]

        order = ["train_loss", "val_loss", "dice_patch", "dice_volume"]
        seen = set()
        for key in order:
            if key in metrics:
                parts.append(f"{key} {metrics[key]:.4f}")
                seen.add(key)
            class_key = f"{key}_classes"
            if class_key in metrics:
                parts.append(_format_class_dice(metrics[class_key]))
                seen.add(class_key)
        for key, val in metrics.items():
            if key in seen or isinstance(val, dict):
                continue
            parts.append(f"{key} {val:.4f}")

        parts.append(f"lr {lr:.2e}")
        if rank is not None:
            parts.append(f"[top-{rank}]")

        print("  ".join(parts))
