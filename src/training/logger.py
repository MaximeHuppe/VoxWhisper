"""Structured per-epoch training logger for VoxWhisper.

Each training run writes one JSONL file inside the run directory::

    runs/processed_T1_FA/baseline/20260902_114200/metrics.jsonl

One JSON object per line, one line per epoch::

    {"epoch": 1, "elapsed_s": 94.2, "lr": 5e-05,
     "train_loss": 1.042, "val_loss": 0.891, "dice_patch": 0.183,
     "dice_patch_classes": {"ATR_left": 0.84, "UF_right": 0.0},
     "monitor_score": 0.183, "monitor_metric": "dice_patch", "rank": null}

Rationale
---------
stdout output is transient — lost when a terminal closes or a job scheduler
captures only part of stderr.  A JSONL file is trivially readable with::

    import pandas as pd
    df = pd.read_json("metrics.jsonl", lines=True)
    df.plot(x="epoch", y=["train_loss", "val_loss"])

Resume behaviour
----------------
When ``--resume`` is used, ``TrainingLogger`` appends to the existing
``metrics.jsonl`` in the run directory (or falls back to the newest
legacy ``run_*.jsonl`` if present).

Stdout format
-------------
A fixed-width table row is printed per epoch so progress is easy to read
while training::

    Ep   1/150  loss 1.0421  val_loss 0.8910  dice_patch 0.1832  lr 5.00e-05
    Ep   5/150  loss 0.9102  val_loss 0.8121  dice_patch 0.2210  dice_vol 0.198  lr 4.98e-05  [top-2]

External backends (optional)
-----------------------------
Controlled by ``config["logging"]["backend"]``:

* ``"tensorboard"`` — writes a TensorBoard ``SummaryWriter`` under
  ``<run_dir>/tb_logs/``.  Launch with::

      tensorboard --logdir runs/

* ``"wandb"`` — logs to Weights & Biases.  Requires ``wandb`` installed
  and a valid API key (``wandb login``).  Project and entity are taken
  from ``config["logging"]["wandb"]``.

* ``"none"`` (default) — JSONL + stdout only.

Both backends are additive: the JSONL file is always written.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

METRICS_FILENAME = "metrics.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Recursively flatten nested metric dicts using '/' as separator.

    ``{"dice_patch_classes": {"ATR_left": 0.9}}``
    →  ``{"dice_patch_classes/ATR_left": 0.9}``
    """
    flat: Dict[str, float] = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_metrics(v, prefix=f"{key}/"))
        else:
            try:
                flat[key] = float(v)
            except (TypeError, ValueError):
                pass
    return flat


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

    Optionally mirrors metrics to TensorBoard or Weights & Biases when
    ``log_cfg`` (from ``config["logging"]``) is provided.

    Parameters
    ----------
    run_dir      : directory where ``metrics.jsonl`` is written.
    total_epochs : total number of training epochs (used for the epoch column width).
    resume       : if ``True``, append to an existing metrics file rather than truncating.
    log_cfg      : ``config["logging"]`` dict; controls external backend selection.
    run_name     : human-readable run identifier forwarded to W&B / TB.
    full_config  : entire config dict logged as W&B hyperparameters.
    """

    def __init__(
        self,
        run_dir: Path,
        total_epochs: int,
        resume: bool = False,
        *,
        log_cfg: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        full_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.total_epochs = total_epochs
        self._start_time = time.monotonic()
        self._file = self._open(resume)

        log_cfg = log_cfg or {}
        backend = str(log_cfg.get("backend", "none")).lower()

        self._wb_run = None
        self._tb_writer = None

        if backend in ("wandb", "both"):
            self._init_wandb(log_cfg.get("wandb", {}), run_name, full_config, resume)
        if backend in ("tensorboard", "both"):
            self._init_tensorboard(log_cfg.get("tensorboard", {}))
        if backend not in ("none", "", "wandb", "tensorboard", "both"):
            logger.warning("Unknown logging.backend %r — using 'none'", backend)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _init_wandb(
        self,
        wb_cfg: Dict[str, Any],
        run_name: Optional[str],
        full_config: Optional[Dict[str, Any]],
        resume: bool,
    ) -> None:
        try:
            import wandb  # type: ignore
        except ImportError:
            logger.warning(
                "W&B backend requested but 'wandb' is not installed. "
                "Run: pip install wandb"
            )
            return

        project = wb_cfg.get("project", "voxwhisper")
        entity = wb_cfg.get("entity") or None  # None → W&B default
        tags = wb_cfg.get("tags") or []

        try:
            self._wb_run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags,
                config=full_config,
                resume="allow" if resume else None,
                dir=str(self.run_dir),
            )
            logger.info("W&B run initialised: %s", self._wb_run.url if self._wb_run else "?")
        except Exception as exc:
            logger.warning("W&B init failed (%s) — continuing without W&B", exc)
            self._wb_run = None

    def _init_tensorboard(self, tb_cfg: Dict[str, Any]) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except ImportError:
            logger.warning(
                "TensorBoard backend requested but 'tensorboard' is not installed. "
                "Run: pip install tensorboard"
            )
            return

        log_dir = tb_cfg.get("log_dir") or str(self.run_dir / "tb_logs")
        try:
            self._tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info("TensorBoard SummaryWriter at %s", log_dir)
        except Exception as exc:
            logger.warning("TensorBoard init failed (%s) — continuing without TensorBoard", exc)
            self._tb_writer = None

    def _open(self, resume: bool):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / METRICS_FILENAME

        if resume:
            if path.exists():
                return open(path, "a", encoding="utf-8")  # noqa: SIM115
            # Legacy flat layout: run_YYYYMMDD_HHMMSS.jsonl
            legacy = sorted(self.run_dir.glob("run_*.jsonl"))
            if legacy:
                path = legacy[-1]
                return open(path, "a", encoding="utf-8")  # noqa: SIM115

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
        self._log_external(epoch, metrics, lr)

    def _log_external(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        lr: float,
    ) -> None:
        """Mirror metrics to W&B / TensorBoard if a backend is active."""
        flat = _flatten_metrics(metrics)
        flat["lr"] = lr

        if self._wb_run is not None:
            try:
                self._wb_run.log(flat, step=epoch)
            except Exception as exc:
                logger.debug("W&B log failed: %s", exc)

        if self._tb_writer is not None:
            try:
                for tag, value in flat.items():
                    self._tb_writer.add_scalar(tag, value, global_step=epoch)
            except Exception as exc:
                logger.debug("TensorBoard log failed: %s", exc)

    def close(self) -> None:
        """Flush and close the log file, and finish any external backend."""
        self._file.flush()
        self._file.close()
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
        if self._wb_run is not None:
            try:
                self._wb_run.finish()
            except Exception:
                pass

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
