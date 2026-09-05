"""Training logger: JSONL + stdout table, with optional W&B mirroring."""
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
    """Recursively flatten nested metric dicts using '/' as separator."""
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

    Optionally mirrors metrics to Weights & Biases when ``log_cfg``
    (from ``config["logging"]``) has ``backend: wandb``.

    Parameters
    ----------
    run_dir      : directory where ``metrics.jsonl`` is written.
    total_epochs : total number of training epochs (used for the epoch column width).
    resume       : if ``True``, append to an existing metrics file.
    log_cfg      : ``config["logging"]`` dict.
    run_name     : human-readable run identifier forwarded to W&B.
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
        if backend == "wandb":
            self._init_wandb(log_cfg.get("wandb", {}), run_name, full_config, resume)
        elif backend not in ("none", ""):
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
        entity = wb_cfg.get("entity") or None
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

    def _open(self, resume: bool):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / METRICS_FILENAME
        if resume and path.exists():
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
        """Write one JSONL entry and print a stdout table row."""
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
        self._log_wandb(epoch, metrics, lr)

    def _log_wandb(self, epoch: int, metrics: Dict[str, Any], lr: float) -> None:
        if self._wb_run is None:
            return
        flat = _flatten_metrics(metrics)
        flat["lr"] = lr
        try:
            self._wb_run.log(flat, step=epoch)
        except Exception as exc:
            logger.debug("W&B log failed: %s", exc)

    def close(self) -> None:
        """Flush and close the log file, and finish W&B if active."""
        self._file.flush()
        self._file.close()
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
        """Scan the log file and return the best epoch for each metric."""
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
