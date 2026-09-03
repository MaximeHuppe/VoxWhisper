"""
Batch experiment launcher for VoxWhisper.

Experiment design
-----------------
Runs are organised in two groups:

GROUP 1 — Baseline → Champion incremental build-up chain  (all 7 re-run for W&B/TB logs)
  Each step adds *exactly one* hyperparameter change on top of the previous step.
  The per-step Δ in Test Mean Dice isolates each parameter's marginal
  contribution to the historical 0.557 → 0.8167 improvement.

  Step  │ Added parameter              │ Run name
  ──────┼──────────────────────────────┼──────────────────────────
    0   │ (baseline)                   │ baseline
    1   │ LR 5e-5 → 1e-4              │ lr1e-4
    2   │ bce_pos_weight 20 → 1        │ chain02_bce_pos_w1
    3   │ eff_batch 2 → 8              │ chain03_eff_batch8
    4   │ warmup 0 → 10 ep             │ chain04_warmup10
    5   │ bce_weight 1.0 → 0.5         │ chain05_bce_weight05
    6   │ large model + patches 4 → 2  │ large_lr1e-4_patches2  (champion)

GROUP 2 — Orthogonal follow-up ablations from champion  (all new)
  Each run is a one-field diff from the champion config, exploring directions
  that are independent of the recipe-improvement chain above.

  ABL-modality_t1t2   : secondary modality FA → T2 (re-test under good recipe)
  ABL-patches4        : train_patches_per_subject 2 → 4  (more crop diversity)
  ABL-patch160        : patch_size 128 → 160  (larger receptive field)

Shared defaults come from config/tracts.yaml; every key in each run dict is
written into the generated YAML so the config is fully self-contained.

Each run: 1) writes a config YAML
          2) trains  (pipeline/train.py)
          3) evaluates on test split  (pipeline/evaluate.py)
          4) updates runs/EXPERIMENT_LOGS.md automatically

Usage:
    python prepared_runs.py                          # run all 10 experiments
    python prepared_runs.py --dry-run                # write configs only
    python prepared_runs.py --only chain02_bce_pos_w1       # run one experiment
    python prepared_runs.py --only chain02_bce_pos_w1,chain03_eff_batch8
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.utils.config import get_project_root, load_config
from src.utils.run import find_latest_run_dir, predictions_dir, run_family_dir

# ---------------------------------------------------------------------------
# Baseline recipe
#
# Mirrors the exact config snapshot of the original baseline run
# (runs/processed_T1_FA/baseline/20260902_152249) — every key must be
# present here so that step() can copy the dict without KeyErrors.
# ---------------------------------------------------------------------------

BASELINE: dict = {
    # ── data ──────────────────────────────────────────────────────────────
    "processed":                 "data/processed_T1_FA",
    "secondary":                 "fa",
    # ── architecture ──────────────────────────────────────────────────────
    "embed_dim":                 128,
    "encoder_channels":          [16, 32, 64, 128],
    # ── training ──────────────────────────────────────────────────────────
    "epochs":                    150,
    "learning_rate":             5e-5,
    "warmup_epochs":             0,
    "batch_size":                2,
    "effective_batch_size":      2,    # no grad accumulation
    "bce_pos_weight":            20,
    "dice_weight":               1.0,
    "bce_weight":                1.0,
    # ── patch / data loading ──────────────────────────────────────────────
    "train_patches_per_subject": 2,
    "val_patches_per_subject":   4,
    "patch_size":                [128, 128, 128],
    # ── preprocessing ─────────────────────────────────────────────────────
    "normalization":             "zscore",
    # ── W&B metadata (semantic role tags — values are auto-generated) ────
    "tags":                      ["chain", "step-0"],
}


def step(prev: dict, run_name: str, **overrides) -> dict:
    """Return a new run dict = previous step + exactly one override field."""
    cfg = {**prev, "run_name": run_name}
    cfg.update(overrides)
    return cfg


def ablation(base: dict, run_name: str, **overrides) -> dict:
    """Return a new run dict = base (usually CHAMPION) + one override field."""
    return step(base, run_name, **overrides)


# ===========================================================================
# GROUP 1 — Incremental build-up chain
# ===========================================================================

# ── Step 0 ──────────────────────────────────────────────────────────────────
# Anchor run.  Establishes the initial recipe before any of the recent
# improvements were introduced.  Test Mean Dice (Ch) ≈ 0.5545.
CHAIN_00_BASELINE = step(BASELINE, "baseline")

# ── Step 1 ──────────────────────────────────────────────────────────────────
# Adds: learning_rate 5e-5 → 1e-4.
# Hypothesis: a 2× higher LR helps AdamW + cross-attention networks escape
# flat loss regions early; AdamW with weight-decay already limits divergence.
# Test Mean Dice (Ch) ≈ 0.6748 (+0.120 vs step 0).
CHAIN_01_LR = step(
    CHAIN_00_BASELINE, "lr1e-4",
    learning_rate=1e-4,
    tags=["chain", "step-1"],
)

# ── Step 2  NEW RUN ──────────────────────────────────────────────────────────
# Adds: bce_pos_weight 20 → 1.
# Hypothesis: the extreme pos-weight (20) caused Run 02 (large_pe_bce20) to
# over-segment every voxel ("all foreground" plateau at val ≈ 0.51).  It is
# positioned HERE — immediately after LR — because it is the most fundamental
# *pathological* issue: with pos_weight=20 the model achieves low BCE loss
# by simply predicting all-foreground, and no other recipe change can fix that.
# Removing it to 1 forces the model to also learn confident background,
# expected to produce the second-largest step-change after LR.
CHAIN_02_BCE_POSW = step(
    CHAIN_01_LR, "chain02_bce_pos_w1",
    bce_pos_weight=1,
    tags=["chain", "step-2"],
)

# ── Step 3  NEW RUN ──────────────────────────────────────────────────────────
# Adds: effective_batch_size 2 → 8  (grad accumulation over 4 micro-batches).
# Hypothesis: a larger effective batch reduces gradient variance in the
# cross-attention layers, smoothing the loss landscape.  This step is placed
# BEFORE warmup because warmup is most effective when the early gradient steps
# are already larger and fewer — its stabilisation benefit is amplified by
# the higher effective batch.
CHAIN_03_EFF_BATCH = step(
    CHAIN_02_BCE_POSW, "chain03_eff_batch8",
    effective_batch_size=8,
    tags=["chain", "step-3"],
)

# ── Step 4  NEW RUN ──────────────────────────────────────────────────────────
# Adds: warmup_epochs 0 → 10.
# Hypothesis: linear LR warmup prevents large gradient spikes in the first
# optimizer steps — especially important at eff_batch=8 where each step is
# a 4-sample gradient and the attention layers' initial random Q/K/V weights
# can produce extreme updates.  Warmup logically follows eff_batch.
CHAIN_04_WARMUP = step(
    CHAIN_03_EFF_BATCH, "chain04_warmup10",
    warmup_epochs=10,
    tags=["chain", "step-4"],
)

# ── Step 5  NEW RUN ──────────────────────────────────────────────────────────
# Adds: bce_weight 1.0 → 0.5.
# Hypothesis: down-weighting BCE relative to Dice in the combined loss shifts
# the gradient signal toward the overlap metric we actually optimise at eval.
# Placed last in the recipe phase — with LR, pos_weight, eff_batch and warmup
# all already tuned — so it is a *refinement* rather than a fundamental fix.
# Expected to show the smallest Δ of the five recipe steps.
CHAIN_05_BCE_WEIGHT = step(
    CHAIN_04_WARMUP, "chain05_bce_weight05",
    bce_weight=0.5,
    tags=["chain", "step-5"],
)

# ── Step 6 ──────────────────────────────────────────────────────────────────
# Adds: encoder [16,32,64,128] → [32,64,128,256], embed_dim 128 → 256,
#        train_patches_per_subject 4 → 2 (historically bundled with the wider
#        model to offset GPU memory usage — same grouping used in Run 02/03).
# Scale-up is the LAST step: once the recipe is fully optimised (LR, loss,
# batch, schedule), increasing capacity is expected to produce the maximum
# return without risk of over-fitting to an unstable training loop.
# This is the current champion.  Test Mean Dice (Ch) = 0.8167.
CHAMPION = step(
    CHAIN_05_BCE_WEIGHT, "large_lr1e-4_patches2",
    embed_dim=256,
    encoder_channels=[32, 64, 128, 256],
    train_patches_per_subject=2,
    tags=["chain", "step-6", "champion"],
)

# ===========================================================================
# GROUP 2 — Orthogonal follow-up ablations from CHAMPION
# ===========================================================================

# ── ABL-modality_t1t2 ────────────────────────────────────────────────────────
# Swaps secondary modality FA → T2 at full champion-recipe quality.
# The earlier T1+T2 run (processed_T1_T2_07/baseline) used the bad baseline
# recipe (lr=5e-5, bce_pos_weight=20, eff_batch=2), so FA appeared +0.055
# better — but that comparison was fully confounded.  This run cleanly
# isolates modality choice under the good recipe.
ABL_MODALITY = ablation(
    CHAMPION, "ABL-modality_t1t2",
    processed="data/processed_T1_T2_07",
    secondary="t2",
    tags=["ablation", "abl-modality"],
)

# ── ABL-patches4 ─────────────────────────────────────────────────────────────
# Restores train_patches_per_subject to 4 (champion uses 2, historically
# reduced to offset the large model's GPU memory usage).
# Hypothesis: more crop diversity per epoch improves segmentation of short /
# curved tracts (UF); expected cost is approximately 2× wall-clock per epoch.
ABL_PATCHES4 = ablation(
    CHAMPION, "ABL-patches4",
    train_patches_per_subject=4,
    tags=["ablation", "abl-patches"],
)

# ── ABL-patch160 ─────────────────────────────────────────────────────────────
# Larger 160³ patch gives the model more spatial context per crop.  Physical
# batch drops to 1 to stay within GPU memory; effective batch remains 8 via
# 8 accumulation steps.
# Hypothesis: long tracts (ATR) benefit most from the wider field of view;
# total training signal per epoch is similar since more context per sample
# partially compensates for fewer samples per step.
ABL_PATCH160 = ablation(
    CHAMPION, "ABL-patch160",
    patch_size=[160, 160, 160],
    batch_size=1,
    # eff_batch stays 8, accum becomes 8 steps instead of 4
    tags=["ablation", "abl-patch-size"],
)

# ===========================================================================
# Active run list — edit here to (de-)activate experiments.
#
# Reference / already-trained entries (CHAIN_00, CHAIN_01, CHAMPION) are
# intentionally excluded so they are NOT re-trained.
# ===========================================================================

RUNS: list[dict] = [
    # ── Group 1: full incremental build-up chain (all steps, re-run for logs) ──
    CHAIN_00_BASELINE,      # step 0: baseline                (anchor)
    CHAIN_01_LR,            # step 1: + lr 1e-4               (largest single gain)
    CHAIN_02_BCE_POSW,      # step 2: + bce_pos_weight 1      (fix pathological loss)
    CHAIN_03_EFF_BATCH,     # step 3: + eff_batch 8           (gradient stability)
    CHAIN_04_WARMUP,        # step 4: + warmup 10             (LR schedule stabilisation)
    CHAIN_05_BCE_WEIGHT,    # step 5: + bce_weight 0.5        (loss balance refinement)
    CHAMPION,               # step 6: + large model + patches 2  (scale-up)
    # ── Group 2: orthogonal follow-ups from champion ───────────────────────
    ABL_MODALITY,           # T1+T2 modality under good recipe
    ABL_PATCHES4,           # more train patches / subject
    ABL_PATCH160,           # 160³ patch size
]


# ---------------------------------------------------------------------------
# Config application
# ---------------------------------------------------------------------------

def _apply(config: dict, run: dict) -> None:
    """Write all run-level overrides into a deep-copied config dict."""
    config["data"]["paths"]["processed"] = run["processed"]
    config["data"]["modalities"]["secondary"] = run["secondary"]

    config["training"]["run_name"] = run["run_name"]
    config["training"]["epochs"] = run["epochs"]
    config["training"]["learning_rate"] = run["learning_rate"]
    config["training"]["warmup_epochs"] = run["warmup_epochs"]
    config["training"]["batch_size"] = run["batch_size"]
    config["training"]["effective_batch_size"] = run.get(
        "effective_batch_size", run["batch_size"]
    )
    config["training"]["bce_pos_weight"] = run["bce_pos_weight"]
    config["training"]["dice_weight"] = run["dice_weight"]
    config["training"]["bce_weight"] = run["bce_weight"]

    config["data"]["patch"]["size"] = list(run["patch_size"])
    config["data"]["patch"]["train_patches_per_subject"] = run["train_patches_per_subject"]
    config["data"]["patch"]["val_patches_per_subject"] = run["val_patches_per_subject"]

    config["model"]["embed_dim"] = run["embed_dim"]
    config["model"]["encoder"]["channels"] = list(run["encoder_channels"])

    config["preprocessing"]["normalization"] = run["normalization"]

    # ── W&B tags: semantic (manual, per-run) + value (auto-generated) ─────
    semantic_tags = list(run.get("tags", []))
    value_tags = _auto_tags(run)
    config["logging"]["wandb"]["tags"] = semantic_tags + value_tags


def _auto_tags(run: dict) -> list[str]:
    """Generate value-encoding W&B tags from the run config dict.

    These are fully automatic so they are always in sync with the actual
    hyperparameters — no risk of a tag drifting from the real value.

    Examples
    --------
    lr-1e-4 · eff-batch-8 · warmup-10 · posw-1 · bce-w-0.5 · dataset-T1-FA
    · modality-fa · model-small · patch-128 · patches-per-subj-4
    """
    tags: list[str] = []

    # Dataset / modality
    dataset_label = os.path.basename(run["processed"])   # e.g. processed_T1_FA
    # Strip leading "processed_" prefix for brevity: T1_FA → T1-FA
    dataset_short = dataset_label.removeprefix("processed_").replace("_", "-")
    tags.append(f"dataset-{dataset_short}")
    tags.append(f"modality-{run['secondary'].lower()}")

    # Model size
    ch0 = run["encoder_channels"][0]
    tags.append("model-large" if ch0 >= 32 else "model-small")

    # Learning rate
    tags.append(f"lr-{_fmt_lr(run['learning_rate'])}")

    # Effective batch size (highlights grad accumulation)
    eff = run.get("effective_batch_size", run["batch_size"])
    tags.append(f"eff-batch-{eff}")

    # Warmup (only tag when non-zero to keep zero-warmup runs uncluttered)
    if run["warmup_epochs"] > 0:
        tags.append(f"warmup-{run['warmup_epochs']}")

    # BCE pos weight
    tags.append(f"posw-{run['bce_pos_weight']}")

    # BCE weight (only when non-default 1.0)
    if run["bce_weight"] != 1.0:
        tags.append(f"bce-w-{run['bce_weight']}")

    # Patch size (first dimension is enough — patches are always cubic)
    tags.append(f"patch-{run['patch_size'][0]}")

    # Patches per subject during training
    tags.append(f"patches-per-subj-{run['train_patches_per_subject']}")

    return tags


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def read_val_patch_dice(run_dir: Path) -> float | None:
    """Return the rank-1 val patch Dice score from vox_whisper_topk.json."""
    topk_path = run_dir / "vox_whisper_topk.json"
    if not topk_path.exists():
        return None
    data = json.loads(topk_path.read_text())
    entries = data.get("entries", [])
    if not entries:
        return None
    rank1 = min(entries, key=lambda e: e["rank"])
    return float(rank1["score"])


def read_test_mean_dice(run_dir: Path, split: str = "test") -> float | None:
    """Return foreground-mean Test Dice from predictions/{split}/dice_summary.csv."""
    summary = run_dir / "predictions" / split / "dice_summary.csv"
    if not summary.exists():
        return None
    means = []
    with open(summary, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["structure"].lower() == "background":
                continue
            means.append(float(row["mean"]))
    return sum(means) / len(means) if means else None


def read_test_per_tract_dice(run_dir: Path, split: str = "test") -> dict[str, float]:
    """Return per-structure mean Test Dice (foreground only)."""
    summary = run_dir / "predictions" / split / "dice_summary.csv"
    if not summary.exists():
        return {}
    result = {}
    with open(summary, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["structure"].lower() == "background":
                continue
            result[row["structure"]] = float(row["mean"])
    return result


# ---------------------------------------------------------------------------
# EXPERIMENT_LOGS.md auto-updater
# ---------------------------------------------------------------------------

_LOG_PATH = get_project_root() / "runs" / "EXPERIMENT_LOGS.md"

# Regex: match a markdown table row that contains the run_name in backticks
# (handles both completed `name` and planned *`name`* formatting)
_ROW_RE_TMPL = r"^\|[^|]*\|\s*\[?\*?`{name}`\*?\]?[^|]*\|.*$"


def _fmt_lr(lr: float) -> str:
    """Format learning rate as a concise string like 1e-4 or 5e-5."""
    s = f"{lr:.0e}"           # e.g. '1e-04'
    s = s.replace("e-0", "e-").replace("e+0", "e")  # '1e-4'
    s = re.sub(r"e-(\d)$", r"e-\1", s)
    return s


def _size_label(channels: list[int]) -> str:
    if channels[0] == 32:
        return "Large"
    if channels[0] == 16:
        return "Normal"
    return "Custom"


def _modality_label(secondary: str) -> str:
    labels = {"fa": "T1 / FA", "t2": "T1 / T2", "b0": "T1 / B0"}
    return labels.get(secondary.lower(), f"T1 / {secondary.upper()}")


def _run_anchor(name: str) -> str:
    """Normalise a run_name to a markdown anchor id (lowercase, keep separators)."""
    return name.lower()


def _build_table_row(
    row_num: int,
    run: dict,
    date_str: str,
    val_dice: float | None,
    test_dice: float | None,
    status: str,
) -> str:
    name = run["run_name"]
    anchor = _run_anchor(name)
    modalities = _modality_label(run["secondary"])
    size = _size_label(run["encoder_channels"])
    channels = str(run["encoder_channels"]).replace(" ", " ")
    patches = f"{run['train_patches_per_subject']} / {run['val_patches_per_subject']}"
    phys = run["batch_size"]
    eff = run.get("effective_batch_size", phys)
    batch_str = f"{phys} / {eff}"
    lr_str = _fmt_lr(run["learning_rate"])
    val_str = f"{val_dice:.4f}" if val_dice is not None else "—"
    test_str = f"{test_dice:.4f}" if test_dice is not None else "—"

    return (
        f"| **{row_num:02d}** | [`{name}`](#{anchor}) | {date_str} | {modalities} | "
        f"{size} | `{channels}` | {patches} | {batch_str} | "
        f"{run['epochs']} | {lr_str} | {run['warmup_epochs']} | "
        f"{run['bce_pos_weight']} | {run['dice_weight']} | {run['bce_weight']} | "
        f"{val_str} | {test_str} | {status} |"
    )


def _build_detail_section(
    run: dict,
    date_str: str,
    run_dir: Path,
    val_dice: float | None,
    test_per_tract: dict[str, float],
    test_mean: float | None,
) -> str:
    name = run["run_name"]
    anchor = _run_anchor(name)
    modalities = _modality_label(run["secondary"])
    size = _size_label(run["encoder_channels"])

    lines = [
        f'<a id="{anchor}"></a>',
        f"### Run `{name}`",
        f"*   **Date:** {date_str}",
        f"*   **Modalities:** {modalities}",
        f"*   **Run directory:** `{run_dir}`",
        "",
        "#### Hyperparameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Model size | {size} — `{run['encoder_channels']}` (embed_dim {run['embed_dim']}) |",
        f"| Encoder channels | `{run['encoder_channels']}` |",
        f"| Patch size | `{run['patch_size']}` |",
        f"| Patches / subj (Tr/Val) | {run['train_patches_per_subject']} / {run['val_patches_per_subject']} |",
        f"| Batch (phys / eff) | {run['batch_size']} / {run.get('effective_batch_size', run['batch_size'])} |",
        f"| Learning rate | {_fmt_lr(run['learning_rate'])} |",
        f"| Warmup epochs | {run['warmup_epochs']} |",
        f"| BCE pos weight | {run['bce_pos_weight']} |",
        f"| Dice weight / BCE weight | {run['dice_weight']} / {run['bce_weight']} |",
        f"| Epochs | {run['epochs']} |",
        f"| Normalisation | {run['normalization']} |",
        "",
    ]

    if val_dice is not None:
        lines += [f"*   **Val Patch Dice (best epoch):** {val_dice:.4f}"]

    if test_per_tract:
        lines += [
            "",
            "#### Test Dice per tract (channel Dice @ threshold 0.5)",
            "",
            "| Tract | Mean | Notes |",
            "|---|---|---|",
        ]
        for tract, mean in test_per_tract.items():
            lines.append(f"| {tract} | {mean:.4f} | |")
        if test_mean is not None:
            lines.append(f"| **Foreground mean** | **{test_mean:.4f}** | |")

    lines.append("")
    return "\n".join(lines)


def update_experiment_log(
    run: dict,
    config: dict,
    *,
    val_dice: float | None,
    test_dice: float | None,
    test_per_tract: dict[str, float],
    status: str,
    date_str: str,
    run_dir: Path,
) -> None:
    """Upsert a completed row into runs/EXPERIMENT_LOGS.md."""
    log_path = _LOG_PATH
    if not log_path.exists():
        print(f"  Warning: {log_path} not found — skipping log update")
        return

    content = log_path.read_text(encoding="utf-8")
    name = run["run_name"]

    # ── Upsert master table row ──────────────────────────────────────────────
    row_re = re.compile(
        _ROW_RE_TMPL.format(name=re.escape(name)),
        re.MULTILINE,
    )
    match = row_re.search(content)

    if match:
        # Extract existing row number from the matched line
        existing_row = match.group(0)
        num_match = re.search(r"\*\*?(\d+)\*\*?", existing_row)
        row_num = int(num_match.group(1)) if num_match else 99
        new_row = _build_table_row(row_num, run, date_str, val_dice, test_dice, status)
        content = content[: match.start()] + new_row + content[match.end():]
    else:
        # Count existing numbered rows to assign the next row number
        existing_nums = re.findall(r"\|\s*\*\*(\d+)\*\*\s*\|", content)
        row_num = (max(int(n) for n in existing_nums) + 1) if existing_nums else 1
        new_row = _build_table_row(row_num, run, date_str, val_dice, test_dice, status)
        # Insert before the first blank line after the table header separator
        table_end = re.search(r"(\n\n)", content[content.find("|---|"):])
        if table_end:
            insert_at = content.find("|---|") + table_end.start() + 1
            content = content[:insert_at] + new_row + "\n" + content[insert_at:]
        else:
            content += "\n" + new_row + "\n"

    # ── Append or replace detail section ───────────────────────────────────
    anchor = _run_anchor(name)
    detail = _build_detail_section(
        run, date_str, run_dir, val_dice, test_per_tract, test_dice
    )
    anchor_re = re.compile(
        rf'<a id="{re.escape(anchor)}"></a>.*?(?=\n---\n\n<a id="|$)',
        re.DOTALL,
    )
    if anchor_re.search(content):
        content = anchor_re.sub(detail.rstrip(), content, count=1)
    else:
        if not content.endswith("\n"):
            content += "\n"
        content += "\n" + detail

    log_path.write_text(content, encoding="utf-8")
    print(f"  Updated EXPERIMENT_LOGS.md → row {row_num:02d}  {name}")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch VoxWhisper experiment launcher"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write config YAMLs only; skip training, evaluation and log update",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="NAME[,NAME,...]",
        help="Comma-separated run_name(s) to execute; all others are skipped",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    only_set = (
        {n.strip() for n in args.only.split(",")} if args.only else None
    )

    baseline_config = load_config("config/baseline.yaml")

    active = [r for r in RUNS if only_set is None or r["run_name"] in only_set]
    if only_set:
        missing = only_set - {r["run_name"] for r in active}
        if missing:
            print(f"Warning: --only names not found in RUNS: {sorted(missing)}")

    print(f"Preparing {len(active)} experiment(s)  "
          f"{'(dry-run — no training)' if args.dry_run else ''}\n")

    for i, run in enumerate(active, start=1):
        dataset_label = os.path.basename(run["processed"])
        config_path = (
            f"config/tracts_{dataset_label}_{run['secondary']}_{run['run_name']}.yaml"
        )

        config = copy.deepcopy(baseline_config)
        _apply(config, run)

        with open(config_path, "w") as f:
            yaml.dump(
                config, f,
                default_flow_style=False, allow_unicode=True, sort_keys=False,
            )

        sep = "─" * 64
        eff = run.get("effective_batch_size", run["batch_size"])
        print(sep)
        print(f"  [{i}/{len(active)}]  {run['run_name']}  ·  {dataset_label}  ·  {run['secondary']}")
        print(sep)
        print(f"  config        : {config_path}")
        print(f"  epochs        : {run['epochs']}   lr: {_fmt_lr(run['learning_rate'])}"
              f"   warmup: {run['warmup_epochs']} ep")
        print(f"  batch         : {run['batch_size']} (eff {eff})   "
              f"bce_pos_w: {run['bce_pos_weight']}")
        print(f"  loss weights  : dice={run['dice_weight']}  bce={run['bce_weight']}")
        print(f"  patches/subj  : {run['train_patches_per_subject']}   "
              f"patch_size: {run['patch_size']}   "
              f"channels: {run['encoder_channels']}   "
              f"embed_dim: {run['embed_dim']}")
        print()

        if args.dry_run:
            print(f"  [dry-run] Skipping train / evaluate / log update.\n")
            continue

        # ── Train ────────────────────────────────────────────────────────────
        try:
            print("  >>> TRAIN")
            _run([sys.executable, "pipeline/train.py", "--config", config_path])
        except subprocess.CalledProcessError as exc:
            print(f"\n  ERROR: training failed (exit code {exc.returncode})")
            _log_failure(run, f"Train failed (exit {exc.returncode})")
            print(f"  Continuing to next experiment.\n")
            continue

        # ── Evaluate ─────────────────────────────────────────────────────────
        try:
            print("  >>> EVALUATE (test split)")
            _run([
                sys.executable, "pipeline/evaluate.py",
                "--config", config_path, "--split", "test",
            ])
        except subprocess.CalledProcessError as exc:
            print(f"\n  ERROR: evaluation failed (exit code {exc.returncode})")
            _log_failure(run, f"Evaluate failed (exit {exc.returncode})")
            print(f"  Continuing to next experiment.\n")
            continue

        # ── Auto-update EXPERIMENT_LOGS.md ───────────────────────────────────
        try:
            print("  >>> UPDATE EXPERIMENT LOG")
            _update_log_for_run(run, config)
        except Exception as exc:  # noqa: BLE001 — log update must never abort the batch
            print(f"  Warning: log update failed: {exc}")

        print(f"  [{i}/{len(active)}]  Done.\n")

    print("All experiments finished.")


def _find_run_dir(run: dict, config: dict) -> Path | None:
    """Locate the most recent timestamped run directory for this run."""
    try:
        family = run_family_dir(config)
        return find_latest_run_dir(family)
    except Exception:
        return None


def _log_failure(run: dict, reason: str) -> None:
    """Mark the run row in the log as Failed."""
    try:
        log_path = _LOG_PATH
        if not log_path.exists():
            return
        content = log_path.read_text(encoding="utf-8")
        name = run["run_name"]
        row_re = re.compile(
            _ROW_RE_TMPL.format(name=re.escape(name)), re.MULTILINE
        )
        match = row_re.search(content)
        if match:
            existing = match.group(0)
            updated = re.sub(r"\|\s*[^|]*$", f"| Failed — {reason} |", existing)
            content = content[: match.start()] + updated + content[match.end():]
            log_path.write_text(content, encoding="utf-8")
    except Exception:
        pass


def _update_log_for_run(run: dict, config: dict) -> None:
    """Collect metrics from disk and call update_experiment_log."""
    run_dir = _find_run_dir(run, config)
    if run_dir is None:
        print("  Warning: could not locate run directory — skipping log update")
        return

    val_dice = read_val_patch_dice(run_dir)
    test_dice = read_test_mean_dice(run_dir)
    test_per_tract = read_test_per_tract_dice(run_dir)
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    update_experiment_log(
        run,
        config,
        val_dice=val_dice,
        test_dice=test_dice,
        test_per_tract=test_per_tract,
        status="Completed",
        date_str=date_str,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
