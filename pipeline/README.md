# `pipeline/` — preprocess, train, evaluate

Thin CLIs that wire [`../src/`](../src/README.md) and [`../preprocess/`](../preprocess/README.md) together. Always pass (or rely on the default) `--config config/tracts.yaml`.

## Commands

### Preprocess

```bash
python pipeline/run_preprocess.py --config config/tracts.yaml
```

Runs in order:

1. `preprocess_volumes` — normalized full-res primary/secondary NIfTIs
2. `preprocess_masks` — integer `mask.nii.gz` on the T1 grid
3. Drop processed subject folders missing primary, secondary, or `mask.nii.gz`
4. `cache_embedding` — PubMedBERT prompt tensor under `data.paths.cache`

Does **not** download HCP data; use `preprocess/extract_hcp.py` separately if needed.

### Train

```bash
python pipeline/train.py --config config/tracts.yaml
# optional:
python pipeline/train.py --config config/tracts.yaml --name high_bce
python pipeline/train.py --config config/tracts.yaml --resume
python pipeline/train.py --config config/tracts.yaml --run-dir runs/processed_T1_FA/baseline/20260902_130304 --resume
```

- Builds train (and optional val) loaders from `splits`
- Instantiates `VoxWhisper.from_config`, AdamW + cosine LR
- Deep-supervision Dice+BCE (`training.deep_supervision_weights`)
- Each epoch: train loss; if val exists, patch val loss + patch Dice; optionally full-volume Dice when `checkpoint.dice_scope=volume`
- Creates a timestamped run directory under `runs/{processed_leaf}/{run_name}/`

**Each run directory contains:**

| File | When |
|------|------|
| `config.yaml` | Snapshot of the resolved config at launch |
| `meta.json` | Time, git SHA, seed, argv, torch/CUDA |
| `metrics.jsonl` | One JSON object per epoch |
| `vox_whisper_latest.pt` | Every epoch |
| `vox_whisper_e{epoch:03d}.pt` | When score enters top-`keep` |
| `vox_whisper_top{r}.pt` | Symlinks to current top-k ranks |
| `vox_whisper_best.pt` | Symlink to rank-1 top-k |
| `vox_whisper_epoch_N.pt` | Every `checkpoint.every` epochs if `keep_periodic` |
| `vox_whisper_topk.json` | Manifest of top-k scores |

`--resume` continues the newest timestamp child under `{runs}/{dataset}/{run_name}/`. `--name` overrides `training.run_name` for one launch.

Legacy keys (`t1_encoder` / `t2_encoder` / `pos_t1`) are remapped on load so older `.pt` files still work.

### Evaluate

```bash
python pipeline/evaluate.py --config config/tracts.yaml --split test
# or pin a run / checkpoint:
python pipeline/evaluate.py --config config/tracts.yaml \
  --run-dir runs/processed_T1_FA/baseline/20260902_130304 --split test
python pipeline/evaluate.py --config config/tracts.yaml \
  --checkpoint runs/processed_T1_FA/baseline/20260902_130304/vox_whisper_best.pt
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config` | `config/tracts.yaml` | Experiment YAML |
| `--checkpoint` | latest best/top1 under `runs/{dataset}/{run_name}/`, or `inference.checkpoint` | Weights |
| `--run-dir` | — | Timestamped run folder (picks best/top1/latest inside) |
| `--name` | `training.run_name` | Override when auto-discovering the latest run |
| `--split` | `inference.split` | `train` / `val` / `test` from the manifest |
| `--output-dir` | `inference.output_dir` | Per-subject prediction folders |
| `--subject` | — | Evaluate a single subject id instead of a split |

Sliding-window settings (`overlap`, `mode`, `sigma_scale`, `sw_batch_size`, …) come from the `inference` section of the config. Outputs include `pred_labels.nii.gz` (and optionally probabilities) under `{output_dir}/{subject_id}/`.

## Typical flow

```
pipeline/run_preprocess.py  →  data/processed_* + cache/prompts_*.pt
pipeline/train.py           →  runs/{dataset}/{run_name}/{timestamp}/
pipeline/evaluate.py        →  data/predictions/{sid}/pred_labels.nii.gz
```
