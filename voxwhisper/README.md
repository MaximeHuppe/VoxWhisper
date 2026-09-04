# `voxwhisper/` — model, dataset, inference

Installable package (`pip install -e .`). CLIs live under [`../scripts/`](../scripts/).

## `util/`

Shared plumbing used by both stages. Import from here, not from the package root:

| Module | Role |
|--------|------|
| `util/config.py` | YAML load, path resolve, structure injection |
| `util/seed.py` | Global RNG + DataLoader worker seeds |
| `util/run.py` | Timestamped run directories and checkpoint lookup |
| `util/stage.py` | Phase 1 vs 2 dispatch (model, dataset, cohort, batch unpack) |

`scripts/preprocess.py`, `scripts/train.py`, and `scripts/evaluate.py` take `--config` and follow `model.name`.

## Models

**Phase 1 — `VoxDense`** (`models/vox_dense.py`):

```
T1 ──► encoder ──► bottleneck + skips
              │
 text_embeddings ──► PromptDecoder  (Q=text, KV=T1 bottleneck)
              │
              ▼
 Decoder(+ StageVLFusion)  → [pred×1/4, pred×1/2, pred×1]
```

**Phase 2 — `VoxWhisper`** (`models/vox_whisper.py`): dual T1+FA encoders + `CrossVolumeAttention`. Not trained in Phase 1. Load Phase 1 `encoder` weights into `primary_encoder` later.

| Module | File | Role |
|--------|------|------|
| `VoxDense` | `models/vox_dense.py` | T1-only; `from_config(cfg)` |
| `VoxWhisper` | `models/vox_whisper.py` | Dual T1+FA (Phase 2) |
| `Encoder` | `models/encoder.py` | Stem + downsampling; `(bottleneck, skips)` |
| `PromptDecoder` | `models/attention.py` | Projects PubMedBERT tokens into visual space |
| `CrossVolumeAttention` | `models/attention.py` | Unused until Phase 2 |
| `Decoder` | `models/decoder.py` | Upsample + skip concat + per-prompt mask logits |

Typical shapes (`patch.size = [128,128,128]`, three stride-2 stages):

- Patch / full-res prediction: `[B, N_T, 128, 128, 128]`
- Bottleneck: `[B, 256, 16, 16, 16]`
- Deep-supervision stages: spatial `×1/4`, `×1/2`, `×1`

## Dataset

Offline NIfTI under `data/processed_dense/{subject_id}/`:

| File | Content |
|------|---------|
| `t1.nii.gz` | z-scored 1.25 mm T1 (brain-masked) |
| `mask.nii.gz` | Integer dense labels (0 = background, 1…32 = SynthSeg-style tissues) |

Shared prompt tensor at `cache/prompts_dense.pt` with shape `[N_T, text_dim]`.

`VoxDenseDataset` returns `(volume, text_embeddings, gt_mask)`:

| Mode | Behavior |
|------|----------|
| `training=True` | `train_patches_per_subject` crops; sample `prompts_per_crop` foreground names |
| `training=False` | frozen centers; **all** name prompts |

`VoxWhisperDataset` (T1+FA) remains for Phase 2.

Subject lists come from `data/splits.json`, restricted to the `pretrain` side of `config/subject_split.json`.

## Inference

`predict_dense_volume` runs MONAI sliding-window inference on T1 only. Dual `predict_full_volume` is kept for Phase 2.

## Checkpoints

`save_checkpoint` writes `model_state_dict` plus `encoder_state_dict`. Phase 2 reloads with `load_encoder_state`.
