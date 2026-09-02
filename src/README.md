# `src/` — model, dataset, inference

Core library for VoxWhisper. Entry scripts live under [`../pipeline/`](../pipeline/README.md); this package is what they import.

## Model (`models/`)

Four-step forward: dual 3D encoders → bottleneck cross-volume attention → language–visual prompt decoder → hierarchical decoder with deep supervision.

```
primary_volume ──► primary_encoder ──► bottleneck + skips
secondary_volume ► secondary_encoder ► bottleneck (skips discarded)
                         │
                         ▼
              CrossVolumeAttention  (Q=primary, KV=secondary)
                         │
 text_embeddings ──► PromptDecoder  (Q=text, KV=fused map)
                         │
                         ▼
              Decoder(+ StageVLFusion)  → [pred×1/4, pred×1/2, pred×1]
```

| Module | File | Role |
|--------|------|------|
| `VoxWhisper` | `models/vox_whisper.py` | Wires the pipeline; `from_config(cfg)` |
| `Encoder` | `models/encoder.py` | Stem + downsampling stages; returns `(bottleneck, skips)` |
| `CrossVolumeAttention` | `models/attention.py` | Aligns secondary features onto the primary grid |
| `PromptDecoder` | `models/attention.py` | Projects PubMedBERT tokens into visual space |
| `Decoder` / `StageVLFusionBlock` | `models/decoder.py` | Upsample + skip concat + FiLM-style channel gate + per-prompt mask logits |

**Coordinate space.** Predictions and skip connections are always in **primary** space (T1 by default). Secondary only contributes at the bottleneck via cross-attention.

**Typical shapes** (default `patch.size = [128,128,128]`, three stride-2 stages):

- Patch / full-res prediction: `[B, N_T, 128, 128, 128]`
- Bottleneck: `[B, 128, 16, 16, 16]`
- Deep-supervision stages: spatial `×1/4`, `×1/2`, `×1` (length of `training.deep_supervision_weights` must match)

Positional encodings are sized from `patch.size` and encoder `strides` in `from_config`. If a forward pass sees a different bottleneck size, the PE grid is trilinearly interpolated.

**Secondary channels.** `model.input_channels` feeds the primary encoder; optional `model.secondary_encoder.input_channels` can differ (e.g. multi-channel MD later). Old checkpoints that used `t1_encoder` / `t2_encoder` keys are remapped in `utils/checkpoint.py`.

## How the dataset is generated

Two layers: offline NIfTI on disk, then online patch sampling.

### Offline (see [`../preprocess/README.md`](../preprocess/README.md))

Per subject under `data/processed/{subject_id}/`:

| File | Content |
|------|---------|
| `{primary}.nii.gz` | e.g. `t1.nii.gz` — z-scored structural |
| `{secondary}.nii.gz` | e.g. `t2.nii.gz` — same grid as primary |
| `mask.nii.gz` | Integer labels (0 = background, 1…K = tracts from `structures.json`) |

Plus a shared prompt tensor at `cache/{text_encoder.cache_file}` with shape `[N_T, text_dim]` (one mean-pooled embedding per prompt, including background).

Filenames on disk follow `data.modalities.primary` / `secondary` (and `data.volumes.*`). The Python API always talks about **primary** / **secondary**.

### Online (`dataset.py`)

`VoxWhisperDataset` loads full volumes and returns fixed-size crops:

| Mode | Behavior |
|------|----------|
| `training=True` | `train_patches_per_subject` crops per subject per epoch; 50/50 chance of centering on a positive voxel (`patch.positive_ratio`) vs a random valid center |
| `training=False` | `val_patches_per_subject` frozen centers per subject (seeded by `splits.seed`); stable across epochs |

Each item is `(primary, secondary, text_embeddings, gt_mask)`:

- volumes: `[1, D, H, W]` float
- text: `[N_T, text_dim]` (shared cache, not subject-specific)
- mask: `[N_T, D, H, W]` one-hot float from the integer label map

Subject lists come from a split manifest (`data/splits.json`) when `splits.enabled` is true.

## Inference (`infer.py`)

`predict_full_volume` packs primary+secondary along the channel axis and runs MONAI `sliding_window_inference` with Gaussian blending. Only the full-resolution decoder stage is stitched back to the native grid.

## Utilities (`utils/`)

| Module | Role |
|--------|------|
| `config.py` | Load YAML, resolve paths, inject prompts from `structures.json` |
| `nifti_io.py` | Load/save NIfTI, patch extract, path helpers |
| `metrics.py` | `DiceBCELoss`, deep-supervision helper, Dice scores |
| `checkpoint.py` | Top-k / periodic / latest saves; legacy state-dict remap |
| `validate.py` | Patch and full-volume val used during training |
| `splits.py` | Create or load train/val/test subject lists |
| `seed.py` | Global + DataLoader worker seeding |
