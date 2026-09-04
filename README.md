# VoxWhisper

3D multi-modal, language-grounded volumetric segmentation of white matter tracts.  
Fuses a T1 and FA volume, aligns them with clinical text prompts (PubMedBERT), and
produces prompt-conditioned segmentation masks.

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Edit hyperparameters
nano config/best_config.yaml

# 3. Train
python scripts/train.py

# 4. Evaluate
python scripts/evaluate.py --split test
```

---

## Tunable knobs (config/best_config.yaml)

All experiment parameters live in one YAML file. No Python changes needed between runs.

| Key | Description |
|---|---|
| `training.learning_rate` | AdamW peak LR after warm-up |
| `training.warmup_epochs` | Linear LR warm-up duration |
| `training.epochs` | Total training epochs |
| `training.batch_size` | Physical batch size |
| `training.bce_weight` | BCE term multiplier (`loss = Dice + bce_weight × BCE`) |
| `training.deep_supervision_weights` | Per-scale weights `[coarse, mid, fine]` |
| `training.seed` | Global RNG seed |
| `training.checkpoint.monitor` | `dice` (default) or `loss` |
| `training.checkpoint.keep` | Number of top-k checkpoints to keep |
| `model.encoder.channels` | Encoder feature channels, e.g. `[32, 64, 128, 256]` |
| `model.embed_dim` | Bottleneck / attention dimension |
| `model.num_heads` | Attention heads in CrossVolumeAttention + PromptDecoder |
| `data.patch.size` | Training patch size (D × H × W voxels) |
| `data.patch.train_patches_per_subject` | Crops sampled per subject per epoch |
| `data.patch.positive_ratio` | Fraction of crops centred on a foreground voxel |
| `splits.*` | Train/val/test ratios and random seed |
| `logging.wandb.*` | W&B project, entity, tags |

---

## Preprocessing pipeline

Run once before training. Data must be in `data/raw/{subject_id}/`.

```bash
# Step 0 — compute FA maps from raw diffusion (HCP layout)
python scripts/compute_fa.py --workers 4

# Steps 1–4 — normalise volumes, build label maps, cache embeddings
python scripts/preprocess.py
```

---

## Project layout

```
VoxWhisper/
├── config/
│   ├── best_config.yaml       ← only config file; edit this between runs
│   └── structures.json        ← tract names, labels, and prompts
│
├── scripts/                   ← thin CLI wrappers
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── compute_fa.py
│
├── voxwhisper/                ← Python package (pip install -e .)
│   ├── config.py              ← YAML loading, path helpers
│   ├── run.py                 ← run directory management
│   ├── seed.py                ← reproducible seeding
│   ├── infer.py               ← sliding-window inference
│   │
│   ├── models/
│   │   ├── vox_whisper.py     ← top-level model
│   │   ├── encoder.py
│   │   ├── attention.py
│   │   └── decoder.py
│   │
│   ├── data/
│   │   ├── dataset.py         ← patch-based PyTorch dataset
│   │   ├── nifti_io.py        ← NIfTI I/O helpers
│   │   ├── splits.py          ← train/val/test split management
│   │   └── preprocess/
│   │       ├── fa.py          ← DTI FA map computation
│   │       ├── volumes.py     ← z-score normalisation
│   │       ├── masks.py       ← tract mask → integer label map
│   │       └── embeddings.py  ← PubMedBERT prompt caching
│   │
│   └── training/
│       ├── loop.py            ← epoch loop + patch validation
│       ├── loss.py            ← DiceBCELoss + deep_supervision_loss
│       ├── metrics.py         ← Dice score utilities
│       ├── checkpoint.py      ← top-k checkpoint management
│       └── logger.py          ← JSONL + W&B logging
│
├── tests/
├── pyproject.toml
└── requirements.txt
```

---

## Model architecture

Fixed four-step pipeline:

1. **Dual encoders** — independent 3D UNet encoders for T1 (primary) and FA (secondary)
2. **CrossVolumeAttention** — primary bottleneck tokens query secondary tokens via MHA, producing a spatially-aligned fused feature map
3. **PromptDecoder** — frozen PubMedBERT embeddings query the fused visual map, producing language-aligned queries per tract
4. **Hierarchical decoder** — skip connections + channel modulation at 3 scales with deep supervision

---

## Citation

```
@misc{voxwhisper2026,
  title        = {VoxWhisper: Language-Grounded 3D White Matter Tract Segmentation},
  author       = {Huppe, Maxime},
  year         = {2026},
}
```
