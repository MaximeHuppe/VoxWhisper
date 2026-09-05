# VoxWhisper

Two-stage brain-MRI specialist. **Phase 1 (this branch)** pretrains `VoxDense`: a T1-only language-conditioned UNet on FreeSurfer dense structures. Phase 2 (later) freezes that T1 encoder, adds an FA encoder, and segments unseen nerves from spatial descriptions.

Do **not** train FA on FreeSurfer labels.

---

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Preprocess HCP T1 + wmparc onto the 1.25 mm grid
python scripts/preprocess.py --config config/voxdense.yaml

# 3. Train VoxDense
python scripts/train.py --config config/voxdense.yaml

# 4. Evaluate
python scripts/evaluate.py --split test
```

Edit [`config/voxdense.yaml`](config/voxdense.yaml) between runs. Structure names live in [`config/structures_dense.json`](config/structures_dense.json).

---

## Phase 1 model

`T1 volume → encoder → PromptDecoder (frozen PubMedBERT names) → Decoder`

The checkpoint stores `encoder_state_dict` so Phase 2 can reload the T1 encoder into dual `VoxWhisper`. The dual class stays in the tree but is not trained here.

---

## Tunable knobs (`config/voxdense.yaml`)

| Key | Description |
|---|---|
| `training.learning_rate` | AdamW peak LR after warm-up |
| `training.warmup_epochs` | Linear LR warm-up duration |
| `training.epochs` | Total training epochs |
| `training.batch_size` | Physical batch size |
| `training.bce_weight` | BCE term multiplier (`loss = Dice + bce_weight × BCE`) |
| `training.exclude_background` | Soft Dice skips channel 0 (must be background; default true) |
| `training.ignore_empty_targets` | Soft Dice skips empty-target channels (default true) |
| `training.deep_supervision_weights` | Per-scale weights `[coarse, mid, fine]` |
| `training.seed` | Global RNG seed |
| `training.checkpoint.monitor` | `dice` (default) or `loss` |
| `training.checkpoint.keep` | Number of top-k checkpoints to keep |
| `model.encoder.channels` | Encoder feature channels, e.g. `[32, 64, 128, 256]` |
| `model.embed_dim` | Bottleneck / attention dimension |
| `model.num_heads` | Attention heads in PromptDecoder |
| `data.patch.size` | Training patch size (D × H × W voxels) |
| `data.patch.train_patches_per_subject` | Crops sampled per subject per epoch |
| `data.patch.prompts_per_crop` | Foreground name prompts per **train** crop (background always prepended as channel 0; `0` = all prompts) |
| `data.patch.positive_ratio` | Fraction of crops centred on a foreground voxel |
| `splits.*` | Train/val/test ratios; `subject_split` holds pretrain vs nerve holdout |
| `logging.wandb.*` | W&B project, entity, tags |

---

## Preprocessing

HCP FreeSurfer outputs already on disk under `data/raw/{subject_id}/`. Do not re-run `recon-all`. Work in **1.25 mm** space (`T1w_acpc_dc_restore_1.25.nii.gz`). `wmparc` and `brainmask_fs` are nearest-neighbour resampled onto that grid, then collapsed to a SynthSeg/OpenMind ~32-tissue set.

Subjects with `nerve_masks_1.25` are held out for Phase 2 (`config/subject_split.json`). Training uses the pretrain split only. Do not train from `data/processed_T1_FA/`.

```bash
python scripts/preprocess.py
```

Writes `data/processed_dense/{subject}/t1.nii.gz` + `mask.nii.gz` and `cache/prompts_dense.pt`.

---

## Project layout

```
VoxWhisper/
├── config/
│   ├── voxdense.yaml          ← Phase 1 experiment file
│   ├── structures_dense.json  ← ~32 SynthSeg-style names + prompts
│   └── subject_split.json     ← written by preprocess (pretrain vs nerve)
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── compute_fa.py          ← Phase 2; not required to train VoxDense
│
├── voxwhisper/
│   ├── util/                  ← config, seed, run dirs, phase dispatch
│   ├── models/vox_dense.py    ← Phase 1 model
│   ├── models/vox_whisper.py  ← dual T1+FA class (Phase 2)
│   ├── data/                  ← Python only (not NIfTIs)
│   └── training/
│
├── tests/
├── cache/                     ← prompts_dense.pt (gitignored)
└── data/                      ← on-disk volumes (gitignored processed/raw)
```

---

## Citation

```
@misc{voxwhisper2026,
  title        = {VoxWhisper: Language-Grounded 3D Brain MRI Segmentation},
  author       = {Huppe, Maxime},
  year         = {2026},
}
```
