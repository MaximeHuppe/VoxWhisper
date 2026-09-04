# `config/` — experiment configuration

Phase 1 file: [`voxdense.yaml`](voxdense.yaml). Dense structure names: [`structures_dense.json`](structures_dense.json). Phase 2 will use a separate `voxwhisper.yaml` (not in this branch).

Load with `voxwhisper.util.config.load_config(path)`. Defaults to `config/voxdense.yaml`. Relative paths are resolved from the repo root. When `data.masks.structures` is set, `load_config` injects:

- `data.prompts` — ordered prompt strings (including background)
- `data.structure_names` — ordered structure keys (including background)
- `data.patch.positive_labels` — foreground label ids used for positive patch sampling

## `structures_dense.json`

Each key is a structure name; value has:

| Field | Meaning |
|-------|---------|
| `label` | Integer written into `mask.nii.gz` (0 = background, 1–32 = SynthSeg-style tissues) |
| `prompt` | Text fed to PubMedBERT / used as a segmentation channel |

Order of channels follows ascending `label`. wmparc gyral parcels and skull/bone are **not** classes.

## `voxdense.yaml` parameter reference

### `data.paths`

| Key | Meaning |
|-----|---------|
| `raw` | Root of raw HCP subject folders |
| `processed` | Output of T1/mask preprocess (`data/processed_dense`). The leaf name becomes the dataset folder under `runs/` |
| `cache` | Prompt embedding `.pt` files |
| `runs` | Training outputs: `{runs}/{processed_leaf}/{run_name}/{timestamp}/` |

### `data.volumes`

| Key | Meaning |
|-----|---------|
| `t1.filename` | Native 1.25 mm T1 (`T1w_acpc_dc_restore_1.25.nii.gz`) — not resampled |
| `brainmask.filename` | FreeSurfer brainmask, NN onto the T1 grid |
| `wmparc.filename` | FreeSurfer wmparc, NN onto the T1 grid then collapsed |

### `data.masks`

| Key | Meaning |
|-----|---------|
| `structures` | Path to `structures_dense.json` |

### `data.nerve_masks`

| Key | Meaning |
|-----|---------|
| `root` | Tree used only for holdout detection (e.g. `data/raw`) |
| `source` | Subfolder that marks Phase 2 holdout subjects (`tract_masks_1.25`) |

### `data.patch`

| Key | Meaning |
|-----|---------|
| `size` | Train / sliding-window crop size `[D, H, W]` |
| `positive_ratio` | Probability of sampling a crop centered on a positive voxel |
| `train_patches_per_subject` | Independent random crops per subject per training epoch |
| `val_patches_per_subject` | Frozen crops per subject in val/test |
| `prompts_per_crop` | Foreground name prompts sampled per **train** crop. Background (label 0) is **always** prepended as channel 0 so Dice/BCE stay aligned with full-prompt validation. `0` disables sampling and uses every prompt. Val always uses all prompts. |
| `positive_labels` | Injected from structures |

### `preprocessing`

| Key | Meaning |
|-----|---------|
| `zscore_nonzero_only` | If true, mean/std use voxels with `|v| > 0` |
| `apply_brainmask` | Zero T1 and labels outside the NN-resampled brainmask |

### `text_encoder`

| Key | Meaning |
|-----|---------|
| `model_name` | Hugging Face id (PubMedBERT) |
| `cache_file` | Filename under `data.paths.cache` (`prompts_dense.pt`) |

`model.text_dim` must match the encoder hidden size (768 for base PubMedBERT).

### `model`

| Key | Meaning |
|-----|---------|
| `name` | `VoxDense` |
| `input_channels` | T1 encoder input channels |
| `text_dim` | Prompt embedding size |
| `embed_dim` | Shared visual / attention width |
| `num_heads` | Heads for PromptDecoder MHA |
| `encoder.channels` | Channel widths: stem + each stage (last = bottleneck) |
| `encoder.strides` | Downsample stride per stage |

Bottleneck spatial size is `patch.size` divided by the product of `strides`. Changing depth/patch without updating `training.deep_supervision_weights` length will fail at train start.

### `training`

| Key | Meaning |
|-----|---------|
| `run_name` | Experiment folder under `runs/{processed_leaf}/` |
| `seed` | Global RNG + train patch sampling |
| `batch_size` | Patches per step |
| `epochs` | Cosine schedule `T_max` |
| `learning_rate` | AdamW LR |
| `bce_weight` | Multiplier on the BCE term |
| `exclude_background` | Soft Dice skips channel 0 when true (default). Dataset sampling always puts background first. |
| `deep_supervision_weights` | Weight per decoder stage (coarse → fine) |
| `checkpoint.monitor` | `loss` or `dice` |
| `checkpoint.keep` | Keep top-k checkpoints |

Every epoch also writes `vox_whisper_latest.pt`. The payload includes `encoder_state_dict` for Phase 2 reload.

### `splits`

| Key | Meaning |
|-----|---------|
| `train_ratio` / `val_ratio` / `test_ratio` | Must sum to 1.0 (among **pretrain** subjects) |
| `seed` | Split shuffle + frozen val patch centers |
| `manifest` | JSON path (`{"train": [...], "val": [...], "test": [...]}`) |
| `subject_split` | `{"pretrain": [...], "nerve": [...]}` written by preprocess |

### `logging` / `inference`

Same JSONL + optional W&B / TensorBoard contract as before. Evaluate uses T1-only sliding windows (`predict_dense_volume`).
