# `config/` — experiment configuration

Main file: [`tracts.yaml`](tracts.yaml). Structure names and clinical phrases: [`structures.json`](structures.json).

Load with `src.utils.config.load_config(path)`. Relative paths are resolved from the repo root. When `data.masks.structures` is set, `load_config` injects:

- `data.prompts` — ordered prompt strings (including background)
- `data.structure_names` — ordered structure keys from `structures.json` (including background)
- `data.patch.positive_labels` — foreground label ids used for positive patch sampling

## `structures.json`

Each key is a structure name; value has:

| Field | Meaning |
|-------|---------|
| `label` | Integer written into `mask.nii.gz` (0 = background) |
| `prompt` | Text fed to PubMedBERT / used as a segmentation channel |

Order of channels follows ascending `label`. Foreground entries without a matching raw mask file are skipped at preprocess time with a warning.

## `tracts.yaml` parameter reference

### `data.paths`

| Key | Meaning |
|-----|---------|
| `raw` | Root of raw subject folders (HCP-style) |
| `processed` | Output of volume/mask preprocess; also training input. The leaf name (e.g. `processed_T1_FA`) becomes the dataset folder under `runs/` |
| `cache` | Shared artifacts only (prompt `.pt` embeddings) |
| `runs` | Root for training outputs: `{runs}/{processed_leaf}/{run_name}/{timestamp}/` |

### `data.modalities`

| Key | Meaning |
|-----|---------|
| `primary` | Modality key used as output space and skip source (e.g. `t1`) |
| `secondary` | Second encoder input (e.g. `t2`; switch to MD by changing this + adding a `volumes` entry) |

These keys must match entries under `data.volumes` and become filenames `{key}.nii.gz` under each processed subject.

### `data.volumes.<modality>`

| Key | Meaning |
|-----|---------|
| `filename` | Raw NIfTI name under `{raw}/{subject}/T1w/` (or `{raw}/{subject}/`) |

`t1` / `t2` here are **on-disk modality ids**, not the Python API names (`primary` / `secondary`).

### `data.masks`

| Key | Meaning |
|-----|---------|
| `source` | Subfolder under each raw subject that holds per-tract NIfTIs (e.g. `tract_masks_1.25`) |
| `structures` | Path to `structures.json` |

### `data.patch`

| Key | Meaning |
|-----|---------|
| `size` | Train / sliding-window crop size `[D, H, W]` |
| `positive_ratio` | Probability of sampling a crop centered on a positive voxel (train); also drives val mix |
| `train_patches_per_subject` | Independent random crops per subject per training epoch |
| `val_patches_per_subject` | Frozen crops per subject in val/test |
| `positive_labels` | Injected from structures (do not edit by hand if using `structures.json`) |

### `data.mock_volume_shape`

Legacy leftover from a removed mock generator. Unused by the current train/preprocess path.

### `preprocessing`

| Key | Meaning |
|-----|---------|
| `normalization` | `zscore` or `minmax` for volume preprocess |
| `zscore_nonzero_only` | If true, mean/std (or min/max) use voxels with `|v| > 0` |

### `text_encoder`

| Key | Meaning |
|-----|---------|
| `model_name` | Hugging Face id (default PubMedBERT) |
| `cache_file` | Filename under `data.paths.cache` for the prompt tensor |

`text_dim` on the model must match the encoder hidden size (768 for base PubMedBERT).

### `model`

| Key | Meaning |
|-----|---------|
| `input_channels` | Primary encoder input channels |
| `secondary_encoder.input_channels` | Secondary encoder channels (defaults to `input_channels` if omitted) |
| `text_dim` | Prompt embedding size |
| `embed_dim` | Shared visual / attention width (must match last encoder channel) |
| `num_heads` | Heads for cross-volume and prompt MHA |
| `encoder.channels` | Channel widths: stem + each stage (last = bottleneck) |
| `encoder.strides` | Downsample stride per stage (length = `len(channels) - 1`) |
| `encoder.kernel_sizes` | Conv kernel per stage |
| `encoder.paddings` | Conv padding per stage |
| `encoder.num_resblocks` | Residual blocks per stage after the transition |

Bottleneck spatial size is `patch.size` divided by the product of `strides` (e.g. 128 / 8 = 16). Changing depth/patch without updating `training.deep_supervision_weights` length will fail at train start.

### `training`

| Key | Meaning |
|-----|---------|
| `run_name` | Experiment folder under `runs/{processed_leaf}/` (filesystem-safe). Override per launch with `--name` |
| `seed` | Global RNG + train patch sampling |
| `batch_size` | Patches per step |
| `epochs` | Cosine schedule `T_max` |
| `learning_rate` | AdamW LR |
| `bce_pos_weight` | BCE positive-class weight. Scalar: background=1, all tracts=`w`. Or a length-`N_T` list. `1` / omitted = unweighted |
| `deep_supervision_weights` | Weight per decoder stage (coarse → fine); length must match decoder stages |
| `dataloader.shuffle` | Shuffle train loader |
| `dataloader.drop_last` | Drop incomplete train batches |
| `dataloader.num_workers` | DataLoader workers |
| `dataloader.pin_memory` | Pin host memory for CUDA |

#### `training.checkpoint`

| Key | Meaning |
|-----|---------|
| `monitor` | `loss` (minimize val loss) or `dice` (maximize Dice) |
| `dice_scope` | `patch` or `volume` when `monitor=dice` |
| `keep` | Keep top-k checkpoints by monitor (`vox_whisper_top{r}.pt`, `vox_whisper_best.pt`) |
| `every` | Also write `vox_whisper_epoch_N.pt` every N epochs if `keep_periodic` |
| `volume_every` | When `dice_scope=volume`, run full-volume Dice every N epochs |
| `keep_periodic` | If false, skip periodic epoch files (top-k / latest still written) |

Every epoch also writes `vox_whisper_latest.pt`.

### `splits`

| Key | Meaning |
|-----|---------|
| `enabled` | If true, use / create the manifest |
| `train_ratio` / `val_ratio` / `test_ratio` | Must sum to 1.0 |
| `seed` | Split shuffle + frozen val patch centers |
| `manifest` | JSON path (`{"train": [...], "val": [...], "test": [...]}`) |

### `inference`

| Key | Meaning |
|-----|---------|
| `overlap` | Sliding-window overlap fraction |
| `mode` | MONAI blend mode (`gaussian` recommended) |
| `sigma_scale` | Gaussian blend sigma scale |
| `sw_batch_size` | Crops per sliding-window step |
| `split` | Default subject split for `evaluate.py` |
| `output_dir` | Predicted NIfTI root |
| `checkpoint` | Optional default `.pt` path (CLI `--checkpoint` overrides) |
| `threshold` | Sigmoid threshold for channel Dice / binarization reporting |
| `save_probabilities` | If true, also save soft maps |
| `progress` | Progress bar during sliding window |

## Keys used by `extract_hcp.py` but not in this YAML

Download is optional and AWS-gated. To use `preprocess/extract_hcp.py` you must add (at least):

- `data.paths.raw_masks` — directory of subjects that already have masks (used to decide who to download)
- `data.download.bucket`, `dataset_prefix`, optional `modalities`, `limit_subjects`, `limit_count`

See [`../preprocess/README.md`](../preprocess/README.md).
