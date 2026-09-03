# `preprocess/` — raw data → training-ready NIfTI

Turn HCP-style raw volumes and tract masks into full-resolution processed subjects plus a cached prompt embedding tensor. Prefer the orchestrator:

```bash
python pipeline/run_preprocess.py --config config/tracts.yaml
```

which runs volumes → masks → embeddings in order. Individual scripts below are the same steps.

## Expected raw layout

```
data/raw/{subject_id}/
  T1w/
    T1w_acpc_dc_restore.nii.gz      # name from data.volumes.t1.filename
    T2w_acpc_dc_restore.nii.gz      # name from data.volumes.t2.filename
  tract_masks_1.25/                 # name from data.masks.source
    ATR_left.nii.gz
    ATR_right.nii.gz
    ...                             # keys from structures.json (foreground)
```

`resolve_raw_volume_path` also accepts `{raw}/{subject_id}/{filename}` without the `T1w/` subfolder.

## Scripts

### `extract_hcp.py` (optional)

Downloads structural volumes from an S3 HCP-style bucket for subjects that already have masks under `data.paths.raw_masks`. Requires AWS credentials and **extra YAML keys** not shipped in `tracts.yaml`:

- `data.paths.raw_masks`
- `data.download.bucket`, `dataset_prefix`
- optional `data.download.modalities`, `limit_subjects`, `limit_count`

This is not part of `pipeline/run_preprocess.py`. Documented so you do not expect a one-line toggle in the default config.

### `compute_fa.py` / `compute_dec_fa.py` (Step 0)

Fit a DTI model on each subject's `Diffusion/data.nii.gz` and write a map at the subject root:

| Script | Output | Shape |
|--------|--------|-------|
| `compute_fa.py` | `dti_FA.nii.gz` | 3D scalar FA |
| `compute_dec_fa.py` | `dti_DEC_FA.nii.gz` | 4D RGB DEC-FA `(D, H, W, 3)` |

DEC-FA colours the principal eigenvector by FA (Pajevic & Pierpaoli): R = \|e1ₓ\|·FA, G = \|e1ᵧ\|·FA, B = \|e1_z\|·FA. Do not z-score it (`data.volumes.dec_fa.normalize: false`). `compute_dec_fa.py` also writes scalar FA if it is missing.

```bash
python preprocess/compute_dec_fa.py --config config/tracts.yaml
python preprocess/compute_dec_fa.py --config config/tracts.yaml --subject 599469 --workers 4
```

Then point the experiment at it:

```yaml
data.modalities.secondary: dec_fa
data.paths.processed: data/processed_T1_DEC_FA
model.secondary_encoder.input_channels: 3
```

### `preprocess_volumes.py`

For each subject and each of `data.modalities.primary` / `secondary`:

1. Load the raw NIfTI named in `data.volumes.<modality>.filename`
2. Intensity-normalize (`preprocessing.normalization`, optionally nonzero-only)
3. Write `data/processed/{subject_id}/{modality}.nii.gz` at **full resolution** (no cropping)

Patches are sampled later at train time.

### `preprocess_masks.py`

Builds one integer label map per subject on the **T1 / primary anatomical grid**:

1. Load the T1 raw volume as the resampling reference (currently keyed as `data.volumes.t1`)
2. For each foreground structure in `structures.json`, load `{masks.source}/{name}.nii.gz`
3. Nearest-neighbor resample onto T1 (`order=0`), threshold > 0, write `label`
4. Save `data/processed/{subject_id}/mask.nii.gz`

Overlapping voxels keep the last written structure (a warning is printed).

### `cache_embedding.py`

Tokenizes `data.prompts` with `text_encoder.model_name`, mean-pools the last hidden state per prompt, and saves `[N_T, text_dim]` to `cache/{text_encoder.cache_file}`. The text encoder is frozen and not part of the segmentation network.

## Output layout

```
data/processed/{subject_id}/
  t1.nii.gz          # or whatever modalities.primary is
  t2.nii.gz          # or modalities.secondary
  mask.nii.gz

cache/
  prompts_tracts.pt  # text_encoder.cache_file
```

Training then reads these via `src.dataset.VoxWhisperDataset` (see [`../src/README.md`](../src/README.md)).
