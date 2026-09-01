# VoxWhisper

Prompt-conditioned 3D volumetric segmentation. The model fuses a **primary** MRI (output space, typically T1) with a **secondary** volume (T2, mean diffusivity, …), aligns them with clinical text prompts, and predicts multi-label masks on the primary grid.

## Layout

| Path | Role |
|------|------|
| [`src/`](src/README.md) | Model, dataset, inference, shared utilities |
| [`config/`](config/README.md) | YAML + structure prompts (`tracts.yaml`, `structures.json`) |
| [`preprocess/`](preprocess/README.md) | Raw → processed NIfTI + prompt embedding cache |
| [`pipeline/`](pipeline/README.md) | End-to-end preprocess / train / evaluate CLIs |

## Quick start

From the repo root (with the project env activated):

```bash
# 1. Offline data prep (volumes → masks → PubMedBERT prompt cache)
python pipeline/preprocess.py --config config/tracts.yaml

# 2. Train
python pipeline/train.py --config config/tracts.yaml

# 3. Full-volume sliding-window evaluation
python pipeline/evaluate.py --config config/tracts.yaml \
  --checkpoint cache/vox_whisper_best.pt --split test
```

Optional AWS download of HCP structurals is documented under [`preprocess/`](preprocess/README.md) (`extract_hcp.py`); it needs extra config keys not present in `tracts.yaml` by default.

## Config

All scripts take `--config` (default: `config/tracts.yaml`). See [`config/README.md`](config/README.md) for every parameter.
