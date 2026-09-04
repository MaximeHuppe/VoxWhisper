<!-- =========================================================================
     EXPERIMENT_LOGS.md
     Auto-updated by prepared_runs.py after every train+evaluate step.
     Currently retaining completed results for chain steps 0–2 only.
     ========================================================================= -->

## How this file is updated

`python prepared_runs.py` trains and evaluates each entry in `RUNS`, then
calls `update_experiment_log()` immediately after each run's evaluate step.
That function:
1. Reads **Val Patch Dice** from `vox_whisper_topk.json` (rank-1 score).
2. Reads **Test Mean Dice (Ch)** from `predictions/test/dice_summary.csv`
   as the foreground-only mean of the `mean` column (background row excluded).
3. Upserts the matching row in the Master Table by `run_name`.
4. Appends / replaces the run's detail section.

---

## 1. Master Experiment Table

> **Evaluation Convention:** All Test Dice scores are channel Dice at threshold 0.5
> (`channel_dice` with independent per-channel Sigmoid thresholding across foreground
> tracts), matching the multi-label VLM formulation.  Val Patch Dice = rank-1 score
> in `vox_whisper_topk.json`.  Test Mean Dice (Ch) = foreground-only mean of
> `predictions/test/dice_summary.csv`.

| # | Run Name | Date | Modalities | Size | Encoder Channels | Patches / Subj (Tr/Val) | Batch (Phys/Eff) | Ep | Base LR | Warmup | BCE Pos W | Dice W | BCE W | Val Patch Dice | Test Mean Dice (Ch) | Status / Tag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **01** | [`baseline`](#baseline) | 2026-09-03 | T1 / FA | Normal | `[16, 32, 64, 128]` | 2 / 4 | 2 / 2 | 150 | 5e-5 | 0 | 20 | 1.0 | 1.0 | 0.5149 | 0.5271 | Completed — chain step 0 |
| **02** | [`chain01_lr1e-4`](#chain01_lr1e-4) | 2026-09-03 | T1 / FA | Normal | `[16, 32, 64, 128]` | 2 / 4 | 2 / 2 | 150 | 1e-4 | 0 | 20 | 1.0 | 1.0 | 0.5214 | 0.5332 | Completed — chain step 1 |
| **03** | [`chain02_bce_pos_w1`](#chain02_bce_pos_w1) | 2026-09-03 | T1 / FA | Normal | `[16, 32, 64, 128]` | 2 / 4 | 2 / 2 | 150 | 1e-4 | 0 | 1 | 1.0 | 1.0 | 0.5270 | 0.5422 | Completed — chain step 2 |
| **04** | [`chain05_large`](#chain05_large) | 2026-09-04 | T1 / FA | Large | `[32, 64, 128, 256]` | 2 / 4 | 2 / 2 | 150 | 1e-4 | 10 | 1 | 1.0 | 0.5 | 0.7957 | 0.8170 | Completed |
| **05** | [`chain04_bce_weight05`](#chain04_bce_weight05) | 2026-09-04 | T1 / FA | Normal | `[16, 32, 64, 128]` | 2 / 4 | 2 / 2 | 150 | 1e-4 | 10 | 1 | 1.0 | 0.5 | 0.6108 | 0.6309 | Completed |

---

## 2. Baseline → Champion: Incremental Build-Up Chain

Each row adds **exactly one** parameter change on top of the previous step.
`effective_batch_size` is held fixed at 2 across the chain (not ablated).
`train_patches_per_subject` is likewise fixed at 2.

| Step | Parameter added | Run | Test Mean Dice (Ch) | Δ vs. previous |
|---|---|---|---|---|
| 0 | *(baseline)* | [`baseline`](#baseline) | 0.5271 | — |
| 1 | **LR** 5e-5 → 1e-4 | [`chain01_lr1e-4`](#chain01_lr1e-4) | 0.5332 | **+0.0061** |
| 2 | **bce_pos_weight** 20 → 1 | [`chain02_bce_pos_w1`](#chain02_bce_pos_w1) | 0.5422 | **+0.0090** |
| 3 | **warmup** 0 → 10 ep | `chain03_warmup10` | — | — |
| 4 | **bce_weight** 1.0 → 0.5 | `chain04_bce_weight05` | — | — |
| 5 | **large model** `[32,64,128,256]` | `chain05_large` | — | — *(champion)* |

---

## 3. Run Details

---

<a id="baseline"></a>
### Run `baseline`
*   **Date:** 2026-09-03
*   **Modalities:** T1 / FA
*   **Run directory:** `/home/imag2/Documents/IMAG2/VoxWhisper/runs/processed_T1_FA/baseline/20260903_164755`

#### Hyperparameters

| Parameter | Value |
|---|---|
| Model size | Normal — `[16, 32, 64, 128]` (embed_dim 128) |
| Encoder channels | `[16, 32, 64, 128]` |
| Patch size | `[128, 128, 128]` |
| Patches / subj (Tr/Val) | 2 / 4 |
| Batch (phys / eff) | 2 / 2 |
| Learning rate | 5e-5 |
| Warmup epochs | 0 |
| BCE pos weight | 20 |
| Dice weight / BCE weight | 1.0 / 1.0 |
| Epochs | 150 |
| Normalisation | zscore |

*   **Val Patch Dice (best epoch):** 0.5149

#### Test Dice per tract (channel Dice @ threshold 0.5)

| Tract | Mean | Notes |
|---|---|---|
| ATR_left | 0.5303 | |
| ATR_right | 0.5560 | |
| CG_left | 0.5816 | |
| CG_right | 0.5092 | |
| UF_left | 0.4618 | |
| UF_right | 0.5240 | |
| **Foreground mean** | **0.5271** | |

---

<a id="chain01_lr1e-4"></a>
### Run `chain01_lr1e-4`
*   **Date:** 2026-09-03
*   **Modalities:** T1 / FA
*   **Run directory:** `/home/imag2/Documents/IMAG2/VoxWhisper/runs/processed_T1_FA/lr1e-4/20260903_183845`
*   **Note:** On-disk family dir is still `lr1e-4` (pre-rename).

#### Hyperparameters

| Parameter | Value |
|---|---|
| Model size | Normal — `[16, 32, 64, 128]` (embed_dim 128) |
| Encoder channels | `[16, 32, 64, 128]` |
| Patch size | `[128, 128, 128]` |
| Patches / subj (Tr/Val) | 2 / 4 |
| Batch (phys / eff) | 2 / 2 |
| Learning rate | 1e-4 |
| Warmup epochs | 0 |
| BCE pos weight | 20 |
| Dice weight / BCE weight | 1.0 / 1.0 |
| Epochs | 150 |
| Normalisation | zscore |

*   **Val Patch Dice (best epoch):** 0.5214

#### Test Dice per tract (channel Dice @ threshold 0.5)

| Tract | Mean | Notes |
|---|---|---|
| ATR_left | 0.5338 | |
| ATR_right | 0.5592 | |
| CG_left | 0.5884 | |
| CG_right | 0.5176 | |
| UF_left | 0.4683 | |
| UF_right | 0.5321 | |
| **Foreground mean** | **0.5332** | |

---

<a id="chain02_bce_pos_w1"></a>
### Run `chain02_bce_pos_w1`
*   **Date:** 2026-09-03
*   **Modalities:** T1 / FA
*   **Run directory:** `/home/imag2/Documents/IMAG2/VoxWhisper/runs/processed_T1_FA/chain02_bce_pos_w1/20260903_203138`

#### Hyperparameters

| Parameter | Value |
|---|---|
| Model size | Normal — `[16, 32, 64, 128]` (embed_dim 128) |
| Encoder channels | `[16, 32, 64, 128]` |
| Patch size | `[128, 128, 128]` |
| Patches / subj (Tr/Val) | 2 / 4 |
| Batch (phys / eff) | 2 / 2 |
| Learning rate | 1e-4 |
| Warmup epochs | 0 |
| BCE pos weight | 1 |
| Dice weight / BCE weight | 1.0 / 1.0 |
| Epochs | 150 |
| Normalisation | zscore |

*   **Val Patch Dice (best epoch):** 0.5270

#### Test Dice per tract (channel Dice @ threshold 0.5)

| Tract | Mean | Notes |
|---|---|---|
| ATR_left | 0.5402 | |
| ATR_right | 0.5719 | |
| CG_left | 0.5957 | |
| CG_right | 0.5344 | |
| UF_left | 0.4764 | |
| UF_right | 0.5344 | |
| **Foreground mean** | **0.5422** | |

<a id="chain05_large"></a>
### Run `chain05_large`
*   **Date:** 2026-09-04
*   **Modalities:** T1 / FA
*   **Run directory:** `/home/imag2/Documents/IMAG2/VoxWhisper/runs/processed_T1_FA/chain05_large/20260904_073513`

#### Hyperparameters

| Parameter | Value |
|---|---|
| Model size | Large — `[32, 64, 128, 256]` (embed_dim 256) |
| Encoder channels | `[32, 64, 128, 256]` |
| Patch size | `[128, 128, 128]` |
| Patches / subj (Tr/Val) | 2 / 4 |
| Batch (phys / eff) | 2 / 2 |
| Learning rate | 1e-4 |
| Warmup epochs | 10 |
| BCE pos weight | 1 |
| Dice weight / BCE weight | 1.0 / 0.5 |
| Epochs | 150 |
| Normalisation | zscore |

*   **Val Patch Dice (best epoch):** 0.7957

#### Test Dice per tract (channel Dice @ threshold 0.5)

| Tract | Mean | Notes |
|---|---|---|
| ATR_left | 0.8273 | |
| ATR_right | 0.8403 | |
| CG_left | 0.8581 | |
| CG_right | 0.8525 | |
| UF_left | 0.7483 | |
| UF_right | 0.7756 | |
| **Foreground mean** | **0.8170** | |

<a id="chain04_bce_weight05"></a>
### Run `chain04_bce_weight05`
*   **Date:** 2026-09-04
*   **Modalities:** T1 / FA
*   **Run directory:** `/home/imag2/Documents/IMAG2/VoxWhisper/runs/processed_T1_FA/chain04_bce_weight05/20260904_092338`

#### Hyperparameters

| Parameter | Value |
|---|---|
| Model size | Normal — `[16, 32, 64, 128]` (embed_dim 128) |
| Encoder channels | `[16, 32, 64, 128]` |
| Patch size | `[128, 128, 128]` |
| Patches / subj (Tr/Val) | 2 / 4 |
| Batch (phys / eff) | 2 / 2 |
| Learning rate | 1e-4 |
| Warmup epochs | 10 |
| BCE pos weight | 1 |
| Dice weight / BCE weight | 1.0 / 0.5 |
| Epochs | 150 |
| Normalisation | zscore |

*   **Val Patch Dice (best epoch):** 0.6108

#### Test Dice per tract (channel Dice @ threshold 0.5)

| Tract | Mean | Notes |
|---|---|---|
| ATR_left | 0.5288 | |
| ATR_right | 0.5842 | |
| CG_left | 0.5843 | |
| CG_right | 0.8429 | |
| UF_left | 0.4809 | |
| UF_right | 0.7645 | |
| **Foreground mean** | **0.6309** | |
