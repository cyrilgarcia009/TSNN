# Experiment 2 — TSCS Variant Ablation

Ablates the two degrees of freedom in the `TSCS_shift` effect: whether the temporal lag is fixed or random, and whether the cross-sectional permutation is fixed (circular) or random (derangement). Results inform the paper's analysis of how structural complexity within a single effect type affects model performance.

## Default settings

### Data

| Parameter | Value | Notes |
|-----------|-------|-------|
| T (total time steps) | 4,000 | |
| N (series) | 10 | |
| F (features per series) | 5 | All 5 active (equal weight) |
| Look-back window (T_win) | 10 | |
| Train / Val / Test | 50% / 12.5% / 37.5% | Val carved from training split (val_pct=0.2) |

### Effects (4)

The four variants cross two binary axes:

| Axis | Fixed (F) | Random (R) |
|------|-----------|------------|
| TS lag | Hardcoded lag = 1 | Drawn uniformly from [1, 9] per feature and seed |
| CS shift | Circular shift (last series wraps to first) | Random derangement, different each seed |

| Effect | TS lag | CS shift | Description |
|--------|--------|----------|-------------|
| `TSCS_FF` | fixed = 1 | circular | Simplest variant: fully deterministic structure |
| `TSCS_FR` | fixed = 1 | random derangement | Random CS only |
| `TSCS_RF` | random ∈ [1,9] | circular | Random TS only |
| `TSCS_RR` | random ∈ [1,9] | random derangement | Both axes random (matches Experiment 1's `TSCS_shift`) |

> **Note:** `TSCS_RR` is identical to the `TSCS_shift` effect in Experiment 1 and serves as a consistency check.

### Models (15)

Same model set as Experiment 1.

| Model | Type | Key hyperparameters |
|-------|------|---------------------|
| TheoC | Analytical | Returns NaN for TSCS effects (OLS bound only valid for `lin`) |
| Lasso | Classical | Global LassoCV (cv=5) on flattened rolling windows |
| MLP | Classical | Global sklearn MLPRegressor (128→64) on flattened windows |
| GlobalLSTM | Deep | 2 layers, hidden=128, dropout=0.1 |
| TC, CT, TT, CC | Deep | `CustomBiDimensionalTransformer`, 2 blocks |
| TCTC, CTCT | Deep | `CustomBiDimensionalTransformer`, 4 blocks |
| TCTCTC (TC³) | Deep | `CustomBiDimensionalTransformer`, 6 blocks |
| TCTCTCTC (TC⁴) | Deep | `CustomBiDimensionalTransformer`, 8 blocks |
| TFT | Deep | hidden=64, n_head=4, dropout=0.1, max_steps=300 |
| TiDE | Deep | hidden=256, dropout=0.1, max_steps=300 |
| NBEATSx | Deep | 3× identity stacks, MLP 256→256, max_steps=300 |

### Training protocol

| Setting | Value |
|---------|-------|
| Optimizer (PyTorch models) | AdamW, lr=1e-3, β=(0.9, 0.995), wd=1e-4 |
| Val early stopping metric | Correlation to actual target |
| Val warmup | 5 epochs before early stopping activates |
| Early stopping patience | 30 epochs |
| Epoch budget by ρ | ρ=0.02→200, 0.05→150, 0.10→100, 0.20→70, 0.50→40 |
| Seeds | 10 (0–9) → mean ± std reported |
| ρ values | 0.02, 0.05, 0.10, 0.20, 0.50 |

---

## What this experiment produces

| Output | Path | Used for |
|--------|------|----------|
| Raw per-seed rows | `results_TSCS/raw_results.csv` | Bootstrap CIs, reproducibility |
| Aggregated summary | `results_TSCS/summary.csv` | Quick inspection |
| Line plots (ρ vs corr) | `results_TSCS/figures/perf_vs_rho_*.pdf` | Paper figures |
| Heatmap tables | `results_TSCS/figures/heatmap_*.pdf` | Supplementary / slides |
| LaTeX table fragments | `results_TSCS/latex/*.tex` | Paste into paper |

> **Note:** `results_TSCS/` is git-ignored. Regenerate from raw CSV using `analyze.py`.

## Quick start

```bash
# From the project root, activate your environment first:
conda activate tsnn   # or: source .venv/bin/activate

# 1. Smoke test — 1 seed, 2 variants, 2 rho values
/home/ubuntu/miniconda3/envs/tsnn/bin/python experiments/experiment2_TSCS_variants/run.py \
    --seeds 0 --effects TSCS_FF TSCS_RR --rhos 0.1 0.2 --models TCTC Lasso

# 2. Full run (reads config.yaml, resumes if interrupted)
/home/ubuntu/miniconda3/envs/tsnn/bin/python experiments/experiment2_TSCS_variants/run.py

# 3. Analyze results and generate figures + LaTeX tables
/home/ubuntu/miniconda3/envs/tsnn/bin/python experiments/experiment2_TSCS_variants/analyze.py
```

## Launching in tmux (recommended for long runs)

```bash
mkdir -p experiments/experiment2_TSCS_variants/results_TSCS
tmux new-session -d -s exp2 \
    "/home/ubuntu/miniconda3/envs/tsnn/bin/python experiments/experiment2_TSCS_variants/run.py \
     2>&1 | tee experiments/experiment2_TSCS_variants/results_TSCS/run.log"
```

Monitor progress:

```bash
tmux attach -t exp2                                                      # full view; Ctrl+B D to detach
tail -f experiments/experiment2_TSCS_variants/results_TSCS/run.log      # log only
```

## Resuming an interrupted run

`run.py` writes results row-by-row and checks `raw_results.csv` on startup. Any `(effect, rho, seed, model)` combination already in the file is skipped. Just re-run the same command to resume.

## Configuring the experiment

All parameters live in `config.yaml`. The effect variants and their TS/CS configurations are hardcoded in `run.py`'s `TSCS_VARIANTS` dict and cannot be changed from the config alone.

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `rho_values` | 5 values | SNR grid (0.02 → 0.50) |
| `seeds` | 0–9 | 10 seeds → mean ± std in tables |
| `models` | 15 models | Comment out any model to exclude |
| `training.epochs_by_rho` | rho-dependent | More epochs for harder low-SNR problems |

## GPU / CPU

Device is auto-detected (`cuda → mps → cpu`). Estimated runtime on an AWS g4dn.xlarge (1× T4):

| Scope | Approximate time |
|-------|-----------------|
| Smoke test (2 variants, 2 rho, 1 seed, TCTC+Lasso) | ~5 min |
| Full grid, PyTorch models only, 10 seeds | ~1–2 hr |
| Full grid, all models including NeuralForecast, 10 seeds | ~10–14 hr |

> **Note:** NeuralForecast models (TFT, TiDE, NBEATSx) print `LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]` repeatedly during training. This is harmless PyTorch Lightning output.

## Analyzing results

```bash
# Default metric: test_corr_optimal
/home/ubuntu/miniconda3/envs/tsnn/bin/python experiments/experiment2_TSCS_variants/analyze.py

# Alternative metric
/home/ubuntu/miniconda3/envs/tsnn/bin/python experiments/experiment2_TSCS_variants/analyze.py --metric test_corr_actual

# Skip figures, only regenerate LaTeX
/home/ubuntu/miniconda3/envs/tsnn/bin/python experiments/experiment2_TSCS_variants/analyze.py --no-figures
```

## File layout

```
experiment2_TSCS_variants/
├── config.yaml      ← all tunable parameters
├── run.py           ← experiment runner
├── analyze.py       ← figures + tables generator
├── README.md        ← this file
└── results_TSCS/    ← git-ignored, populated by run.py
    ├── raw_results.csv
    ├── run.log
    ├── summary.csv
    ├── figures/
    │   ├── perf_vs_rho_test_corr_optimal.pdf
    │   ├── heatmap_rho0.20_test_corr_optimal.pdf
    │   └── ...
    └── latex/
        ├── table_rho0.20_test_corr_optimal.tex
        └── ...
```
