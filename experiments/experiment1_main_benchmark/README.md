# Experiment 1 — Main Effect Benchmark

Systematically compares all models across all effect types and SNR levels.
Results feed directly into the paper's main result tables and figures.

## Default settings

### Data

| Parameter | Value | Notes |
|-----------|-------|-------|
| T (total time steps) | 4,000 | |
| N (series) | 10 | |
| F (features per series) | 5 | All 5 active (equal weight) |
| Look-back window (T_win) | 10 | |
| Train / Val / Test | 50% / 12.5% / 37.5% | Val carved from training split (val_pct=0.2) |
| TS lag | Random ∈ [1, 10] | Drawn independently per feature and seed |
| CS shift | Random derangement | Different permutation each seed (`shuffle_cs=true`) |

### Effects (8)

| Effect | Order | Description |
|--------|-------|-------------|
| `lin` | 0 | Linear, same-time, same-series |
| `TS_shift` | 1 | Temporal lag, random in [1, T_win] |
| `CS_shift` | 1 | Cross-sectional shift (random derangement) |
| `fea_cond` | 1 | Feature interaction: X_i × sign(X_j) |
| `TSCS_shift` | 2 | Temporal lag + cross-sectional shift combined |
| `TS_cond` | 2 | Temporal lag × nonlinear conditioning |
| `CS_cond` | 2 | Cross-sectional shift × nonlinear conditioning |
| `superposition` | mixed | All 5 effect types simultaneously (one per feature) |

### Models (8)

| Model | Type | Key hyperparameters |
|-------|------|---------------------|
| TheoC | Analytical | OLS bound; valid for `lin` only |
| Lasso | Classical | Global LassoCV (cv=5) on flattened rolling windows |
| MLP | Classical | Global sklearn MLPRegressor (128→64) on flattened windows |
| GlobalLSTM | Deep | 2 layers, hidden=128, dropout=0.1 |
| TCTC | Deep | d_model=50, nhead=1, FF=100, dropout=0.0, layers=TCTC |
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
| Epoch budget by ρ | ρ=0.02→300, 0.05→200, 0.10→150, 0.20→100, 0.50→60 |
| Seeds | 10 (0–9) → mean ± std reported |
| ρ values | 0.02, 0.05, 0.10, 0.20, 0.50 |

> **Note on the current run (seed 0–9, fixed lag=1):** The first full run was launched with `max_ts_lag=1` (fixed lag). From the next run onward the default is `max_ts_lag=10` (random lag ∈ [1, 10]).

---

## What this experiment produces

| Output | Path | Used for |
|--------|------|----------|
| Raw per-seed rows | `results/raw_results.csv` | Bootstrap CIs, reproducibility |
| Aggregated summary | `results/summary.csv` | Quick inspection |
| Line plots (ρ vs corr) | `results/figures/perf_vs_rho_*.pdf` | Paper figures (Section 3) |
| Heatmap tables | `results/figures/heatmap_*.pdf` | Supplementary / slides |
| LaTeX table fragments | `results/latex/*.tex` | Paste into paper |

> **Note:** `results/` is git-ignored. Regenerate from raw CSV using `analyze.py`.

## Quick start

```bash
# From the project root, activate your environment first:
conda activate tsnn   # or: source .venv/bin/activate

# 1. Smoke test — 1 seed, 2 effects, 2 rho values (~5 min on CPU)
python experiments/experiment1_main_benchmark/run.py \
    --seeds 0 --effects TS_shift CS_shift --rhos 0.1 0.2 --models TCTC Lasso

# 2. Full run (reads config.yaml, resumes if interrupted)
python experiments/experiment1_main_benchmark/run.py

# 3. Analyze results and generate figures + LaTeX tables
python experiments/experiment1_main_benchmark/analyze.py
```

## Resuming an interrupted run

`run.py` writes results row-by-row and checks `raw_results.csv` on startup.
Any `(effect, rho, seed, model)` combination already in the file is skipped.
Just re-run the same command to resume.

## Configuring the experiment

All parameters live in `config.yaml`. Key knobs:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `data.n_f` | 5 | Features per series (paper used 20; 5 is cleaner) |
| `effects` | 8 effects | Which dependency structures to test |
| `rho_values` | 5 values | SNR grid (0.02 → 0.50) |
| `seeds` | 0–9 | 10 seeds → mean ± std in tables |
| `models` | 8 models | Comment out any model to exclude |
| `training.epochs_by_rho` | rho-dependent | More epochs for harder low-SNR problems |

To run a subset, either edit `config.yaml` or pass CLI flags:

```bash
# Only TCTC and Lasso, only two rho values
python experiments/experiment1_main_benchmark/run.py \
    --models TCTC Lasso --rhos 0.05 0.20
```

## GPU / CPU

Device is auto-detected (`cuda → mps → cpu`). On AWS with CUDA, the PyTorch
models (TCTC, GlobalLSTM) run on GPU automatically. NeuralForecast models
(TFT, TiDE, NBEATSx) use `accelerator="auto"` and will also use the GPU.

Estimated runtimes on an AWS g4dn.xlarge (1× T4):

| Scope | Approximate time |
|-------|-----------------|
| Smoke test (2 effects, 2 rho, 1 seed, TCTC+Lasso) | ~5 min |
| Full grid, PyTorch models only, 10 seeds | ~2–3 hr |
| Full grid, all models including NeuralForecast, 10 seeds | ~8–12 hr |

## Analyzing results

```bash
# Default metric: test_corr_optimal
python experiments/experiment1_main_benchmark/analyze.py

# Alternative metric
python experiments/experiment1_main_benchmark/analyze.py --metric test_corr_actual

# Skip figures, only regenerate LaTeX
python experiments/experiment1_main_benchmark/analyze.py --no-figures
```

## File layout

```
experiment1_main_benchmark/
├── config.yaml      ← all tunable parameters
├── run.py           ← experiment runner (edit rarely)
├── analyze.py       ← figures + tables generator (edit for paper styling)
├── README.md        ← this file
└── results/         ← git-ignored, populated by run.py
    ├── raw_results.csv
    ├── summary.csv
    ├── figures/
    │   ├── perf_vs_rho_test_corr_optimal.pdf
    │   ├── heatmap_rho0.20_test_corr_optimal.pdf
    │   └── ...
    └── latex/
        ├── table_rho0.20_test_corr_optimal.tex
        └── ...
```
