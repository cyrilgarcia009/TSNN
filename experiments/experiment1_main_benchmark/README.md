# Experiment 1 — Main Effect Benchmark

Systematically compares all models across all effect types and SNR levels.
Results feed directly into the paper's main result tables and figures.

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
