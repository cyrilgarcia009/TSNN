# Time-series forecasting using transformers

We benchmark transformer models for multi-dimensional time-series forecasting under low signal-to-noise ratios. The prediction target is a panel `Y` of shape `(T, N)` from covariates `X` of shape `(T, N, F)`. We implement two-way (temporal + cross-sectional) attention and a dynamic max-sparsity mechanism, then evaluate against classical and neural baselines on synthetic data with known ground truth.

Our paper: [arXiv:2602.09869](https://arxiv.org/abs/2602.09869).

## Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate tsnn
pip install -e .
```

**GPU (CUDA 12, AWS):** after creating the env, replace the CPU torch with the CUDA wheel:

```bash
conda activate tsnn
conda remove pytorch torchvision cpuonly --force
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**macOS (MPS):** the default PyTorch wheel ships with MPS support — no extra steps needed.

### pip (alternative)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For GPU on AWS, replace the `torch` install with the CUDA wheel as above.

## Reproducing experiments

All results are produced by `notebooks/updated_figures_for_paper.ipynb`.
Set the `device` variable at the top of the notebook, or let it auto-detect (`cuda → mps → cpu`).

Result CSVs are written to `notebooks/results/` (git-ignored; regenerate locally).

## Repository layout

```
tsnn/               # Python package (generators, benchmarks, models)
notebooks/
  updated_figures_for_paper.ipynb  # main experiment notebook
  archive/                         # old/exploratory notebooks
context/            # planning documents and AI reviews (not shipped)
```

## Contributors

- [Cyril Garcia](https://github.com/cyrilgarcia009)
- [Guillaume Remy](https://github.com/GuillaumeRemy92)
