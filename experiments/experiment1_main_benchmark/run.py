"""
Experiment 1 — Main Effect Benchmark
=====================================
Runs a grid of (effect × rho × seed) combinations, fitting all configured
models on each synthetic dataset and saving per-row results to CSV.

Usage
-----
From the project root:

    # Full run (reads config.yaml next to this file)
    python experiments/experiment1_main_benchmark/run.py

    # Quick smoke test — 1 seed, subset of effects
    python experiments/experiment1_main_benchmark/run.py \
        --seeds 0 --effects TS_shift CS_shift --rhos 0.1 0.2

    # Resume interrupted run (skips already-saved rows automatically)
    python experiments/experiment1_main_benchmark/run.py

    # Override config path
    python experiments/experiment1_main_benchmark/run.py \
        --config path/to/other_config.yaml
"""

import argparse
import contextlib
import io
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path so `tsnn` is importable when run from anywhere
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tsnn import utils
from tsnn.benchmarks import ml_benchmarks, torch_benchmarks
from tsnn.generators import generators
from tsnn.tstorch import models

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Helpers: data generation
# ---------------------------------------------------------------------------

SUPERPOSITION_EFFECTS = ["lin", "TS_shift", "CS_shift", "fea_cond", "TSCS_shift"]


def _default_corr_split(n_f):
    if n_f >= 10:
        half = n_f // 2
        return [1 / np.sqrt(half)] * half + [0] * half
    return [1 / np.sqrt(n_f)] * n_f


def generate_dataset(effect, T, n_ts, n_f, rho, seed, shuffle_cs=True, max_ts_lag=10):
    """Generate (X, y, y_opt, generator) for a given effect and SNR.

    max_ts_lag: TS/TSCS lags are drawn uniformly from [1, max_ts_lag] inclusive.
                Set to 1 for a fixed lag of 1 (old behaviour).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if effect == "superposition":
        # Cycle through SUPERPOSITION_EFFECTS to assign one effect per feature.
        # This gives equal weight to each effect type.
        effect_list = [SUPERPOSITION_EFFECTS[i % len(SUPERPOSITION_EFFECTS)] for i in range(n_f)]
    else:
        effect_list = [effect] * n_f

    # Generator uses randint(1, random_ts_shift) — exclusive upper bound —
    # so pass max_ts_lag + 1 to get lags uniformly in [1, max_ts_lag].
    random_ts_shift = max_ts_lag + 1 if max_ts_lag > 1 else 1

    z = generators.Generator(T, n_ts, n_f)
    z.generate_dataset_gr_simple(
        global_corr=rho,
        correl_split_by_fea=_default_corr_split(n_f),
        list_type_effects=effect_list,
        list_type_interaction=["cond"],
        random_ts_shift=random_ts_shift,
        shuffle_cs=shuffle_cs,
    )
    return z.X, z.y, z.ys["optimal"], z


# ---------------------------------------------------------------------------
# Helpers: data loaders
# ---------------------------------------------------------------------------

def _collate(batch):
    return utils.collate_pad_beginning(batch)


def build_loaders(X, y, n_rolling, batch_size, train_pct, roll_y=False):
    dataset = utils.TorchDatasetRolling(X, y, n=n_rolling, roll_y=roll_y)
    train_size = int(train_pct * len(dataset))
    train_subset = torch.utils.data.Subset(dataset, range(train_size))
    test_subset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Helpers: metrics
# ---------------------------------------------------------------------------

def corr(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def mse(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean((a[mask] - b[mask]) ** 2)) if mask.sum() > 0 else np.nan


def _eval_torch_model(wrapper, loader, device):
    preds, targets = [], []
    wrapper.model.eval()
    eval_loader = DataLoader(
        loader.dataset, batch_size=loader.batch_size, shuffle=False,
        collate_fn=loader.collate_fn,
    )
    with torch.no_grad():
        for batch in eval_loader:
            if len(batch) == 3:
                Xb, yb, pm = batch
                pred = wrapper.model(Xb.to(device), pad_mask=pm.to(device)).detach().cpu()
            else:
                Xb, yb = batch
                pred = wrapper.model(Xb.to(device)).detach().cpu()
            if pred.ndim == 3:
                pred = pred[:, -1, :]
            if yb.ndim == 3:
                yb = yb[:, -1, :]
            preds.append(pred.numpy())
            targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _eval_torch_vs_optimal(wrapper, loader, y_opt, device):
    preds = []
    wrapper.model.eval()
    eval_loader = DataLoader(
        loader.dataset, batch_size=loader.batch_size, shuffle=False,
        collate_fn=loader.collate_fn,
    )
    with torch.no_grad():
        for batch in eval_loader:
            Xb = batch[0]
            pm = batch[2] if len(batch) == 3 else None
            pred = wrapper.model(
                Xb.to(device),
                **({"pad_mask": pm.to(device)} if pm is not None else {}),
            ).detach().cpu()
            if pred.ndim == 3:
                pred = pred[:, -1, :]
            preds.append(pred.flatten())
    preds = torch.cat(preds).numpy()
    idx = loader.dataset.indices
    opt = y_opt[idx].reshape(-1).numpy()
    return corr(preds, opt)


def _torch_metrics(wrapper, train_loader, test_loader, y_opt, device):
    tr_pred, tr_tgt = _eval_torch_model(wrapper, train_loader, device)
    te_pred, te_tgt = _eval_torch_model(wrapper, test_loader, device)
    return {
        "train_corr_optimal": _eval_torch_vs_optimal(wrapper, train_loader, y_opt, device),
        "test_corr_optimal":  _eval_torch_vs_optimal(wrapper, test_loader, y_opt, device),
        "train_corr_actual":  corr(tr_pred, tr_tgt),
        "test_corr_actual":   corr(te_pred, te_tgt),
        "train_mse_actual":   mse(tr_pred, tr_tgt),
        "test_mse_actual":    mse(te_pred, te_tgt),
    }


# ---------------------------------------------------------------------------
# Helpers: flattened rolling windows for sklearn baselines
# ---------------------------------------------------------------------------

def _build_flat_windows(X, y, n_rolling, train_pct):
    """Return (X_train, y_train, X_test, y_test, X_all, y_all) as numpy arrays.

    X_flat shape: (T, n_rolling * n_ts * n_f)
    y_flat shape: (T * n_ts,)  — flattened over series for global regression
    """
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    T, n_ts, n_f = X_np.shape

    # Build padded rolling windows
    windows = np.zeros((T, n_rolling, n_ts, n_f), dtype=np.float32)
    for t in range(T):
        start = max(0, t - n_rolling + 1)
        w = X_np[start: t + 1]
        windows[t, -w.shape[0]:] = w

    X_flat = windows.reshape(T, -1)   # (T, n_rolling * n_ts * n_f)

    train_cut = int(T * train_pct)
    return (
        X_flat[:train_cut], y_np[:train_cut],
        X_flat[train_cut:], y_np[train_cut:],
    )


def _sklearn_metrics(pred_train, y_train, pred_test, y_test, y_opt_np, train_cut):
    return {
        "train_corr_optimal": corr(pred_train.reshape(-1), y_opt_np[:train_cut].reshape(-1)),
        "test_corr_optimal":  corr(pred_test.reshape(-1), y_opt_np[train_cut:].reshape(-1)),
        "train_corr_actual":  corr(pred_train.reshape(-1), y_train.reshape(-1)),
        "test_corr_actual":   corr(pred_test.reshape(-1), y_test.reshape(-1)),
        "train_mse_actual":   mse(pred_train.reshape(-1), y_train.reshape(-1)),
        "test_mse_actual":    mse(pred_test.reshape(-1), y_test.reshape(-1)),
    }


# ---------------------------------------------------------------------------
# NeuralForecast helpers
# ---------------------------------------------------------------------------

def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)


def _build_nf_df(X, y):
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    T, n_ts, n_f = X_np.shape
    rows = []
    for s in range(n_ts):
        for t in range(T):
            row = {"unique_id": str(s), "ds": int(t), "y": float(y_np[t, s])}
            for f in range(n_f):
                row[f"x{f}"] = float(X_np[t, s, f])
            rows.append(row)
    return pd.DataFrame(rows)


def _nf_rollout(nf, full_df, futr_exog_list, start_t, end_t):
    from neuralforecast import NeuralForecast  # local import so non-NF runs don't need it
    pred_steps = []
    futr_cols = ["unique_id", "ds"] + futr_exog_list
    for t in range(start_t, end_t):
        hist_df = full_df[full_df["ds"] < t].copy()
        futr_df = full_df.loc[full_df["ds"] == t, futr_cols].copy()
        pred_t = _quiet(nf.predict, df=hist_df, futr_df=futr_df)
        model_col = [c for c in pred_t.columns if c not in ("unique_id", "ds")][0]
        pred_t = pred_t.sort_values(["ds", "unique_id"])
        pred_steps.append(pred_t[model_col].to_numpy())
    return np.stack(pred_steps, axis=0)   # (steps, n_ts)


def run_nf_model(model_builder, X, y, y_opt, train_pct, val_pct, n_rolling, test_eval_steps):
    try:
        from neuralforecast import NeuralForecast
    except ImportError:
        raise ImportError("neuralforecast is not installed. Run: pip install neuralforecast")

    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    y_opt_np = y_opt.detach().cpu().numpy() if torch.is_tensor(y_opt) else np.asarray(y_opt)
    T, n_ts, n_f = X_np.shape
    train_cut = int(T * train_pct)
    futr_exog_list = [f"x{f}" for f in range(n_f)]

    full_df = _build_nf_df(X, y)
    train_df = full_df[full_df["ds"] < train_cut].copy()
    val_size = max(1, int(train_cut * val_pct)) if val_pct and val_pct > 0 else 0

    model = model_builder(n_series=n_ts, futr_exog_list=futr_exog_list)
    nf = NeuralForecast(models=[model], freq=1)
    _quiet(nf.fit, train_df, val_size=val_size)

    warmup = max(n_f, 10)
    train_start = min(train_cut - 1, warmup)
    train_end = min(train_cut, train_start + 100)
    test_start = train_cut
    test_end = T if test_eval_steps is None else min(T, test_start + test_eval_steps)

    tr_pred = _nf_rollout(nf, full_df, futr_exog_list, train_start, train_end)
    te_pred = _nf_rollout(nf, full_df, futr_exog_list, test_start, test_end)

    return {
        "train_corr_optimal": corr(tr_pred, y_opt_np[train_start:train_end]),
        "test_corr_optimal":  corr(te_pred, y_opt_np[test_start:test_end]),
        "train_corr_actual":  corr(tr_pred, y_np[train_start:train_end]),
        "test_corr_actual":   corr(te_pred, y_np[test_start:test_end]),
        "train_mse_actual":   mse(tr_pred, y_np[train_start:train_end]),
        "test_mse_actual":    mse(te_pred, y_np[test_start:test_end]),
    }


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_theoc(effect, rho, generator_obj):
    """Analytical OLS bound: equals rho for the linear effect; NaN otherwise."""
    val = float(rho) if effect == "lin" else np.nan
    return {k: val if "optimal" in k else np.nan for k in [
        "train_corr_optimal", "test_corr_optimal",
        "train_corr_actual", "test_corr_actual",
        "train_mse_actual", "test_mse_actual",
    ]}


def run_lasso(X, y, y_opt, n_rolling, train_pct):
    from sklearn.linear_model import LassoCV
    from sklearn.multioutput import MultiOutputRegressor

    X_tr, y_tr, X_te, y_te = _build_flat_windows(X, y, n_rolling, train_pct)
    T = X_tr.shape[0] + X_te.shape[0]
    train_cut = X_tr.shape[0]
    y_opt_np = y_opt.detach().cpu().numpy() if torch.is_tensor(y_opt) else np.asarray(y_opt)

    model = MultiOutputRegressor(LassoCV(cv=5, max_iter=2000, n_jobs=1), n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    return _sklearn_metrics(pred_tr, y_tr, pred_te, y_te, y_opt_np, train_cut)


def run_mlp(X, y, y_opt, n_rolling, train_pct):
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor

    X_tr, y_tr, X_te, y_te = _build_flat_windows(X, y, n_rolling, train_pct)
    train_cut = X_tr.shape[0]
    y_opt_np = y_opt.detach().cpu().numpy() if torch.is_tensor(y_opt) else np.asarray(y_opt)

    model = MultiOutputRegressor(
        MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=0),
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    return _sklearn_metrics(pred_tr, y_tr, pred_te, y_te, y_opt_np, train_cut)


def run_torch_model(model_builder, X, y, y_opt, n_rolling, batch_size,
                    train_pct, epochs, val_pct, val_warmup_epochs,
                    early_stopping_patience, roll_y=False):
    train_loader, test_loader = build_loaders(X, y, n_rolling, batch_size, train_pct, roll_y=roll_y)
    model = model_builder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.995), weight_decay=1e-4)
    wrapper = torch_benchmarks.TorchWrapper(model, optimizer=optimizer, loss_fn=nn.MSELoss(), device=DEVICE)
    wrapper.fit(
        train_loader, test_loader,
        epochs=epochs, plot=False, verbose=0,
        val_pct=val_pct, early_stopping_metric="corr",
        val_warmup_epochs=val_warmup_epochs,
        early_stopping_patience=early_stopping_patience,
    )
    metrics = _torch_metrics(wrapper, train_loader, test_loader, y_opt, DEVICE)
    # best_epoch is 0-indexed; store as 1-indexed for readability.
    best_epoch = wrapper.best_epoch
    metrics["best_epoch"] = (best_epoch + 1) if best_epoch is not None else None
    metrics["max_epochs"] = epochs
    metrics["epoch_pct_used"] = ((best_epoch + 1) / epochs) if best_epoch is not None else None
    return metrics


def _causal_mask(seq_len, device):
    q = torch.arange(seq_len, device=device)
    k = torch.arange(seq_len, device=device)
    return q[:, None] >= k[None, :]


# All CustomBiDimensionalTransformer variants — model name IS the layers string.
TC_VARIANTS = {"TC", "CT", "TT", "CC", "TCTC", "CTCT", "TCTCTC", "TCTCTCTC"}


def build_tc_variant(layers_str, n_ts, n_f, n_rolling):
    mask = _causal_mask(n_rolling, DEVICE)
    return models.CustomBiDimensionalTransformer(
        n_ts, n_f, n_rolling, mask=mask, layers=layers_str,
        nhead=1, dropout=0.0,
        d_model=n_ts * n_f, dim_feedforward=n_ts * n_f * 2,
        sparsify=None, roll_y=True, embeddings="both",
    ).to(DEVICE)


def build_lstm(n_ts, n_f, n_rolling):
    return models.GlobalLSTM(
        n_ts=n_ts, n_f=n_f, n_rolling=n_rolling,
        hidden_dim=128, num_layers=2, dropout=0.1,
    ).to(DEVICE)


# ---------------------------------------------------------------------------
# NeuralForecast model builders
# ---------------------------------------------------------------------------

def _nf_trainer_kwargs():
    return dict(
        accelerator="auto", devices=1,
        enable_progress_bar=False, enable_model_summary=False,
        logger=False, log_every_n_steps=10 ** 9,
    )


def make_nf_builders(cfg_nf, seed):
    try:
        from neuralforecast.models import TFT, TiDE, NBEATSx
    except ImportError:
        return {}

    ms = cfg_nf["max_steps"]
    vc = cfg_nf["val_check_steps"]
    lr = cfg_nf["learning_rate"]
    tr = _nf_trainer_kwargs()

    return {
        "TFT": lambda n_series, futr_exog_list: TFT(
            h=1, input_size=10, futr_exog_list=futr_exog_list,
            hidden_size=64, n_head=4, dropout=0.1,
            max_steps=ms, val_check_steps=vc, learning_rate=lr,
            scaler_type="standard", random_seed=seed, **tr,
        ),
        "TiDE": lambda n_series, futr_exog_list: TiDE(
            h=1, input_size=10, futr_exog_list=futr_exog_list,
            exclude_insample_y=False, hidden_size=256, dropout=0.1,
            max_steps=ms, val_check_steps=vc, learning_rate=lr,
            scaler_type="standard", random_seed=seed, **tr,
        ),
        "NBEATSx": lambda n_series, futr_exog_list: NBEATSx(
            h=1, input_size=10, futr_exog_list=futr_exog_list,
            exclude_insample_y=False,
            stack_types=["identity", "identity", "identity"],
            mlp_units=[[256, 256], [256, 256], [256, 256]],
            max_steps=ms, val_check_steps=vc, learning_rate=lr,
            scaler_type="standard", random_seed=seed, **tr,
        ),
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_keys(path):
    """Return set of (effect, rho, seed, model) already saved."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        return set(zip(df["effect"], df["rho"].astype(float).round(6), df["seed"].astype(int), df["model"]))
    except Exception:
        return set()


def append_row(path, row):
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run Experiment 1 — Main Effect Benchmark")
    p.add_argument("--config", default=str(_HERE / "config.yaml"))
    p.add_argument("--effects", nargs="+", help="Override effects from config")
    p.add_argument("--rhos", nargs="+", type=float, help="Override rho_values from config")
    p.add_argument("--seeds", nargs="+", type=int, help="Override seeds from config")
    p.add_argument("--models", nargs="+", help="Override models from config")
    p.add_argument("--dry-run", action="store_true", help="Print the run plan without executing")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    effects = args.effects or cfg["effects"]
    rhos = args.rhos or cfg["rho_values"]
    seeds = args.seeds or cfg["seeds"]
    model_list = args.models or cfg["models"]

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    nf_cfg = cfg["neuralforecast"]
    out_cfg = cfg["output"]

    results_dir = _HERE / out_cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / out_cfg["raw_file"]

    T = data_cfg["T"]
    n_ts = data_cfg["n_ts"]
    n_f = data_cfg["n_f"]
    n_rolling = data_cfg["n_rolling"]
    train_pct = data_cfg["train_pct"]
    batch_size = data_cfg["batch_size"]
    shuffle_cs = data_cfg["shuffle_cs"]
    max_ts_lag = data_cfg.get("max_ts_lag", n_rolling)

    total = len(effects) * len(rhos) * len(seeds)
    print(f"Device: {DEVICE}")
    print(f"Run plan: {len(effects)} effects × {len(rhos)} rhos × {len(seeds)} seeds = {total} datasets")
    print(f"Models per dataset: {model_list}")
    print(f"Results → {raw_path}")

    if args.dry_run:
        print("\nDry run — exiting.")
        return

    completed = load_completed_keys(raw_path)
    print(f"Already completed: {len(completed)} (effect, rho, seed, model) combinations")

    outer_bar = tqdm(total=total, desc="datasets", unit="dataset")

    for seed in seeds:
        nf_builders = make_nf_builders(nf_cfg, seed)

        for effect in effects:
            for rho in rhos:
                outer_bar.set_postfix(effect=effect, rho=rho, seed=seed)

                # Check if all models for this (effect, rho, seed) are done
                pending = [m for m in model_list
                           if (effect, round(rho, 6), seed, m) not in completed]
                if not pending:
                    outer_bar.update(1)
                    continue

                # Generate dataset once per (effect, rho, seed)
                try:
                    X, y, y_opt, gen_obj = generate_dataset(
                        effect, T, n_ts, n_f, rho, seed,
                        shuffle_cs=shuffle_cs, max_ts_lag=max_ts_lag,
                    )
                except Exception as e:
                    tqdm.write(f"  [ERROR] generate_dataset({effect}, rho={rho}, seed={seed}): {e}")
                    outer_bar.update(1)
                    continue

                epochs = train_cfg["epochs_by_rho"].get(rho, train_cfg["epochs_by_rho"].get(str(rho), 100))

                for model_name in pending:
                    key = (effect, round(rho, 6), seed, model_name)
                    if key in completed:
                        continue

                    try:
                        if model_name == "TheoC":
                            metrics = run_theoc(effect, rho, gen_obj)

                        elif model_name == "Lasso":
                            metrics = run_lasso(X, y, y_opt, n_rolling, train_pct)

                        elif model_name == "MLP":
                            metrics = run_mlp(X, y, y_opt, n_rolling, train_pct)

                        elif model_name == "GlobalLSTM":
                            metrics = run_torch_model(
                                lambda: build_lstm(n_ts, n_f, n_rolling),
                                X, y, y_opt, n_rolling, batch_size, train_pct,
                                epochs=epochs,
                                val_pct=train_cfg["val_pct"],
                                val_warmup_epochs=train_cfg["val_warmup_epochs"],
                                early_stopping_patience=train_cfg["early_stopping_patience"],
                                roll_y=False,
                            )

                        elif model_name in TC_VARIANTS:
                            _layers = model_name  # capture for lambda closure
                            metrics = run_torch_model(
                                lambda layers=_layers: build_tc_variant(layers, n_ts, n_f, n_rolling),
                                X, y, y_opt, n_rolling, batch_size, train_pct,
                                epochs=epochs,
                                val_pct=train_cfg["val_pct"],
                                val_warmup_epochs=train_cfg["val_warmup_epochs"],
                                early_stopping_patience=train_cfg["early_stopping_patience"],
                                roll_y=True,
                            )

                        elif model_name in nf_builders:
                            metrics = run_nf_model(
                                nf_builders[model_name],
                                X, y, y_opt,
                                train_pct=train_pct,
                                val_pct=nf_cfg["val_pct"],
                                n_rolling=n_rolling,
                                test_eval_steps=nf_cfg["test_eval_steps"],
                            )

                        else:
                            tqdm.write(f"  [SKIP] Unknown model: {model_name}")
                            continue

                    except Exception:
                        tqdm.write(f"  [ERROR] {model_name} | {effect} | rho={rho} | seed={seed}")
                        tqdm.write(traceback.format_exc())
                        continue

                    row = {
                        "effect": effect,
                        "rho": rho,
                        "seed": seed,
                        "model": model_name,
                        **metrics,
                    }
                    # Ensure epoch columns are always present so every row
                    # has identical schema (non-torch models get None).
                    for _col in ("best_epoch", "max_epochs", "epoch_pct_used"):
                        row.setdefault(_col, None)
                    append_row(raw_path, row)
                    completed.add(key)

                outer_bar.update(1)

    outer_bar.close()
    print(f"\nDone. Raw results saved to {raw_path}")
    print("Run analyze.py to generate tables and figures.")


if __name__ == "__main__":
    main()
