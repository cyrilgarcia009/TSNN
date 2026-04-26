"""
Experiment 1 — Analysis & Visualization
========================================
Loads raw_results.csv produced by run.py and generates:
  - summary.csv        (mean ± std across seeds, per effect × model × rho)
  - figures/           (line plots and heatmap tables)
  - latex/             (LaTeX table fragments ready to paste into the paper)

Usage
-----
From the project root:

    python experiments/experiment1_main_benchmark/analyze.py

    # Only regenerate figures, skip summary CSV
    python experiments/experiment1_main_benchmark/analyze.py --no-summary

    # Use a different metric
    python experiments/experiment1_main_benchmark/analyze.py --metric test_corr_actual
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_FIGURES = _RESULTS / "figures"
_LATEX = _RESULTS / "latex"

# Canonical model order for display
MODEL_ORDER = ["TheoC", "Lasso", "MLP", "GlobalLSTM", "TCTC", "TFT", "TiDE", "NBEATSx"]

# Canonical effect order for display
EFFECT_ORDER = ["lin", "TS_shift", "CS_shift", "fea_cond", "TSCS_shift", "TS_cond", "CS_cond", "superposition"]

EFFECT_LABELS = {
    "lin":         "Linear",
    "TS_shift":    "TS-Shift",
    "CS_shift":    "CS-Shift",
    "fea_cond":    "Fea-Nonlin",
    "TSCS_shift":  "TSCS-Shift",
    "TS_cond":     "TS-Nonlin",
    "CS_cond":     "CS-Nonlin",
    "superposition": "Superposition",
}

MODEL_LABELS = {
    "TheoC":     "TheoC",
    "Lasso":     "Lasso",
    "MLP":       "MLP",
    "GlobalLSTM": "LSTM",
    "TCTC":      "TCTC",
    "TFT":       "TFT",
    "TiDE":      "TiDE",
    "NBEATSx":   "NBEATSx",
}

MODEL_COLORS = {
    "TheoC":     "#888888",
    "Lasso":     "#e41a1c",
    "MLP":       "#ff7f00",
    "GlobalLSTM": "#984ea3",
    "TCTC":      "#377eb8",
    "TFT":       "#4daf4a",
    "TiDE":      "#a65628",
    "NBEATSx":   "#f781bf",
}

MODEL_MARKERS = {
    "TheoC": "s", "Lasso": "^", "MLP": "D", "GlobalLSTM": "v",
    "TCTC": "o", "TFT": "P", "TiDE": "X", "NBEATSx": "*",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(path=None):
    path = path or (_RESULTS / "raw_results.csv")
    if not path.exists():
        raise FileNotFoundError(f"Raw results not found at {path}. Run run.py first.")
    df = pd.read_csv(path)
    df["rho"] = df["rho"].astype(float).round(6)
    df["seed"] = df["seed"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(df, metric):
    summary = (
        df.groupby(["effect", "rho", "model"])[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["mean_std"] = summary.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}" if not np.isnan(r["mean"]) else "—", axis=1
    )
    return summary


# ---------------------------------------------------------------------------
# Line plots: rho on x-axis, one curve per model, one panel per effect
# ---------------------------------------------------------------------------

def plot_performance_vs_rho(df, metric="test_corr_optimal", effects=None, models=None, out_dir=None):
    out_dir = out_dir or _FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)

    effects = [e for e in EFFECT_ORDER if e in df["effect"].unique()] if effects is None else effects
    models  = [m for m in MODEL_ORDER if m in df["model"].unique()] if models is None else models

    n_effects = len(effects)
    ncols = min(4, n_effects)
    nrows = int(np.ceil(n_effects / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f"Effect Benchmark — {metric}", fontsize=13, y=1.01)

    for ax_idx, effect in enumerate(effects):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        sub = df[df["effect"] == effect]

        for model in models:
            msub = sub[sub["model"] == model]
            if msub.empty:
                continue
            agg = msub.groupby("rho")[metric].agg(["mean", "std"]).reset_index().sort_values("rho")
            label = MODEL_LABELS.get(model, model)
            color = MODEL_COLORS.get(model, None)
            marker = MODEL_MARKERS.get(model, "o")
            ax.plot(agg["rho"], agg["mean"], label=label, color=color, marker=marker, linewidth=1.8, markersize=5)
            ax.fill_between(
                agg["rho"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                alpha=0.12, color=color,
            )

        ax.set_title(EFFECT_LABELS.get(effect, effect), fontsize=11)
        ax.set_xlabel("ρ (SNR)")
        ax.set_ylabel("Correlation to optimal predictor" if "optimal" in metric else metric)
        ax.set_xscale("log")
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for ax_idx in range(n_effects, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    # Shared legend outside the plot area
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(models), 4),
               bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=9)

    plt.tight_layout()
    out_path = out_dir / f"perf_vs_rho_{metric}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    out_path_png = out_dir / f"perf_vs_rho_{metric}.png"
    fig.savefig(out_path_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Heatmap tables: effect × model for a fixed rho, colored by value
# ---------------------------------------------------------------------------

def plot_heatmap_table(df, rho, metric="test_corr_optimal", out_dir=None):
    out_dir = out_dir or _FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df[df["rho"].round(4) == round(rho, 4)]
    effects = [e for e in EFFECT_ORDER if e in sub["effect"].unique()]
    model_cols = [m for m in MODEL_ORDER if m in sub["model"].unique()]

    pivot = (
        sub.groupby(["effect", "model"])[metric]
        .mean()
        .unstack("model")
        .reindex(index=effects, columns=model_cols)
    )
    pivot.index = [EFFECT_LABELS.get(e, e) for e in pivot.index]
    pivot.columns = [MODEL_LABELS.get(m, m) for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(6, len(model_cols) * 1.2), max(3, len(effects) * 0.7)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.3, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text = f"{val:.3f}" if not np.isnan(val) else "—"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.6 or np.isnan(val) else "black")

    ax.set_title(f"{metric}  |  ρ = {rho}", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / f"heatmap_rho{rho:.2f}_{metric}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table: mean (std) across seeds, bold = best per row
# ---------------------------------------------------------------------------

def latex_table(df, rho, metric="test_corr_optimal", caption="", label=""):
    sub = df[df["rho"].round(4) == round(rho, 4)]
    effects = [e for e in EFFECT_ORDER if e in sub["effect"].unique()]
    model_cols = [m for m in MODEL_ORDER if m in sub["model"].unique()]

    pivot_mean = (
        sub.groupby(["effect", "model"])[metric].mean()
        .unstack("model").reindex(index=effects, columns=model_cols)
    )
    pivot_std = (
        sub.groupby(["effect", "model"])[metric].std()
        .unstack("model").reindex(index=effects, columns=model_cols)
    )

    col_names = [MODEL_LABELS.get(m, m) for m in model_cols]
    eff_names = [EFFECT_LABELS.get(e, e) for e in effects]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l" + "r" * len(col_names) + "}")
    lines.append(r"\toprule")
    lines.append("Effect & " + " & ".join(col_names) + r" \\")
    lines.append(r"\midrule")

    for eff, eff_label in zip(effects, eff_names):
        row_means = pivot_mean.loc[eff]
        row_stds  = pivot_std.loc[eff]
        best_val  = row_means.dropna().max() if not row_means.dropna().empty else None
        cells = []
        for m in model_cols:
            mn = row_means.get(m, np.nan)
            sd = row_stds.get(m, np.nan)
            if np.isnan(mn):
                cells.append("—")
            else:
                txt = f"{mn:.3f}"
                if not np.isnan(sd):
                    txt += f" {{\\scriptsize ±{sd:.3f}}}"
                if best_val is not None and abs(mn - best_val) < 1e-9:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
        lines.append(f"{eff_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    cap = caption or f"{metric} (mean ± std across seeds), ρ = {rho}"
    lbl = label or f"tab:exp1_rho{int(rho*100):02d}"
    lines.append(rf"\caption{{{cap}}}")
    lines.append(rf"\label{{{lbl}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(_RESULTS / "raw_results.csv"))
    p.add_argument("--metric", default="test_corr_optimal")
    p.add_argument("--no-summary", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-latex", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_raw(Path(args.results))
    metric = args.metric

    n_seeds = df["seed"].nunique()
    print(f"Loaded {len(df)} rows | {df['effect'].nunique()} effects | "
          f"{df['model'].nunique()} models | {df['rho'].nunique()} rho values | {n_seeds} seeds")

    if not args.no_summary:
        summary = build_summary(df, metric)
        summary_path = _RESULTS / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")

    if not args.no_figures:
        _FIGURES.mkdir(parents=True, exist_ok=True)
        plot_performance_vs_rho(df, metric=metric)
        for rho in sorted(df["rho"].unique()):
            plot_heatmap_table(df, rho=rho, metric=metric)

    if not args.no_latex:
        _LATEX.mkdir(parents=True, exist_ok=True)
        for rho in sorted(df["rho"].unique()):
            tex = latex_table(df, rho=rho, metric=metric)
            out = _LATEX / f"table_rho{rho:.2f}_{metric}.tex"
            out.write_text(tex)
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()
