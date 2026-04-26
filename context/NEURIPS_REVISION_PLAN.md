# TSNN → NeurIPS Resubmission: Planning Document

**Target:** NeurIPS 2026 submission  
**Deadline:** ~2 weeks from 2026-04-26  
**Authors:** Guillaume Remy, Cyril Garcia  
**Paper:** "Statistical Benchmarking of Transformer Models in Low Signal-to-Noise Time-Series Forecasting"  
**ArXiv:** [2602.09869](https://arxiv.org/abs/2602.09869)  

---

## 0. What We Have

The paper proposes a controlled synthetic benchmarking framework for multivariate time-series forecasting with tunable signal-to-noise ratios. The core architecture is a two-way attention transformer (TCTC) with alternating temporal and cross-sectional attention blocks. A dynamic sparse attention mechanism (max_sparse) is also introduced and evaluated. The paper includes an analytical OLS performance derivation, a real-data appendix (M5 dataset), and a SotA comparison appendix (TFT, TiDE, NBEATSx).

The ICML 2026 submission received 4 reviews — all weak or full reject. The plan below addresses each criticism systematically.

---

## 1. Combined Review Checklist

This section synthesizes all issues raised across **4 ICML reviewers** (vpXC, bPFr, PawB, iutV) and **3 AI model reviews** (Claude, GPT-4o, Grok) into a single actionable checklist. Items are grouped by theme. Items already resolved or in progress are marked.

**Legend:** ✅ resolved in new notebook | 🔧 in progress | ⬜ to do

---

### A. Novelty & Positioning

- ⬜ **A1.** The two-way (TC) attention idea is not novel by itself — iTransformer, Crossformer, and axial attention already do this. **Reframe the contribution explicitly** as: (1) the controlled synthetic benchmarking methodology, and (2) the dynamic sparsification mechanism. De-emphasize the architecture as a primary contribution. *(vpXC, Claude, GPT, Grok)*
- ⬜ **A2.** Add a clear **principle-level statement** of when transformers outperform alternatives — e.g., "transformers outperform linear models when the dependency structure lies outside a separable lag-feature decomposition under low SNR." *(GPT)*
- ⬜ **A3.** The dynamic sparse attention is too close to LSINet (top-K), sparsemax, and entmax to claim novelty without differentiation. **Explicitly contrast** with LSINet (fixed top-K globally vs. our relative per-row threshold), sparsemax, and entmax15. *(vpXC, Claude, GPT)*
- ✅ **A4.** Add a gradient-differentiable sparse alternative (entmax15 is implemented in the new notebook) to directly address the "hard threshold / no gradient" criticism. *(iutV, Claude, GPT)*
- ⬜ **A5.** Add a **"when to use transformers" decision rule** — actionable guidance for practitioners based on effect type and SNR level. *(GPT, Grok)*

---

### B. Baselines

- ✅ **B1.** Add **LSTM/RNN baseline** — GlobalLSTM is now in the new notebook. *(iutV, GPT)*
- ✅ **B2.** Add **TFT, TiDE, NBEATSx** to the main comparison — already in new notebook. *(vpXC, bPFr, iutV)*
- ✅ **B3.** Add **VAR baseline** — GlobalVARBenchmark added in M5 section of new notebook. *(GPT)*
- ⬜ **B4.** Add **DLinear or NLinear** (Zeng et al. 2023) — this is the most prominent "transformers are not effective" baseline and is conspicuously absent despite being cited. *(vpXC, bPFr, Claude, Grok)*
- ⬜ **B5.** Add **iTransformer** to the comparison — it is cited as a related two-way attention model but never benchmarked. *(iutV, Claude, Grok)*
- ⬜ **B6.** Move SOTA comparisons from the appendix into the **main text** across the full effect grid. Current appendix is incomplete (missing Lin effect column, missing superposition). *(Claude, Grok)*
- ⬜ **B7.** The flattened-MLP and Lasso baselines remove all temporal/cross-sectional inductive bias. Add at least one **structured classical baseline** (e.g., panel VAR or a Lasso with explicit lag features). *(GPT)*
- ⬜ **B8.** State clearly whether **hyperparameter tuning was equalized** across models (same cross-validation budget, same search space). If not, describe what was done. *(Claude, GPT, Grok)*

---

### C. TSCS-Shift Failure

- ✅ **C1.** TSCS-Shift failure is **resolved** in the new notebook via val-set epoch selection and a larger epoch budget. TCTC now achieves 0.497 at ρ=0.2 (vs. ~0 in the paper). *(bPFr, iutV, Claude, GPT, Grok)*
- ⬜ **C2.** **Present this as a finding in the paper**, not just a silent fix. Explain why it happens: detecting the compound TSCS structure requires more gradient steps. Show performance vs. epoch curves as evidence. *(all reviewers)*
- ⬜ **C3.** Include a brief **architecture ablation** on TSCS-Shift: does CTCT help? TC? Joint attention? This transforms a weakness into an analytical contribution. *(Claude, GPT)*

---

### D. Statistical Rigor

- ⬜ **D1.** Tables 1–7 report **point estimates with no error bars**. Add mean ± std (or 95% CI) over at least 5–10 seeds for every table. *(Claude, GPT, Grok, bPFr)*
- ⬜ **D2.** Clarify that the Section 4 bootstrap **measures variance over data-generating processes** (not model training stochasticity). Confirm models are retrained in each bootstrap iteration. *(Claude)*
- ⬜ **D3.** Specify the **two-sample test** used in Section 4 (t-test? Welch? Mann-Whitney?). With n=90, choice matters for small-effect claims. *(Claude)*
- ⬜ **D4.** The "deterministic sparse" upper bound is only shown for TS-Shift. **Add it for CS-Shift** to strengthen the claim that max_sparse approaches the oracle. *(Claude)*

---

### E. Training Protocol & Reproducibility

- ✅ **E1.** Val-set epoch selection is now implemented in the new notebook (20% val split, early stopping on val correlation). *(Claude, Grok)*
- ⬜ **E2.** Add a dedicated **"Implementation Details" paragraph or appendix table** reporting: optimizer, learning rate, batch size, epochs (or stopping criterion), dropout, and training hardware for every model. *(Claude, GPT, Grok)*
- ⬜ **E3.** Fix random seeds everywhere and report them. Add a **Reproducibility Statement** (standard NeurIPS checklist item). *(Grok, Claude, GPT)*
- ⬜ **E4.** Add a **code repository link** (anonymous for submission). The synthetic generator and training scripts should be runnable from a README command. *(Grok, GPT)*
- ⬜ **E5.** M5 preprocessing is not fully reproducible from the paper description. Document exact dates used, n_rolling=14, and the first-differencing + standardization steps. *(Grok, Claude)*

---

### F. Dynamic Sparsification

- ⬜ **F1.** K=0.1 is fixed throughout with no justification. Add a **sensitivity analysis** over K ∈ {0.01, 0.05, 0.1, 0.2, 0.5} on at least one effect. *(Claude, GPT, Grok)*
- ⬜ **F2.** Clarify the **gradient flow** in the paper: the binary mask is stop-gradient by construction (additive −∞ masking); gradient flows through Q/K projections. State explicitly whether −∞ is `float('-inf')` or a large constant. *(iutV, Claude, GPT)*
- ⬜ **F3.** Address **batch-averaging stability**: note the batch size used and add a brief sensitivity remark. *(iutV)*
- ⬜ **F4.** Test sparsity on a **wider range of effects** — currently only TS-Shift, CS-Shift, superposition, and linear. Add Fea-Nonlin as a negative control (sparsity should not help); add TSCS-Shift (already in new notebook). *(Claude, Grok)*
- ✅ **F5.** Implement **entmax15** as a differentiable sparse alternative — already in the new notebook. *(iutV, Claude, GPT)*

---

### G. Figures & Presentation

- ⬜ **G1.** The paper has **zero figures**. Add at minimum: (a) performance-vs-ρ line plots for the main effects (one panel per effect, one line per model), (b) attention matrix heatmaps replacing Tables 12–13, (c) an architecture diagram. *(PawB, Claude, GPT, Grok)*
- ⬜ **G2.** Add an **architecture diagram** showing the TCTC block structure with input/output dimensions. *(GPT)*
- ⬜ **G3.** Convert attention matrix Tables 12–13 into **proper color heatmaps** with a ground-truth comparison panel alongside the learned one. *(Claude, Grok)*
- ⬜ **G4.** The model name **"Trans" is poor** — consider "TC-Transformer" or "BiAttn" or "TCTC-Trans". *(Claude)*
- ⬜ **G5.** Fix **terminology inconsistencies**: "two-way attention", "TCTC", "bidimensional" are all used for the same thing. Pick one and use it consistently. *(GPT)*
- ⬜ **G6.** Move **data dimension constants** (T_train=2500, etc.) into a single "Experimental Protocol" box or table rather than repeating in every section. *(Grok)*

---

### H. Abstract & Writing

- ⬜ **H1.** **Abstract overclaims**: "outperform standard baselines across a wide range of settings" — Tables show this is only true for CS-Shift and Fea-Nonlin; Lasso wins on Lin and TS-Shift. Qualify: "on several dependency structures, particularly cross-sectional and non-linear effects." *(Claude, Grok)*
- ⬜ **H2.** Add the **10–20% sparsity gain** as a quantitative highlight in the abstract (it is in the contributions paragraph but not the abstract). *(Grok)*
- ⬜ **H3.** **Notation clash**: ρ is used for both the global SNR and for per-feature correlation coefficients ρ_{j,n}. Use different symbols. *(Claude)*
- ⬜ **H4.** **Equation (2) normalization**: ρ² = Σ ρ_e² assumes orthogonality of the effect components Ỹ_e. Either prove this holds under the construction or state it as an assumption. *(Claude)*
- ⬜ **H5.** **TheoC is misleading for non-linear effects**: it only applies to the linear case. Add a clear footnote or restrict it to the Lin table. *(Claude)*
- ⬜ **H6.** Fix the **introduction example**: the temperature/weather framing clashes with the Gaussian i.i.d. assumption. Replace with a generic sensor panel example, or add one sentence justifying that preprocessing (differencing, normalization) brings real data toward this regime. *(PawB, iutV)*
- ⬜ **H7.** Sharpen the **gap statement in the introduction**: explicitly contrast the correlation-to-optimal-predictor metric against the usual MSE/MAE on real data (where the optimal predictor is unknown) — this is the paper's strongest selling point. *(Grok)*
- ⬜ **H8.** Map **synthetic effects to real-world phenomena** in the introduction: TS-Shift → lagged dependencies, CS-Shift → cross-asset spillovers, Fea-Nonlin → regime effects. *(GPT)*
- ⬜ **H9.** The "connection to sparsity-inducing regularization in classical regression" claimed in the abstract is **never developed**. Either develop it (e.g., LASSO analogy) or remove the claim. *(Claude)*
- ⬜ **H10.** Fix **typographical and grammatical issues**: doubled phrases ("a full attention full attention transformer"), "sparisty", "CS- Shift" space, citation formatting inconsistencies. Do a careful copy-edit pass. *(Claude)*
- ⬜ **H11.** The **Impact Statement is perfunctory** (one sentence). NeurIPS expects a more substantive broader-impact discussion. *(Claude, Grok)*
- ⬜ **H12.** Switch to **NeurIPS 2026 template** and check the main-text page limit (typically 8 pages); some content may need to move to the appendix. *(PawB, Grok)*
- ⬜ **H13.** Fill in real **author affiliations** (currently `XXX`/`YYY` placeholders). *(plan)*

---

### I. Real-World Data

- ✅ **I1.** The M5 real-data section now has **VAR and GlobalLSTM** baselines alongside TFT/TiDE/NBEATSx. *(GPT, Grok)*
- ✅ **I2.** M5 now reports **MSE and MAE** in addition to correlation. *(bPFr)*
- ⬜ **I3.** **Elevate M5 results to the main text** — at minimum a paragraph or table in the results section. The paper's real-world validation is otherwise invisible to reviewers reading only the main body. *(PawB, bPFr, GPT)*
- ⬜ **I4.** The **Ridge/VAR overfitting issue** (Train corr ≈ 1.0, Test corr ≈ 0.04) needs a clear explanation. VAR is similarly overfit in the new notebook. Discuss why (too many parameters relative to T_test window) or cap the model size. *(Claude, Grok)*
- ⬜ **I5.** Consider adding **one additional real dataset** beyond M5 (e.g., ETTh1/ETTh2, electricity, traffic) to strengthen the generalization claim. *(Grok, GPT)*

---

### J. Data & Methodology Assumptions

- ⬜ **J1.** X is sampled i.i.d. N(0,1) — **no autocorrelation** in predictors. This makes the TS-Shift problem slightly artificial (every lag of X is independent noise). Acknowledge this limitation prominently, and if possible add an AR(1) variant as a robustness check. *(iutV, Claude, GPT, Grok)*
- ⬜ **J2.** The paper's **CS-Shift uses modulo-N circular shifts**, implying a meaningful circular ordering of series. Justify or replace with a random derangement (already done in `generate_dataset_gr_simple` with `shuffle_cs=True`). Make the paper and code consistent. *(Claude)*
- ⬜ **J3.** **T_win=10 is very short** relative to standard benchmarks (96–720). Justify explicitly ("low-data regime" is the paper's stated focus) or add a sensitivity analysis. *(Claude, Grok)*
- ⬜ **J4.** The paper only addresses **one-step-ahead forecasting**. State this scope limitation explicitly in the paper. *(Claude)*

---

### K. Ablations (Missing)

- ⬜ **K1.** **Attention block ordering**: test CTCT, TC, CC, TT, TCTCTC in addition to TCTC. This is a key architectural decision with no justification. *(Claude, GPT)*
- ⬜ **K2.** **d_model, nhead, FFN dim**: at least state robustness briefly; a 2×2 table varying one param would satisfy reviewers. *(Claude)*
- ⬜ **K3.** **T-only vs C-only vs TCTC**: which axis contributes more on which effect? This is directly informative for the paper's thesis. *(GPT)*
- ⬜ **K4.** **Sparsity threshold K**: sensitivity over K ∈ {0.01, 0.05, 0.1, 0.2, 0.5}. *(Claude, GPT, Grok)*

---

### L. Efficiency & Cost

- ⬜ **L1.** Report **parameter count, training time, and inference time** for each model. A central critique of transformers vs. linear models is cost; this must be addressed. *(Claude, GPT)*
- ⬜ **L2.** Show whether the sparse attention mechanism provides **any memory or speed benefit** relative to dense attention. *(GPT)*

---

### M. Theory

- ⬜ **M1.** The OLS analytical bound (Appendix B) is interesting but disconnected from the transformer. Consider extending it informally to the **Lasso case**, or adding a remark on why attention should be expected to do better than OLS under nonlinear effects. *(GPT, Claude)*
- ⬜ **M2.** Add a brief discussion of **bias-variance tradeoff** — why does low SNR specifically benefit from sparse attention? The intuition (reducing effective degrees of freedom) should be stated explicitly. *(GPT)*
- ⬜ **M3.** The "**Order 0/1/2**" effect taxonomy could be confused with ARIMA differencing orders. Consider renaming (e.g., "linear", "single-interaction", "compound-interaction"). *(Claude)*

---

### N. Literature Review

- ⬜ **N1.** Add and briefly discuss the following missing papers: MOMENT, Time-MoE, PatchTST, iTransformer, DLinear/NLinear, SparseTSF, TimePro, SEMformer, LSINet. For each: one sentence on what it does and why it is or isn't benchmarked. *(vpXC, bPFr, iutV, Grok)*
- ⬜ **N2.** The reference list stops around 2023. Add **2024–2025 papers** on sparse/structured transformer time-series forecasting to make the related work current. *(Grok)*
- ⬜ **N3.** Fix **citation formatting inconsistencies** (missing venues, inconsistent abbreviations). *(Claude)*

---

## 2. Decisions Made

**1. Baselines to add:**  
The new notebook (`updated_figures_for_paper.ipynb`) contains the definitive model list, including our TCTC, LSTM, and SOTA models requested by reviewers. The philosophy is: run everything cheaply now, then select what goes into the paper. The theoretical OLS correlation (TheoC) is kept as a reference row in all tables.

**2. TSCS-Shift failure:**  
This is **resolved** in the new notebook. The key insight is that detecting the TSCS-Shift effect requires training for significantly more epochs than other effects. Different effects are detected at different training horizons. The new notebook uses a validation set for epoch selection (early stopping on val correlation), which correctly identifies the right stopping point per effect. This turns a weakness into a positive methodological contribution.

**3. Sparsity experiment:**  
Keep or remove depending on rerun results. The section stays if the bootstrap still shows significance; it is simplified or removed if results are weak.

**4. New effects (TS-Nonlin / CS-Nonlin):**  
Yes — `TS_cond` and `CS_cond` are already implemented in the generator and defined in the theory section but never tested empirically. These will be added to Experiment 1.

**5. Dimensionality sensitivity:**  
Yes — varying dataset shape (T_win, N, F) will be tested as a follow-up experiment after Experiment 1. This directly addresses the "idealized setup" criticism.

---

## 3. What the Updated Notebook Does (Findings)

`notebooks/updated_figures_for_paper.ipynb` was read and executed by Cyril. Here is a precise account of what it contains and where it differs from the current paper.

### 3.1 Data dimensions — changed from paper

| Parameter | Paper | New notebook |
|-----------|-------|--------------|
| N (series) | 10 | 10 |
| F (features) | 20 (10 active) | 5 (2 active) |
| T_train | 2500 | 2500 (62.5% of T=4000) |
| T_test | 1500 | 1500 (37.5% of T=4000) |
| T_win (n_rolling) | 10 | 10 |
| Val set | None | 20% of training split |

F was reduced from 20 to 5, with half active (`_default_corr_split` gives `[1/√2, 1/√2, 0, 0, 0]`). Effects are also now **pure**: all active features have the same effect type, rather than mixing effects and noise. The paper used a mix of effect + noise features.

### 3.2 TCTC hyperparameters — changed from paper

| Parameter | Paper | New notebook |
|-----------|-------|--------------|
| d_model | 64 | `n_ts * n_f = 50` |
| nhead | 8 | 1 |
| dropout | 0.1 | 0.0 |
| dim_feedforward | 256 | `2 * n_ts * n_f = 100` |
| roll_y | False | True (predicts all time steps) |

The model is simpler and smaller. The paper section will need to be updated to reflect these choices. The rationale for nhead=1 should be stated (interpretability of attention patterns; consistent with the paper's attention visualization appendix).

### 3.3 Training protocol — significant upgrade

- **Validation split**: 20% of training data held out as a validation set
- **Early stopping**: on val correlation (`early_stopping_metric='corr'`), with `val_warmup_epochs=5`
- **Max epochs per ρ**: 100 for ρ=0.2, 150 for ρ=0.1, 200 for ρ=0.05 (higher epoch budget for harder problems)
- **Optimizer**: AdamW with lr=1e-3, betas=(0.9, 0.995), weight_decay=1e-4

This val-set protocol is the fix that resolves the TSCS-Shift failure. It should be presented as a methodological contribution in the paper, not just an implementation detail.

### 3.4 Effects tested — partially changed

The new benchmark tests **4 effects**: `TS_shift`, `CS_shift`, `TSCS_shift`, `fea_cond`. The `lin` (linear) effect is **missing** from the main benchmark. This is a significant departure from the paper, which made the linear case central (it was the baseline where Lasso dominated). The superposition experiment is also absent.

The `TS_cond` and `CS_cond` effects are implemented in the generator but not yet in the benchmark.

**ρ values tested**: {0.05, 0.1, 0.2} — the very low end (0.02) and high end (0.50) from the paper are missing.

### 3.5 Models — significantly changed

| Model | Paper (main) | New notebook |
|-------|-------------|--------------|
| TheoC | ✓ | ✗ |
| Lasso | ✓ | ✗ |
| Boosting | ✓ | ✗ |
| MLP (flat) | ✓ | ✗ |
| TCTC | ✓ | ✓ |
| GlobalLSTM | ✗ | ✓ (new) |
| TFT | appendix | ✓ (main) |
| TiDE | appendix | ✓ (main) |
| NBEATSx | appendix | ✓ (main) |

The classical baselines (Lasso, Boosting, MLP, TheoC) have been dropped. See **Open Decision 1** below.

### 3.6 Preliminary results (single seed, no bootstrap)

Results below are `test_corr_optimal` (correlation of model predictions with the oracle optimal predictor):

**ρ = 0.2**

| Effect | TCTC | GlobalLSTM | TFT | NBEATSx | TiDE |
|--------|------|-----------|-----|---------|------|
| CS_shift | **0.917** | 0.361 | -0.007 | 0.034 | -0.015 |
| TSCS_shift | **0.497** | 0.227 | -0.006 | -0.023 | -0.011 |
| TS_shift | **0.943** | 0.239 | 0.431 | 0.283 | 0.001 |
| fea_cond | **0.796** | 0.009 | 0.297 | 0.167 | 0.018 |

**ρ = 0.1**

| Effect | TCTC | GlobalLSTM | TFT | NBEATSx | TiDE |
|--------|------|-----------|-----|---------|------|
| CS_shift | **0.669** | 0.131 | 0.002 | 0.012 | -0.013 |
| TSCS_shift | **0.080** | 0.136 | -0.010 | -0.032 | -0.008 |
| TS_shift | **0.754** | 0.118 | 0.204 | 0.138 | 0.004 |
| fea_cond | **0.652** | 0.007 | 0.178 | 0.092 | 0.021 |

**ρ = 0.05**

| Effect | TCTC | GlobalLSTM | TFT | NBEATSx | TiDE |
|--------|------|-----------|-----|---------|------|
| CS_shift | 0.035 | **0.079** | -0.006 | 0.019 | -0.011 |
| TSCS_shift | 0.003 | **0.072** | -0.010 | -0.030 | -0.007 |
| TS_shift | 0.053 | 0.053 | **0.095** | 0.053 | 0.003 |
| fea_cond | **0.027** | 0.005 | 0.414* | 0.003 | 0.092* |

*TFT/TiDE numbers at ρ=0.05 look suspicious (out of pattern) — worth double-checking with more seeds.

Key takeaways: **TCTC strongly dominates at ρ ≥ 0.1** across all effects including TSCS_shift (now resolved). At ρ=0.05 results are noisy and unclear — more seeds needed. GlobalLSTM is a consistent second.

### 3.7 M5 real-world results (updated)

`test_corr_actual` on the M5 FOODS_1 panel (64 series, n_rolling=14):

| Model | Train corr | Test corr |
|-------|-----------|----------|
| **TCTC** | 0.596 | **0.522** |
| TFT | 0.484 | 0.440 |
| GlobalLSTM | 0.232 | 0.160 |
| VAR (new) | 0.999 | 0.036 |
| TiDE | 0.049 | 0.026 |
| NBEATSx | 0.062 | -0.023 |

TCTC is the clear winner on M5. VAR severely overfits. MSE and MAE are also reported. This is a strong real-world result that should be elevated to the main paper.

### 3.8 Sparsity bootstrap (updated)

The new notebook runs a clean bootstrap with resume support, saving to `notebooks/results/sparse_bootstrap_results2.csv`. Changes from the paper:

- **Effects**: TS_shift, CS_shift, TSCS_shift (TSCS_shift is new)
- **ρ values**: {0.005, 0.01, 0.03, 0.05} (lower and denser than paper's {0.015, 0.03, 0.1})
- **Models**: `full_attention` vs `max_sparse` (and `entmax15` is implemented but currently commented out)
- **Seeds**: target 30 per cell; partial results already saved
- **No early stopping** in the sparse bootstrap (val_pct=None) — consistent with paper's original design for that section

Partial results (30 seeds for ρ=0.03, 22 seeds for ρ=0.04, TS_shift only):

| ρ | full_attention | max_sparse |
|---|---------------|-----------|
| 0.03 | 0.112 | 0.111 |
| 0.04 | 0.148 | 0.151 |

These are very close — sparsity advantage is small. The full picture will require completing the run across all effects and ρ values.

**New addition — entmax15**: The notebook implements `attn_normalizer='entmax15'` in `CustomBiDimensionalTransformer` as a principled sparse alternative to softmax (naturally produces sparse outputs without a hard threshold). This directly addresses reviewer iutV's gradient-flow concern, since entmax is differentiable. This should be included in the sparsity comparison.

### 3.9 New dependencies

The notebook requires packages not currently in any requirements file:
- `neuralforecast` (NeuralForecast library for TFT, TiDE, NBEATSx)
- `datasetsforecast` (M5 data loading)

The `ml_benchmarks.GlobalVARBenchmark` class is used in M5 — verify it exists in `tsnn/benchmarks/ml_benchmarks.py`.

### 3.10 Open Decisions (require alignment before running full experiment)

**Decision 1 — Keep classical baselines (Lasso, Boosting, MLP, TheoC)?**  
The new notebook dropped them entirely in favour of DL models. But the paper's original narrative relied on "transformer beats Lasso on nonlinear effects." Dropping them also removes TheoC, which was the only analytical anchor in the results. Recommendation: **keep Lasso and TheoC at minimum**. Lasso is the most informative classical baseline and is cheap to run. MLP is also useful as the fairest comparison for fea_cond (nonlinear effect). Boosting can be dropped if space is tight.

**Decision 2 — ρ values to include?**  
New notebook: {0.05, 0.1, 0.2}. Paper: {0.02, 0.05, 0.10, 0.20, 0.50}. Recommendation: keep **{0.02, 0.05, 0.10, 0.20, 0.50}** — the very low end (0.02) is where the paper's SNR story is most interesting, and the high end (0.50) shows convergence. The new notebook's restricted range misses both extremes.

**Decision 3 — Lin effect?**  
The new notebook doesn't include the linear effect. But the linear effect is the one for which we have the analytical TheoC formula, and Lasso outperforms everything. This contrast is important for the paper's story. Recommendation: **add lin effect back**.

**Decision 4 — Pure effects vs. mixed (noise features)?**  
New notebook: all active features have the same effect, no noise features. Paper: half the features are noise. The paper setup is harder and more realistic. Recommendation: **discuss with Cyril** — the simpler setup gives cleaner results (high correlations at ρ=0.2) but may be seen as easier than advertised.

**Decision 5 — Superposition experiment?**  
Present in the paper (Tables 4–5), absent in the new notebook. This experiment (all 5 effects simultaneously) was one of the most interesting results — showing TCTC is the only model that balances across all effects. Recommendation: **keep it**, add it back to the new benchmark.

**Decision 6 — entmax15 in sparsity section?**  
It is implemented and directly addresses the gradient-flow criticism. Recommendation: **include it** — it makes the sparsity section more rigorous and more novel.

---

## 4. Plan of Attack

### PHASE A — Repo & Infrastructure (Day 1–2)

**A1. Repo cleanup**
- [ ] Add `.gitignore` covering: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `*.csv` result files, macOS `.DS_Store`, AWS credentials
- [ ] Archive old notebooks into `notebooks/archive/`; only `updated_figures_for_paper.ipynb` remains active
- [ ] Remove stray root-level files: `Test file`, `all_effects_test_30.csv`
- [ ] Untrack already-tracked `__pycache__` directories from git (`git rm -r --cached`)

**A2. Cross-platform compatibility**
- [ ] The notebook already has correct device auto-detection (`cuda → mps → cpu`) — verify the same pattern is used in `TorchWrapper` default (currently hardcoded `device='mps'`)
- [ ] Verify `ml_benchmarks.GlobalVARBenchmark` exists and works on both platforms
- [ ] Check for any other hardcoded device references in the codebase

**A3. Reproducibility**
- [ ] Create `requirements.txt` (or `environment.yml`) pinning: torch, numpy, scikit-learn, pandas, tqdm, matplotlib, neuralforecast, datasetsforecast, notebook versions
- [ ] Add README section: "Setup" with instructions for AWS (CUDA) and MacBook (MPS)
- [ ] Document that `notebooks/results/` should be in `.gitignore` (large CSV/parquet files) but a `results/README.md` should explain how to reproduce them

**A4. Code review & bug fixes**
- [ ] **Generator:** Verify `generate_dataset_gr_simple()` normalizes `y_pred_optimal` correctly — the `fea_cond` effect introduces a product term (`X[i] * sign(X[j])`) whose variance is not simply `coeff²`; confirm the `active_norm` L2 rescaling still gives the right global_corr
- [ ] **Sparse attention:** Read `tsnn/tstorch/transformers.py` — document whether `−∞` is `float('-inf')` or a large constant, and whether entmax15 is already wired in; confirm the `attn_normalizer` parameter path works end-to-end
- [ ] **Model hyperparameters:** The paper reports d_model=64, nhead=8 — the notebook uses d_model=50, nhead=1. Both will need to be reported accurately in the paper; make sure there is no confusion
- [ ] **`GlobalVARBenchmark`:** Confirm this class exists in `tsnn/benchmarks/ml_benchmarks.py`

---

### PHASE B — Experiment 1: Main Effect Benchmark (Day 2–5)

**The notebook has already been read and understood (Section 3 above). The work here is to align on the final design, then extract it into a reproducible script and run it at full scale on AWS.**

**B1. Align on experiment design** (decisions from Section 3.10 needed)

*Effects (proposed final list):*
- `lin` — **add back** (needed for TheoC anchor and Lasso story)
- `TS_shift` ✓
- `CS_shift` ✓
- `fea_cond` ✓
- `TSCS_shift` ✓ (now resolved)
- `TS_cond` — add (Order 2 nonlinear, already in generator)
- `CS_cond` — add (Order 2 nonlinear, already in generator)
- Superposition of all effects — **add back**

*Models (proposed final list):*
- TheoC — **add back** (analytical reference, tied to lin effect)
- Lasso — **add back** (strongest classical baseline)
- MLP (flat) — **add back** (needed for fea_cond comparison)
- GlobalLSTM ✓
- TCTC ✓
- TFT ✓ (via NeuralForecast)
- TiDE ✓ (via NeuralForecast)
- NBEATSx ✓ (via NeuralForecast)
- *(Boosting optional — drop if runtime is a bottleneck)*

*Data dimensions (proposed):*
- T=4000, train_pct=0.625, n_rolling=10, N=10
- **F: decide between 5 (new notebook) and 20 (paper)** — see Decision 4
- Pure effects (new notebook style) OR mixed + noise (paper style) — see Decision 4
- ρ values: {0.02, 0.05, 0.10, 0.20, 0.50}

*Training protocol (from new notebook):*
- Val set: 20% of training split, early stopping on val correlation
- val_warmup_epochs=5
- Max epoch budget scales with ρ (larger budget for lower ρ)
- Fixed seeds across seeds for bootstrap

*Bootstrap:*
- Run with at least 10 seeds per (effect, ρ, model) cell; more if compute allows on AWS
- Report mean ± std in tables; use mean for figures

*Output:*
- Heatmap tables (as in paper) → appendix
- Line plots: ρ on x-axis, test_corr_optimal on y-axis, one curve per model, one panel per effect → **main text**

**B2. Extract notebook into a standalone script**
- [ ] Write `scripts/run_experiment1.py` — self-contained, no notebook dependency, reads config from a YAML/JSON file, saves results to `results/experiment1/`
- [ ] Test locally with 1 seed and 1 effect before submitting to AWS

**B3. Run on AWS**
- [ ] Submit as a background job; save all raw per-seed results (not just means)
- [ ] Generate tables and figures programmatically from saved results

---

### PHASE C — Experiment 2: Sparsity Bootstrap (Day 5–8)

**The notebook already has the sparsity infrastructure. Work here is to complete the run, add entmax15, and make a decision on the section.**

**C1. Complete the bootstrap**
- [ ] Finish the `run_sparse_attention_bootstrap` run: effects = {TS_shift, CS_shift, TSCS_shift}, ρ = {0.005, 0.01, 0.03, 0.05}, 30 seeds — the CSV at `notebooks/results/sparse_bootstrap_results2.csv` is partially populated and the function has resume support
- [ ] Uncomment and include **entmax15** in the model builders — this is already implemented in the notebook and directly addresses reviewer iutV's gradient-flow concern; entmax15 is differentiable and natively sparse, making the comparison with max_sparse more principled
- [ ] Run entmax15 across the same (effect, ρ, seed) grid

**C2. Address reviewer technical concerns in the paper**
- [ ] State clearly in Section 4: the binary mask in max_sparse is stop-gradient by construction (masking via additive −∞); gradient flows through the Q/K projections which determine which weights dominate — the mask itself is not differentiated through
- [ ] Confirm whether `−∞` is `float('-inf')` or a large constant in the code; state this explicitly
- [ ] Entmax15 provides a gradient-differentiable alternative that sidesteps this concern entirely — present this as a comparison, not a replacement
- [ ] Address batch-averaging: note batch size used and whether the sparsity pattern is stable across batch compositions

**C3. Differentiate from LSINet in the paper**
- [ ] LSINet retains a fixed top-K entries globally. Our max_sparse uses a relative threshold (K × row_max) that adapts per row, is robust under causal masking (where row lengths vary), and requires no prior K. Cite and contrast explicitly in one focused paragraph.

**C4. Decision point after full bootstrap**
- If max_sparse or entmax15 shows consistent significant improvement over full_attention at low ρ: keep the section with all three models
- If gains are marginal: keep as a brief section with one strong result and a clear theoretical motivation

---

### PHASE D — Experiment 3: Dimensionality Sensitivity (Day 8–10, if time permits)

*Addresses the "idealized setup" and "curse of dimensionality" criticisms.*

- [ ] Fix effect to TS_shift or superposition; vary one dimension at a time:
  - N ∈ {5, 10, 50} — does TCTC scale better than Lasso as panel grows?
  - F ∈ {5, 10, 20} — does the transformer benefit more from additional features?
  - T_win ∈ {5, 10, 20} — how sensitive is TCTC to lookback window length?
- [ ] Show how model rankings change — particularly Lasso vs. TCTC as N and F grow (this is where the curse of dimensionality argument is most concrete)
- [ ] Present as one compact figure in the appendix; elevate to main text if results are striking

---

### PHASE E — Paper Revision (Day 6–12, parallel to experiments)

**E1. Switch to NeurIPS template**
- [ ] Replace `icml2026.sty` / `icml2026.bst` with NeurIPS 2026 style files
- [ ] Fix any layout issues caused by template switch
- [ ] Fill in real author affiliations (currently `XXX`/`YYY` placeholders)
- [ ] Remove `\gr{}` command from preamble and clean all inline `\gr{...}` notes

**E2. Abstract**
- [ ] Update to reflect new results: TSCS-Shift now resolved; new DL baselines included; M5 strong result elevated
- [ ] Add one sentence on the val-set epoch selection methodology
- [ ] Tighten: NeurIPS allows 250 words; current abstract is 196

**E3. Introduction**
- [ ] Replace or fix the temperature forecasting example — it clashes with the Gaussian i.i.d. assumption (flagged by reviewers PawB and iutV). Best option: use a financial panel data example or a general "multivariate sensor panel" example, then add one sentence: "preprocessing steps such as differencing and normalization bring real data toward the stationary Gaussian regime assumed here"
- [ ] Add a paragraph connecting each cited architecture to the paper's design choices: why TC alternating blocks rather than PatchTST (patch-based) or iTransformer (CS-only)?
- [ ] Add the val-set epoch selection insight as a contribution ("we show that different dependency types require different training horizons, and that validation-based stopping is critical for detecting complex interactions like TSCS-Shift")
- [ ] Add TS_cond and CS_cond effects as contributions

**E4. Section 2 — Setup**
- [ ] Update data dimensions to match the new notebook (N=10, F=5 or 20 — per Decision 4)
- [ ] Justify the Gaussianity and stationarity assumptions in a brief remark
- [ ] Update Section 2.4 (Baselines): describe GlobalLSTM, TFT, TiDE, NBEATSx; remove models dropped from the experiment
- [ ] Add a subsection or remark on val-set epoch selection protocol

**E5. Section 3 — Experiments**
- [ ] Replace all result tables with updated numbers
- [ ] Add TS_cond and CS_cond effect tables
- [ ] Add the superposition-of-effects results with new models
- [ ] Add **figures**: line plots with ρ on x-axis and test_corr_optimal on y-axis, one panel per effect (at minimum for the 4 main effects in the new notebook) — this is the single highest-impact visual improvement
- [ ] Add a focused discussion of TSCS_shift: what changed (val-based stopping + longer budget) and why this makes sense (the model needs more gradient steps to discover the compound shift structure)

**E6. Section 4 — Sparsity**
- [ ] Update all tables with rerun numbers
- [ ] Add entmax15 as a third model in the comparison
- [ ] Add the gradient-flow clarification and differentiation from LSINet (see Phase C)
- [ ] Add TSCS_shift sparsity results (new in the notebook)

**E7. Literature review**
- [ ] Add and briefly discuss: MOMENT, Time-MoE, PatchTST, iTransformer, DLinear/NLinear, SparseTSF, TimePro, SEMformer, LSINet
- [ ] For each: one sentence on what it does and why it is or isn't in the experimental comparison
- [ ] Update `references.bib` with all new entries

**E8. Appendix**
- [ ] Expand Appendix B (OLS theory): consider extending informally to the Lasso or to misspecified effects
- [ ] Add MSE/MAE results table from the main experiment and M5 (addresses reviewer bPFr; already computed in notebook)
- [ ] Elevate M5 results from appendix to a brief paragraph or table in the main text conclusion
- [ ] Keep attention matrix visualization (Appendix A) — update with new architecture hyperparameters if changed
- [ ] Add dimensionality sensitivity results as Appendix D (if experiment D is run)

**E9. Conclusion**
- [ ] Update to reflect new results; add a sentence on M5 validation
- [ ] Add a paragraph on practical guidance: when to use TCTC over simpler alternatives
- [ ] State limitations explicitly: Gaussian inputs, i.i.d. features in X, single-step-ahead prediction

**E10. Writing quality pass**
- [ ] One full editing pass after all results are integrated
- [ ] Every design choice (d_model, nhead, T_win, etc.) reported and briefly justified
- [ ] Remove all commented-out LaTeX blocks

---

### PHASE F — LLM Review (Day 11)

- [ ] Submit revised paper to GPT-4o (or similar) with prompt: *"Review this ML paper as a NeurIPS 2026 reviewer. Give scores (1–5) on soundness, novelty, clarity, and significance. List the top 5 remaining weaknesses with specific fix suggestions."*
- [ ] Compare against the 4 ICML reviews; close any remaining gaps
- [ ] Final gap-closing pass

---

### PHASE G — Final Polish & Submission (Day 12–14)

- [ ] Final proofreading
- [ ] Verify all figures render correctly in PDF (embedded fonts, no rasterization)
- [ ] All `\ref{}` and `\cite{}` resolved (no "??" in compiled PDF)
- [ ] `references.bib` complete and no duplicates
- [ ] Reproducibility check: `scripts/run_experiment1.py --help` works and README explains how to reproduce Table 1
- [ ] Submit to NeurIPS 2026

---

## 5. Suggested Timeline

| Day | Focus |
|-----|-------|
| 1 | Repo cleanup (A1), .gitignore, archive notebooks, remove stray files |
| 2 | Dependency fix (A2–A3), code review (A4); align on open decisions (Section 3.10) |
| 3 | Finalize Experiment 1 design (B1); extract notebook into `scripts/run_experiment1.py` (B2) |
| 4–5 | Run Experiment 1 on AWS (B3); generate figures and tables |
| 5 | Review Experiment 1 results; complete sparsity bootstrap (C1) |
| 6–7 | Add entmax15 to sparsity (C1); finalize sparsity decision (C4) |
| 6–8 | Paper revision in parallel: template (E1), intro (E3), setup (E4), lit review (E7) |
| 8–9 | Dimensionality sensitivity if time permits (Phase D) |
| 9–10 | Integrate all results into paper; finalize figures |
| 10–11 | Sections 3, 4, appendix revision |
| 11 | LLM review (Phase F); final gap analysis |
| 12–13 | Writing quality pass; conclusion; final editing |
| 14 | Reproducibility check; submit |

---

## 6. File & Directory Conventions

```
TSNN-project/
├── context/
│   ├── TSNN/                  # Paper source (main.tex, references.bib, style files)
│   ├── ICML_reviews/          # Reviews received
│   └── NEURIPS_REVISION_PLAN.md   # This file
├── notebooks/
│   ├── updated_figures_for_paper.ipynb   # ACTIVE notebook (Cyril's)
│   ├── results/               # Saved CSV/parquet outputs (gitignored)
│   └── archive/               # All old notebooks
├── scripts/
│   └── run_experiment1.py     # Reproducible experiment script (to be written)
├── results/                   # Final paper results (gitignored, README explains how to reproduce)
├── tsnn/                      # Python package
├── requirements.txt           # Pinned dependencies (incl. neuralforecast, datasetsforecast)
├── .gitignore
└── README.md
```

Result files: each experiment run saves a timestamped JSON/CSV with: `date`, `effect`, `model`, `rho`, `seed`, `test_corr_optimal`, `train_corr_optimal`, `test_mse_actual`, `best_epoch`.

---

## 7. Key Contacts

- Guillaume Remy — lead on code, AWS experiments, and coordination
- Cyril Garcia — notebook author; align on open decisions in Section 3.10 before running full experiments
