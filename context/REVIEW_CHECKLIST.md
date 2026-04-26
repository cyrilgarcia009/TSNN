# Combined Review Checklist

Synthesizes all issues raised across **4 ICML reviewers** (vpXC, bPFr, PawB, iutV) and **3 AI model reviews** (Claude, GPT-4o, Grok) into a single actionable checklist, grouped by theme.

**Legend:** ✅ resolved in new notebook | ⬜ to do

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
