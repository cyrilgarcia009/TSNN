Comprehensive list of issues and improvements for resubmission to NeurIPS
I have read the full paper (all 12 pages + appendices) carefully. Below is an exhaustive list of issues, grouped by section/theme. I flag scientific/methodological weaknesses, presentation/clarity problems, overclaims vs. evidence, reproducibility gaps, formatting/NeurIPS-specific concerns, and opportunities to strengthen the paper. These are based on typical NeurIPS reviewer expectations (strong empirical validation, clear novelty, rigorous baselines, reproducibility, and balanced claims). Many issues likely contributed to the ICML rejection.
1. Abstract & High-Level Positioning

Overclaiming performance: The abstract says the two-way transformer “can outperform standard baselines … across a wide range of settings, including low signal-to-noise regimes.” Tables 1–7 show this is only sometimes true (strong on CS-Shift/Fea-Nonlin, weak/poor on TSCS-Shift and often on superposition where Lasso wins optimal correlation). Qualify with “in several dependency structures and especially under very low SNR” or focus claims on sparsity gains + balanced effect capture.
Missing quantitative highlight: The 10–20% gain from dynamic sparsity (mentioned in contributions) never appears in the abstract. Add it.
NeurIPS fit: NeurIPS values papers that advance architectural understanding or practical robustness. The synthetic + limited real-data setup is fine, but explicitly contrast with “real-benchmark overfitting” papers (Zeng et al., Das et al.) earlier to sell the controlled-ground-truth advantage.

2. Introduction & Related Work

Gap statement could be sharper: You correctly cite the recent skepticism of transformers, but do not contrast your direct correlation-to-optimal-predictor metric with the usual MSE/MAE on real data (where optimal predictor is unknown). This is your strongest selling point—emphasize it.
Two-way attention not novel enough: You cite Ho et al. (2019) and Liu et al. (2024) (iTransformer). Reviewers will ask “what exactly is new beyond alternating T/C blocks?” Clarify the precise ordering (“TCTC”) and any differences from iTransformer.
Missing recent citations: Add 2024–2025 works on sparse/transformer time-series (e.g., any follow-ups to iTransformer, PatchTST, or sparse attention in forecasting). The reference list stops feeling current.

3. Methodology & Experimental Setup (biggest section for improvements)

Hyperparameters completely unspecified: d_model=64, FF dim=256, L (number of blocks?), learning rate, optimizer, epochs, batch size, dropout=0.1 are given but never justified or ablated. Was any tuning done? For NeurIPS this is a red flag—add a short “Implementation details” paragraph or appendix table.
Baselines are weak/incomplete:
No univariate time-series classics (ARIMA, Exponential Smoothing) despite intro mention.
Global Lasso/Boosting/MLP flatten everything—fair, but add a temporal baseline (e.g., Temporal Fusion Transformer or simple LSTM/GRU) earlier, not just in Appendix C.
Appendix C compares TFT/TiDE/NBEATSx but only on 4 effects at 2 ρ values and no full superposition. Move expanded version to main paper or justify why omitted.

Data generation assumptions: X ~ i.i.d. N(0,1) stationary is convenient but unrealistic for real finance/energy data. Discuss (or add ablation with AR(1) noise, heavy tails, etc.).
Effect definitions: Order 0/1/2 is nice, but TSCS-Shift (the one your model fails on) is the most realistic “joint spatio-temporal” case. The paper acknowledges the shortcoming but does not analyze why (attention ordering? masking? depth?). Add ablation or visualization.
Dimensions too small/specific: T_train=2500 (~7 years daily), N=10, F=20, Twin=10. Justify vs. real panels (M5 has 64 series but you use only FOODS_1). Add sensitivity (vary N, Twin) or at least state it represents “low-data regime”.
ρ selection: You test 0.02–0.50 but sparsity experiments use even lower (0.015). Be consistent; include 0.015 in all tables.
No error bars / statistical testing except in sparsity section. Every table needs std. dev. or bootstrap p-values (you already do 90 bootstraps—extend it).

4. Dynamic Sparsity Section (strongest contribution—needs polishing)

Algorithm clarity: “Max attention” (K=0.1) is clever and data-driven, but K is fixed with no ablation/sensitivity. Add a small table varying K=0.05/0.1/0.2.
Comparison to prior sparse attention: You cite Child et al. (2019) but do not show your method is better than top-k, strided, or learned sparsity patterns on the same architecture. Add one ablation.
Generalization: You only test on pure TS-Shift/CS-Shift + linear + superposition. Show it helps on Fea-Nonlin or M5.
Visualization: Tables 12–13 are useful but ugly in text form. Convert to heatmaps (color scale) in the appendix or main paper.

5. Results & Discussion

Mixed results not sufficiently discussed:
Lasso often wins “Optimal” correlation in superposition (Tables 6–7) while Trans is better at balanced effect capture. Frame this as a strength (interpretability via effects) rather than weakness.
TSCS-Shift failure is repeatedly called out but never fixed or deeply explained.
At ρ=0.02 the absolute correlations are tiny (0.03–0.05). Discuss practical significance (e.g., in finance this still beats random).

Real-data experiment (M5, Appendix D) is under-developed:
Only one dataset, one preprocessing (first-differenced sales).
n_rolling window size never specified numerically.
Train correlation for Ridge=1.000 screams massive overfitting—discuss regularization or why it fails.
Expand to 2–3 more standard benchmarks (electricity, traffic, weather) if possible; otherwise acknowledge “preliminary real-data validation”.

No ablation studies: No removal of positional embeddings, no varying number of T/C blocks, no multi-head vs. single-head comparison.

6. Appendices & Supplementary Material

Appendix C (SotA): Too limited—move a cleaner version to main body or expand.
Appendix D (M5): Needs full reproducibility details (exact dates, n_rolling, code for preprocessing).
No code or data: NeurIPS strongly prefers public code for synthetic generator + trained models. Plan to release on GitHub with anonymous link.

7. Writing, Clarity & Presentation

Equation rendering: Many �Y, �Y etc. placeholders in the provided text (likely PDF-to-text artifacts); ensure final LaTeX is perfect.
Table formatting: Colored cells in Tables 1–11 look nice but NeurIPS style may prefer clean bolds. Add caption summaries (“higher is better”; best in bold).
Repetition: Data dimensions (T_train=2500 etc.) repeated many times—move to a “Experimental protocol” box.
Long sentences: Introduction and Section 2.2 are dense. Break them.
Figure quality: Only tables for attention matrices. Use proper heatmaps + perhaps learned vs. optimal attention overlay.
Length: 9 pages main + appendices. NeurIPS main text limit is typically 8 pages (2025 cycle); you may need to move more to appendix.
Ethics / Broader Impact: Only a one-sentence Impact Statement. NeurIPS requires a proper one (even if minimal).

8. Reproducibility & Open-Science Gaps (critical for acceptance)

No seed, no exact ρ_j,n sampling distribution, no training code.
M5 preprocessing not fully reproducible from description.
Add a “Reproducibility statement” (standard NeurIPS checklist item).

9. Strategic Advice for NeurIPS Resubmission

Address likely ICML reviewer concerns (guessing from content): limited novelty of two-way attention, weak real-data validation, mixed results, no error bars, no hyperparameter details. Explicitly respond in a rebuttal-style paragraph (even if not required) or cover letter.
Strengthen claims: Lead with “We show that two-way transformers + dynamic max-sparsity outperform dense baselines and several SotA models in low-SNR regimes (ρ ≤ 0.05) where classical methods fail, while remaining competitive otherwise.”
New experiments to add (high impact, low effort):
Error bars + p-values on all tables.
Ablation on sparsity threshold K.
One more real dataset.
Comparison to iTransformer (cited but not benchmarked).

Minor fixes: Fix all OCR-style equation artifacts, consistent table numbering, update NeurIPS template, double-check references for 2024–2025 papers.

Implementing the major items (hyperparams, error bars, expanded real-data/M5 details, qualified claims, sparsity ablations) will make the paper significantly stronger. The core idea (synthetic benchmarking + dynamic sparsity for noisy multivariate forecasting) is solid and timely—NeurIPS reviewers will like the statistical rigor and the “why transformers work under noise” angle. If you share the LaTeX source or specific reviewer comments from ICML I can give even more targeted advice. Good luck with the resubmission!