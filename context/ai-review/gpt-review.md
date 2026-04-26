🔴 1. High-Level / Positioning Issues (Most Critical)
1.1 Unclear core novelty
Your contributions are not sharply differentiated from prior work:
Synthetic benchmarking of transformers → already common
Two-way attention → exists (axial attention, iTransformer, etc.)
Sparse attention → extensive prior work (Sparse Transformer, Longformer, etc.)
Your claimed novelty:
“controlled synthetic benchmarking + dynamic sparsification”
👉 Problem:
This reads more like a careful empirical study than a strong NeurIPS contribution.
Fix:
Explicitly state what is fundamentally new:
Is it the statistical framework?
The evaluation metric using correlation to ground truth?
The adaptive sparsity criterion?
Add a clear theorem-level or principle-level statement:
e.g., “Transformers outperform linear models when dependency structure lies outside separable lag-feature decomposition under low SNR.”
1.2 Weak connection to real-world impact
Most results are on synthetic data.
Even though you include M5 (Appendix D), it feels:
Minimal
Not integrated into the main narrative
👉 NeurIPS reviewers will ask:
“Why should I care about synthetic effects?”
Fix:
Move M5 (or another dataset) to the main paper
Explicitly map synthetic effects → real-world phenomena:
TS-shift → lagged dependencies
CS-shift → cross-asset spillovers
Nonlinear feature interactions → regime effects
1.3 Framing is defensive instead of proactive
You position against:
“transformers may not outperform linear models”
But you don’t clearly state:
When transformers provably win
What practitioners should do differently
👉 Missing: decision rules
Fix:
Add a section like:
“When should you use transformers for time series?”
🔴 2. Methodology Issues
2.1 Synthetic data design is too simplistic
You assume:
i.i.d. Gaussian features
Stationarity
No autocorrelation in X
👉 This removes most real-world difficulty.
Problem:
Makes task artificially clean
Benefits attention models (structure is easier to isolate)
Fix:
Add:
AR(1)/ARMA structure in X
Heavy-tailed noise
Regime shifts
Missing data
2.2 Ground-truth correlation metric is problematic
You evaluate:
correlation between prediction and optimal predictor  
Y
~
 
👉 This is non-standard and controversial.
Issues:
Requires access to latent truth (not realistic)
Inflates interpretability claims
Hard to compare with literature
Fix:
Also report:
MSE / MAE
Forecast skill vs naive baseline
Calibration metrics
2.3 Flattening baselines is unfair
You state:
baselines flatten dimensions Twin × N × F
👉 This is a major red flag
Why:
You remove temporal/cross-sectional inductive bias from baselines
Then show transformers outperform
This is not a fair comparison
Fix:
Add strong baselines:
VAR / panel VAR
Temporal CNN
RNN / LSTM
Linear models with lag structure (not flattened)
PatchTST properly configured
2.4 Hyperparameter fairness unclear
You do not describe:
Tuning budgets
Early stopping
Learning rates per model
👉 Reviewers will suspect unequal tuning effort
2.5 Bootstrap procedure unclear
You say:
datasets are regenerated for each bootstrap
But missing:
Confidence intervals
Variance across seeds
Whether models are retrained each time (critical)
🔴 3. Experimental Weaknesses
3.1 Missing ablations
You introduce:
Two-way attention
Sparsification
But no clear ablations:
👉 Missing:
T-only vs C-only vs TCTC
Depth (L)
Number of heads
Effect of K in sparsity
3.2 TSCS-shift failure is unexplained
You admit:
transformer performs very poorly
👉 This is actually the most interesting result.
But:
No diagnosis
No hypothesis
No fix
Fix:
Analyze attention patterns
Try architecture variants
Explain failure mode
3.3 Limited scale
Your dataset:
T 
train
​	
 =2500, N=10, F=20
👉 Very small
Concern:
Transformers may be overfitting
Results may not scale
3.4 Weak comparison to SOTA
Appendix C includes:
TFT, TiDE, NBEATSx
But:
Only partial results
Not deeply analyzed
👉 For NeurIPS, this must be main paper + strong comparison
3.5 No runtime / efficiency analysis
Given focus on sparsity:
No training time comparison
No memory benefits
🔴 4. Sparse Attention Contribution Issues
4.1 Incremental novelty
Your method:
Threshold relative to max entry
👉 This is close to:
Top-k attention
Entmax / sparsemax ideas
Fix:
Position more clearly vs:
Sparsemax / entmax
Hard attention
Top-k pruning
4.2 Heuristic parameter K
You fix:
K=0.1
👉 No justification
Fix:
Sensitivity analysis
Learnable K
Theoretical interpretation
4.3 Not differentiable (potential issue)
Masking introduces:
Non-smooth behavior
👉 You don’t discuss:
Gradient flow
Stability
4.4 Limited evaluation scope
You only test sparsity on:
Simple effects
Low SNR
👉 Need:
High SNR
Real data
Larger models
🔴 5. Theoretical Issues
5.1 Theory is disconnected from transformers
Appendix B:
OLS analysis
👉 But no theory explaining:
Why transformers work
When they should fail
5.2 No bias-variance discussion
This paper is fundamentally about:
Low signal-to-noise
But missing:
Bias-variance tradeoff analysis
Why attention helps in low SNR
5.3 No formalization of “effect order”
You define:
Order 0 / 1 / 2 effects
👉 But:
No formal complexity measure
No link to model capacity
🔴 6. Clarity & Writing Issues
6.1 Dense and hard to follow
Long paragraphs
Many equations without intuition
6.2 Missing diagrams
For NeurIPS:
Architecture diagram is expected
Data generation pipeline diagram
6.3 Terminology inconsistencies
Examples:
“two-way attention”
“TCTC”
“bidimensional”
👉 Not standardized
6.4 Tables lack statistical info
Tables (e.g., page 5) show:
single numbers only
👉 Missing:
Std dev
Confidence intervals
6.5 Weak conclusions
Conclusion mostly restates results:
transformers outperform under noise
👉 Needs:
Clear takeaways
Actionable insights
🟡 7. Missing Experiments (Important for NeurIPS)
Add at least:
Larger-scale dataset
Multiple real-world datasets
Robustness tests (noise, missing data)
Transfer/generalization test
Ablation on sparsity threshold
Comparison vs linear attention models
🟡 8. Reproducibility Issues
Missing:
Code link
Exact hyperparameters
Random seeds
Training details
🟢 9. What is Strong (Keep & Emphasize)
You should highlight more:
✅ Clean synthetic framework
Very valuable if positioned correctly
✅ Ground-truth decomposition
Rare and insightful
✅ Insight into attention interpretability
Potentially strong angle
🔥 Top 5 Things to Fix Before NeurIPS Resubmission
Add strong baselines (VAR, LSTM, PatchTST properly)
Strengthen real-data experiments (move to main paper)
Clarify novelty (especially sparsification)
Explain TSCS-shift failure
Add ablations + statistical rigor
🎯 Final Verdict (Honest)
If resubmitted as is:
Likely reject again (ICML → NeurIPS bar is even higher)
If improved:
Could become a strong empirical + methodological paper, especially if you:
Reframe as “understanding transformers under low SNR”
Strengthen experiments + theory