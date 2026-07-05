# Pre-registration Amendment 01: JT Fallback & Hyperparameter Clarification

**Date**: 2026-07-04
**Reason for Amendment**: 
During the initial review of the codebase and the experimental design, two critical issues were identified:
1. The Generalized Estimating Equation (GEE) model implemented in `mixed_effects_model.py` fails when the data is highly degenerate (e.g., exactly 0 successes across all trials), leading to a singular variance matrix and aborting the pipeline rather than recognizing a clear floor effect.
2. The initial manifest erroneously specified hyperparameters that conflicted with the actual MLP-BC implementation.

**Amendments to Protocol**:
1. **Jonckheere-Terpstra Fallback**: If `df["successes"].sum() == 0` or `df["successes"].sum() == df["n"].sum()`, the GEE model will be skipped. Instead, a Jonckheere-Terpstra exact test (implemented via Kendall's tau-b) will be used to detect monotonic trends in the degenerate data. This fallback explicitly handles extreme floor/ceiling effects without pipeline failure.
2. **Hyperparameter Pinning**: The manifest will be updated to include a `hyperparameters_hash` field. This hash is computed from the *actual* hyperparameters used by the trainer, explicitly accepting the MLP-BC configuration. If the trainer uses unapproved hyperparameters, the verification pipeline will fail closed.
