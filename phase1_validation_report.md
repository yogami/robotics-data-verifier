# Phase 1 Validation Report — V2 (Fable 5 V11 Overhaul)

## Methodology
- **Trajectory Generator V2**: Realistic physics with band-limited tremor (4-12 Hz),
  asymmetric bimanual tasks, leader-follower lag (50-100ms), and encoder noise (1e-3 rad).
- **Hesitation injection**: Extends trajectory (no truncation/label leakage).
- **Confounders**: Legitimate pauses at 0.2s, 0.3s, 0.4s, 0.45s; asymmetric bimanual;
  leader-follower lag; noisy clean baselines.
- **Statistical rigor**: 10 seeds, Clopper-Pearson CIs.

## Adversarial ROC Results
- **AUC**: 0.8735 ± 0.0386 (across 10 seeds)
- **FPR (threshold=0.5)**: 0.0000
- **Clopper-Pearson 95% CI**: [0.0000, 0.0074]

## Psychometric Boundary Curve
Detection rate by hesitation duration:

| Duration (s) | Detection Rate |
|:---:|:---:|
| 0.1 | 0.247 |
| 0.2 | 1.000 |
| 0.3 | 1.000 |
| 0.4 | 1.000 |
| 0.5 | 1.000 |
| 0.6 | 1.000 |
| 0.7 | 1.000 |
| 0.8 | 1.000 |
| 0.9 | 1.000 |
| 1.0 | 1.000 |
| 1.1 | 1.000 |
| 1.2 | 1.000 |
| 1.3 | 1.000 |
| 1.4 | 1.000 |
| 1.5 | 1.000 |
| 1.6 | 1.000 |
| 1.7 | 1.000 |
| 1.8 | 1.000 |
| 1.9 | 1.000 |
| 2.0 | 1.000 |

## Plots
- ROC Curve: `static/phase1_roc_curve.png`
- Score Histogram: `static/phase1_score_histogram.png`
- Boundary Sweep: `static/phase1_boundary_sweep.png`
- Real Data Distribution: `static/phase1_real_data_distribution.png`
