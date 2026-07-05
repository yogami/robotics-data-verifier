# V12 Review

First, credit where it's due: reporting the 100% real-data flag rate instead of burying it is the first genuinely trustworthy thing that's happened in this project. That line is worth more than every synthetic AUC you've ever shown me. Now let me tell you why the rest of this still doesn't clear the bar.

---## 1. Is the methodology satisfactory? No. Better, but no.

**The headline numbers are internally contradictory, and you didn't notice.**

- Your ROC uses **threshold = 0.5** for the FPR claim. Your real-data entropy distribution has **mean 0.73, min 0.35**. At your operational threshold, the FPR on real production data is approximately **90-100%**, not 0.0000. That Clopper-Pearson CI of [0.0000, 0.0074] is an exquisitely precise measurement of the wrong universe. You computed a tight confidence interval on your simulator, then walked outside and every measurement disagreed. The CI is not the achievement; it's the indictment.
- Your boundary sweep uses **detection = entropy > 0.05** — a *different* threshold. At the deployment threshold of 0.5, what is your detection rate at 0.3s? 0.5s? You don't know, because you evaluated sensitivity and specificity at two different operating points. This is the classic move of a system that has no coherent operating point at all. Pick one threshold, report the full confusion picture at that threshold.

**The psychometric curve isn't a psychometric curve.** 24.7% at 0.1s, 100% at 0.2s, flat forever. That's a step function. Real psychometric functions have a sigmoid transition because real signals have variance. Yours doesn't because your "10 seeds" barely inject any: `generate_trajectory_v2` draws tremor phases and encoder noise from the **global** NumPy RNG — your `RandomState(seed)` only sprinkles 5e-4 noise post-hoc. Your seeds don't control the generator. The 300 trials per bin are near-identical clones. Also: your design threshold is 0.5s, but you detect 0.2s pauses at 100% — meaning your detector's effective boundary is 2.5x more aggressive than designed, which is exactly why real data (full of legitimate sub-second dwells) all flags.

**AUC 0.8735 on your own sandbox is bad, not good.** You wrote the corruption generator *and* the detector. In a world where you control both sides, anything below ~0.98 means your detector fundamentally can't separate your own synthetic classes. If it's 0.87 at home, it's a coin flip in the wild. And your real-data result confirms it already is.

**Residual issues:**
- `np.roll` on the right arm wraps the end of the movement to t=0 — a synthetic discontinuity artifact in every "clean" bimanual trajectory.
- Your dwell detector — a headline feature — **abstains on all 14-dim data**. Real Mobile ALOHA data is 14-dim. So the dwell channel never fires in production. You built a smoke detector that turns itself off inside buildings.
- Percentiles over 10 seed-means is not a 95% CI. n=10 percentile bootstrap is numerology.

---

## 2. How to recalibrate SPARC → entropy

The wrong answer — which I suspect you're fishing for — is "slide the threshold until only 5% of real episodes flag." That's percentile outlier detection wearing a lab coat, and it won't transfer to the next dataset, robot, or control frequency. Do this instead:

**a) Fix the SPARC computation itself.** SPARC (Balasubramanian et al.) is defined for *discrete point-to-point movements*, typically 1-5 seconds. You're applying it to entire multi-minute teleop episodes containing dozens of submovements, pauses, and regrasps. Of course it reads "rough" — you're measuring the spectral arc length of a concatenation. Segment episodes into movement units (velocity-threshold segmentation, minimum-duration filter), compute SPARC per segment, aggregate with a robust statistic (median). Consider LDLJ as a cross-check. Until you do this, no threshold is meaningful because the quantity itself is malformed.

**b) Make the gate self-calibrating per dataset.** Absolute thresholds from synthetic sinusoids are dead — bury them. Fit the null on the target dataset with robust statistics: median and MAD of segment-level SPARC. Map to entropy via a smooth robust z-score, e.g. `entropy_sparc = σ(k · (median − sparc)/MAD − b)`, flagging beyond ~2.5-3 MAD. This assumes most production data is acceptable, which for a curated dataset like `aloha_mobile_cabinet` is a defensible prior — and you should *state* that prior explicitly.

**c) Validate the mapping with ground truth, because right now you have none.** Pull the top-k flagged episodes and bottom-k, watch the videos, blind-label them. Report precision@k. If your flagged tail doesn't look visibly worse than the clean bulk to a human, the gate measures nothing. This costs one afternoon and is worth more than another 500-episode synthetic ROC.

**d) Cross-dataset sanity.** Run the recalibrated gate on 4-5 other LeRobot datasets (different tasks, different operators). If the null distributions and flag rates are wildly inconsistent, your normalization is wrong.

---

## 3. Verdict on Phase 1

**Fail, with an honorable mention for honesty.**

What you've actually demonstrated: a detector that achieves mediocre separation on a toy world of its own construction, and flags 100% of real, curated, known-good production data at its operational threshold. Every synthetic number in this report — the AUC, the FPR CI, the psychometric cliff — is a statement about your trajectory generator, not about teleop data. The generator produces clean SPARC of −1.56; real data averages −3.72. Your simulator and reality don't overlap. Everything calibrated in one is undefined in the other.

**As an investor:** I don't kill this, but I fund exactly one more milestone, and it is not another synthetic eval. The product claim is "our gate improves imitation learning." So prove it: train ACT (or DP) on `aloha_mobile_cabinet` three ways — unweighted, gate-weighted, gate-filtered — and show me success rates or at minimum validation loss / action MSE deltas on held-out episodes. That's the only experiment that matters. If gate-weighting doesn't move the downstream metric, the entire signal-processing edifice — SPARC, dwell, entropy — is an elaborate hobby, and no amount of Clopper-Pearson intervals will make it a company.

Stop polishing the sandbox. Go touch the real objective.