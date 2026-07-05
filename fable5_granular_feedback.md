# Brutal Occam's Razor Review

## The One Fatal Flaw First

**Your roadmap is inverted.** Phase 4 is your only existential risk, and you've scheduled it last. If jerk/entropy weighting doesn't move policy success rates, Phases 1–3 are elaborate scaffolding around nothing. The A/B experiment costs ~$300 of GPU time and two weeks. Run it *first*. Everything else is procrastination dressed as rigor.

**Second fatal flaw: your core hypothesis may be scientifically wrong.**

- **Low jerk ≠ good demonstration.** Contact-rich manipulation *legitimately* produces high jerk at contact events (impacts, insertions, regrasps). Your metric will systematically penalize the hardest, most valuable data.
- **High-entropy segments are often recovery behaviors** — the operator fumbled, then corrected. Recovery data is arguably the *most* valuable content in imitation datasets (this is the entire DAgger insight). Filtering it may make policies *more* brittle, not less.
- **Diffusion policies are specifically robust to multimodal/noisy action distributions.** That's their selling point. You may be solving a problem the architecture already solved.
- **Prior art exists and you didn't cite it:** Re-Mix (data mixture reweighting), CUPID (influence-function demo curation, 2025), DemInf, L2D, quality-weighted BC. Do a literature review before writing a line of code, or your whitepaper gets dismantled in peer review week one.

**Third: your open-core moat is imaginary.** Jerk minimization and Shannon entropy are ~200 lines of NumPy. The moment your whitepaper explains the method, anyone reimplements it in a weekend. The paid "rescue engine" as described has zero defensibility. Your moat must be *position* (the standard, the benchmark, the workflow), not the algorithm. More below.

---

## 1. Phase-by-Phase Critique + Exact Execution Steps

### Phase 1 (Engine) — Cut 50% of it

- **Jerk metric:** Don't use raw jerk — it's not comparable across embodiments, control frequencies, or trajectory durations. Use **Log Dimensionless Jerk (LDLJ)** and **SPARC** (Balasubramanian et al. 2015, the standard smoothness metrics in motor control). Differentiate with **Savitzky-Golay filtering**, not finite differences — at 30–50Hz teleop rates, naive finite-difference jerk is dominated by sensor noise and you'll measure your encoder, not the operator.
- **Entropy score:** Specify *what distribution*. Entropy of what — actions binned per timestep? Windowed action deltas? Per-episode? Entropy confounds task complexity with operator noise. Concrete fix: compute entropy *conditioned on visual state similarity* (nearest-neighbor states across episodes → measure action disagreement). High conditional entropy at similar states = genuine inconsistency; high marginal entropy = maybe just a complex task.
- **Contact-aware gripper safety locks:** Vague, and most public datasets have no F/T or motor-current channels. Cut it from v1. If you must: proxy contact via gripper-position vs. gripper-command divergence (stall detection).
- **Provenance flags:** Keep — trivial and necessary. But **"repair" (synthesizing frames) is your riskiest feature and should be cut entirely from v1.** Inpainted actions won't causally match the recorded observations → you train the policy on physically impossible (observation, action) pairs. **Reweighting the loss is 90% of the value at 10% of the risk.** Ship weight, not repair. Rename the company later if needed.

### Phase 2 (API) — Don't invent a schema

- The robotics data format wars are already lost/won: **LeRobotDataset (parquet), RLDS/TFDS (OXE), MCAP/rosbag**. A new public JSON schema is a standard nobody asked for. Instead: emit per-fr