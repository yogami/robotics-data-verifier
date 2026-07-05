Short answer: you get a **conditional greenlight on Phase 1's unit-test tier only**, and I'm about to explain why your victory lap over the schema is actually a confession of a much worse problem.

## The Schema "Correction" Is a Self-Own

You've verified that `lerobot/aloha_mobile_cabinet` is 14-dim with **no base odometry at all**. Think about what that means for five seconds:

1. Your carefully repaired 16-dim slicing logic is **dead code on the exact dataset you intend to score**. You fixed a code path that never executes in production.
2. Worse: on a *mobile manipulation* dataset with no base signal, your dwell detector is **structurally blind**. The cabinet task has extended driving phases where both arms are static in the base frame. Your FK runs in the base frame. Every driving segment looks identical to a frozen operator. Your `bimanual_dwell_fraction` will systematically flag *driving* — a required task phase — as hesitation, inflate entropy, and down-weight episodes for the crime of locomotion.
3. And your Phase 1 plan **cannot catch this**, because (a) your synthetic unit tests generate 16-dim trajectories that exercise the dead code path, and (b) `gym-aloha` is *static* ALOHA — it has no base, so your severity sweeps validate nothing about mobile confounders.

So your test plan validates the code path that never runs, on a simulator that can't reproduce the failure mode, for a dataset where the failure mode is guaranteed. That's not validation, that's theater.

**Mandatory fix:** For 14-dim mobile datasets, the dwell detector must either (a) abstain and emit `dwell_fraction = NaN` with an explicit "base-unobservable" flag, or (b) recover base motion from a proxy — optical flow on the base camera stream is the obvious candidate. Silently reporting a corrupted metric is worse than reporting nothing.

## Actual Bugs in the Code You Shipped Me

**1. The `v_base` length-matching logic crashes.** In the `np.diff` fallback branch, `len(v_l) = N-1 < len(v_base) = N`, so `np.pad(v_base, (len(v_l) - len(v_base), 0), 'edge')` gets a **negative pad width** and raises `ValueError`. You padded the longer array. You want truncation, and you got the direction backwards. This is exactly the class of off-by-one sloppiness that got the base slice backwards last time.

**2. Your `temporal_consistency` gate in `analyze_real_parquet` is vacuous.** You changed it to a t-statistic: `mean / (std / sqrt(n))`. With n in the hundreds of stable frames, SEM is tiny and this value is enormous for *any* nonzero mean. Threshold `> 2.0` passes essentially always. Meanwhile `analyze()` uses plain `mean/std > 1.5`. Two copies of the same logic, silently divergent, one of them a no-op. This is what happens when you duplicate 150 lines between `analyze()` and `analyze_real_parquet()` instead of extracting a shared `score_episode()`.

**3. Hardcoded 25-frame dwell threshold, dynamic `fs`.** You compute `fs = 1/mean(dt)` from timestamps, then compare dwell segments against `> 25` frames as if fs is always 50. Express the threshold in seconds: `(e - s) > 0.5 * fs`.

**4. No hysteresis on the activity mask.** `v_max > noise_floor` is a raw per-frame threshold. One noisy frame at 11 mm/s bisects a 1.2s dwell into two sub-threshold segments and your detector reports zero dwell. Morphologically close the idle mask (kernel ~3-5 frames) before segmenting, or your Phase 1 sensitivity numbers will be garbage at low severity.

**5. Verify the gripper normalization empirically, don't assume it.** Your `states[:, 6] > 0.9` contact-free gate assumes act-plus-plus [0,1] normalized grippers survived the LeRobot conversion. If they're raw radians, the mask is all-False, `tcp_drifts` is empty, and your calibration detector silently disables itself for the entire dataset with no warning. Add an assertion on the observed gripper range and log coverage: what fraction of frames pass `final_mask` per episode? If it's 2%, your drift statistics are noise.

**6. Dead memory bloat:** `episode_drifts[ep_idx] = (actions, states, final_mask)` accumulates full trajectories for every episode and is never read. Delete it.

## The Experiment Plan

The plan itself is now acceptable in shape — synthetic corruption tiers, confounder injection, explicit FPR on legitimate pauses, ROC gating before Phase 2, paired per-seed deltas with bootstrap CIs. Fine. That's what I asked for. Two additions:

- **Add a 14-dim mobile test tier:** synthesize episodes where base motion exists in ground truth but is *absent from the observation* (i.e., the actual production condition), and assert your detector abstains or proxies correctly. This is now the single most important test case and it's not in your plan.
- Report metric distributions on the *real* dataset before any flagging, so we can see whether your `-4.0` SPARC and `-10.0` LDLJ thresholds sit in the tail or in the bulk. If half the dataset trips the threshold, your thresholds are wrong, not the data.

## Verdict

**Conditional greenlight on Phase 1, Unit Test Tier only**, gated on:

1. The 14-dim base-unobservability handling (abstain or proxy) — blocking.
2. Fix the negative-pad crash — blocking.
3. Fix or delete the vacuous t-statistic gate, and deduplicate the two analyze paths — blocking, because I'm not reviewing two divergent implementations again.
4. Hysteresis + time-based dwell threshold + gripper range assertion — required before the severity sweep results mean anything.

Phase 2 remains contingent on the ROC gate, as agreed. And to be clear about the venture framing: the fact that your production dataset can't even represent the failure mode your flagship metric detects is not an implementation detail. It's a question about whether the product works on real data. Answer it in Phase 1 or don't bother with Phase 2.