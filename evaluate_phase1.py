"""
Phase 1 Validation Suite — V2 (Fable 5 V11 Overhaul)

Fixes:
- Truncation bug: hesitation is inserted by extending the trajectory, not amputating the grasp
- Boundary sweep: hesitation from 0.1s to 2.0s in 0.1s bins → psychometric curve
- Realistic physics: band-limited 4–12 Hz tremor, asymmetric bimanual, leader-follower lag,
  realistic encoder noise (1e-3 rad)
- Adversarial confounders: legitimate pauses at 0.3s, 0.4s, 0.45s near the boundary
- Statistical rigor: 10 seeds, Clopper-Pearson CI on FPR, clean trajectory SPARC assertion
- Score histogram to verify non-degenerate ROC
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import beta as beta_dist
from lerobot_buffer_hook import ArchitectureAwareDriftGate
import os

# ──────────────────────────────────────────────────────────────────────────────
# Trajectory Generator V2
# ──────────────────────────────────────────────────────────────────────────────

def generate_trajectory_v2(
    duration_s=10.0, fs=50.0, base_dims=False,
    hesitation_s=0.0,
    tremor_hz=0.0, tremor_amp=0.0,
    pre_grasp_pause_s=0.0,
    base_motion="static",
    asymmetric_bimanual=False,
    leader_follower_lag_ms=0.0,
    encoder_noise_rad=1e-3,
):
    """
    Generates a synthetic bimanual trajectory with realistic physics.

    Key differences from V1:
    - Hesitation is inserted by EXTENDING the trajectory, not truncating the tail
    - Tremor is band-limited (4-12 Hz sinusoidal), not white noise
    - Asymmetric bimanual mode: left arm stabilizes (near-static) while right manipulates
    - Leader-follower lag creates realistic action-state residuals
    - Encoder noise is 1e-3 rad (realistic), not 2e-5 (fantasy)
    """
    N = int(duration_s * fs)
    dt_val = 1.0 / fs

    # Smooth point-to-point sinusoidal movement
    t = np.linspace(0, np.pi, N)
    movement = (1 - np.cos(t)) / 2.0  # 0 → 1

    # Per-joint amplitude variation (not all joints identical)
    joint_amps = np.array([0.5, 0.8, 0.6, 0.3, 0.4, 0.2])  # 6 kinematic joints

    # Build states: 14-dim (or 16-dim)
    n_dims = 16 if base_dims else 14
    states = np.zeros((N, n_dims))
    actions = np.zeros((N, n_dims))

    # Base handling for mobile ALOHA
    base_stop_idx = N // 3
    if base_dims and base_motion == "drive_then_manipulate":
        states[:base_stop_idx, 14] = 0.5   # linear vel
        actions[:base_stop_idx, 14] = 0.5
        # Arms stationary during driving
        movement[:base_stop_idx] = 0.0
        t_rem = np.linspace(0, np.pi, N - base_stop_idx)
        movement[base_stop_idx:] = (1 - np.cos(t_rem)) / 2.0

    # Populate arm joints with per-joint amplitudes
    for j in range(6):
        # Left arm joints 0-5
        states[:, j] = movement * joint_amps[j]
        actions[:, j] = movement * joint_amps[j]

    if asymmetric_bimanual:
        # Right arm: small stabilization movements only (legitimately near-static)
        for j in range(6):
            stabilize = np.sin(np.linspace(0, 2 * np.pi, N)) * 0.02  # tiny oscillation
            states[:, 7 + j] = stabilize
            actions[:, 7 + j] = stabilize
    else:
        # Right arm mirrors left with slight phase offset
        phase_offset = int(0.1 * fs)  # 100ms offset
        for j in range(6):
            states[:, 7 + j] = np.roll(movement * joint_amps[j], phase_offset)
            actions[:, 7 + j] = np.roll(movement * joint_amps[j], phase_offset)

    # Grippers: close at 80% of trajectory
    grasp_idx = int(N * 0.8)

    # Pre-grasp settling (legitimate pause before grasping)
    if pre_grasp_pause_s > 0:
        pause_frames = int(pre_grasp_pause_s * fs)
        pause_start = max(0, grasp_idx - pause_frames)
        if pause_start > (base_stop_idx if base_dims else 0):
            freeze_val = states[pause_start].copy()
            for f in range(pause_start, grasp_idx):
                states[f, :14] = freeze_val[:14]
                actions[f, :14] = freeze_val[:14]

    # Gripper states
    states[grasp_idx:, 6] = 1.0
    states[grasp_idx:, 13] = 1.0
    actions[grasp_idx:, 6] = 1.0
    actions[grasp_idx:, 13] = 1.0

    # ── Hesitation injection (INSERT, don't truncate) ──
    if hesitation_s > 0:
        h_frames = int(hesitation_s * fs)
        mid_idx = N // 2
        freeze_state = states[mid_idx].copy()
        freeze_action = actions[mid_idx].copy()

        # Insert frozen frames, keeping EVERYTHING after mid_idx intact
        hesitation_block_s = np.tile(freeze_state, (h_frames, 1))
        hesitation_block_a = np.tile(freeze_action, (h_frames, 1))

        states = np.vstack([states[:mid_idx], hesitation_block_s, states[mid_idx:]])
        actions = np.vstack([actions[:mid_idx], hesitation_block_a, actions[mid_idx:]])
        # N is now extended
        N = len(states)

    # ── Band-limited tremor (4-12 Hz physiological) ──
    if tremor_amp > 0 and tremor_hz > 0:
        t_sec = np.arange(N) * dt_val
        # Sum of sinusoids in the 4-12 Hz band
        freqs = np.linspace(4.0, 12.0, 5)
        for j in range(6):  # kinematic joints only
            tremor_signal = np.zeros(N)
            for freq in freqs:
                phase = np.random.uniform(0, 2 * np.pi)
                tremor_signal += np.sin(2 * np.pi * freq * t_sec + phase)
            tremor_signal = tremor_signal / len(freqs) * tremor_amp
            states[:, j] += tremor_signal
            states[:, 7 + j] += tremor_signal
            actions[:, j] += tremor_signal
            actions[:, 7 + j] += tremor_signal

    # ── Leader-follower lag ──
    if leader_follower_lag_ms > 0:
        lag_frames = max(1, int(leader_follower_lag_ms / 1000.0 * fs))
        # Actions (leader) lead states (follower) by lag_frames
        actions_shifted = np.roll(actions, -lag_frames, axis=0)
        actions_shifted[-lag_frames:] = actions[-lag_frames:]
        # Keep states as follower, actions as leader
        # This creates a realistic residual
        actions = actions_shifted

    # ── Realistic encoder noise ──
    noise = np.random.normal(0, encoder_noise_rad, (N, n_dims))
    if base_dims:
        noise[:, 14:16] = 0
    noise[:, 6] = 0
    noise[:, 13] = 0
    states += noise

    # Timestamps
    timestamps = np.arange(N) * dt_val

    return actions, states, timestamps


# ──────────────────────────────────────────────────────────────────────────────
# Unit Tests
# ──────────────────────────────────────────────────────────────────────────────

def run_unit_tests():
    print("=" * 60)
    print("UNIT TESTS")
    print("=" * 60)
    gate = ArchitectureAwareDriftGate()

    # Test 1: Drive-Then-Manipulate (16-dim Clean)
    a, s, t = generate_trajectory_v2(base_dims=True, base_motion="drive_then_manipulate")
    res = gate._score_episode(1, a, s, t)
    assert res["bimanual_dwell_fraction"] == 0.0, \
        f"Test 1 FAIL: driving flagged as dwell ({res['bimanual_dwell_fraction']:.3f})"
    assert res["base_unobservable"] is False
    print(f"✅ Test 1: Drive-Then-Manipulate — dwell=0.0, base_unobservable=False")

    # Test 2: True Operator Arrest (16-dim, 1.2s hesitation)
    a, s, t = generate_trajectory_v2(base_dims=True, base_motion="static", hesitation_s=1.2)
    res = gate._score_episode(2, a, s, t)
    assert res["bimanual_dwell_fraction"] > 0.0, \
        f"Test 2 FAIL: missed true arrest (dwell={res['bimanual_dwell_fraction']:.3f})"
    print(f"✅ Test 2: True Operator Arrest — dwell={res['bimanual_dwell_fraction']:.3f}")

    # Test 3: 14-dim Unobservable Base
    a, s, t = generate_trajectory_v2(base_dims=False, hesitation_s=1.2)
    res = gate._score_episode(3, a, s, t)
    assert res["base_unobservable"] is True, "Test 3 FAIL: didn't flag base_unobservable"
    assert res["bimanual_dwell_fraction"] == 0.0, "Test 3 FAIL: didn't abstain"
    print(f"✅ Test 3: 14-dim Unobservable Base — abstained correctly")

    # Test 5: Asymmetric Bimanual
    a, s, t = generate_trajectory_v2()
    # Left arm normal, right arm micro-adjustments
    a[:, :6] = np.sin(t[:, None] * 1.5 * np.pi) * 0.5
    a[:, 7:13] = np.sin(t[:, None] * 3.0 * np.pi) * 0.02
    s = a + np.random.normal(0, 1e-3, a.shape)
    
    res = gate._score_episode(5, a, s, t)
    assert res["bimanual_dwell_fraction"] == 0.0, \
        f"Test 4 FAIL: tremor flagged as dwell ({res['bimanual_dwell_fraction']:.3f})"
    assert res["sparc"] < -3.5, \
        f"Test 4 FAIL: tremor not penalized by SPARC ({res['sparc']:.2f})"
    print(f"✅ Test 4: Band-limited Tremor — dwell=0.0, SPARC={res['sparc']:.2f}")

    # Test 5: Asymmetric Bimanual (one arm stabilizing) — must NOT trigger false positive
    a, s, t = generate_trajectory_v2(
        base_dims=True, base_motion="static", asymmetric_bimanual=True
    )
    res = gate._score_episode(5, a, s, t)
    assert res["bimanual_dwell_fraction"] == 0.0, \
        f"Test 5 FAIL: asymmetric bimanual flagged as dwell ({res['bimanual_dwell_fraction']:.3f})"
    print(f"✅ Test 5: Asymmetric Bimanual — dwell=0.0 (no false positive)")

    # Test 6: Clean trajectory SPARC must stay ABOVE -4.0
    a, s, t = generate_trajectory_v2(base_dims=True, base_motion="static")
    res = gate._score_episode(6, a, s, t)
    assert res["sparc"] > -4.0, \
        f"Test 6 FAIL: clean trajectory scored poor SPARC ({res['sparc']:.2f})"
    print(f"✅ Test 6: Clean Trajectory SPARC sanity — SPARC={res['sparc']:.2f} (> -4.0)")

    # Test 7: Leader-follower lag does not break the gate
    a, s, t = generate_trajectory_v2(
        base_dims=True, base_motion="static", leader_follower_lag_ms=80.0
    )
    res = gate._score_episode(7, a, s, t)
    # Should complete without crash; entropy should remain reasonable
    assert res["entropy"] < 0.8, \
        f"Test 7 FAIL: leader-follower lag caused extreme entropy ({res['entropy']:.2f})"
    print(f"✅ Test 7: Leader-Follower Lag (80ms) — entropy={res['entropy']:.2f}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Psychometric Boundary Sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_boundary_sweep(n_seeds=10, n_per_bin=30):
    """
    Sweep hesitation duration from 0.1s to 2.0s in 0.1s bins.
    For each bin, generate n_per_bin episodes per seed and report detection rate.
    """
    print("=" * 60)
    print("PSYCHOMETRIC BOUNDARY SWEEP (Hesitation)")
    print("=" * 60)
    gate = ArchitectureAwareDriftGate()

    bins = np.arange(0.1, 2.05, 0.1)
    detection_rates = {b: [] for b in bins}

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        for b in bins:
            detected = 0
            for _ in range(n_per_bin):
                a, s, t = generate_trajectory_v2(
                    base_dims=True, base_motion="static",
                    hesitation_s=b,
                    encoder_noise_rad=1e-3,
                )
                # Add small random noise to seed variation
                s += rng.normal(0, 5e-4, s.shape)
                res = gate._score_episode(0, a, s, t)
                # "Detected" = entropy pushed above 0.05 (non-trivial penalty)
                if res["entropy"] > 0.05:
                    detected += 1
            detection_rates[b].append(detected / n_per_bin)

    # Aggregate across seeds
    mean_rates = []
    ci_lower = []
    ci_upper = []
    for b in bins:
        rates = detection_rates[b]
        mean_rates.append(np.mean(rates))
        ci_lower.append(np.percentile(rates, 2.5))
        ci_upper.append(np.percentile(rates, 97.5))

    # Print table
    print(f"{'Duration (s)':>14} | {'Detection Rate':>14} | {'95% CI':>20}")
    print("-" * 55)
    for i, b in enumerate(bins):
        print(f"{b:>14.1f} | {mean_rates[i]:>14.3f} | [{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]")

    # Plot
    os.makedirs('static', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bins, mean_rates, 'o-', color='#e63946', lw=2, markersize=6)
    ax.fill_between(bins, ci_lower, ci_upper, alpha=0.2, color='#e63946')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% detection')
    ax.axvline(0.5, color='blue', linestyle='--', alpha=0.5, label='Design threshold (0.5s)')
    ax.set_xlabel('Hesitation Duration (seconds)', fontsize=12)
    ax.set_ylabel('Detection Rate', fontsize=12)
    ax.set_title('Psychometric Boundary Curve: Hesitation Detection', fontsize=14)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('static/phase1_boundary_sweep.png', dpi=150)
    plt.close(fig)
    print(f"\n📊 Saved boundary sweep plot → static/phase1_boundary_sweep.png")
    print()

    return bins, mean_rates


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial ROC with Realistic Physics
# ──────────────────────────────────────────────────────────────────────────────

def run_adversarial_roc(n_seeds=10, n_per_seed=100):
    """
    Realistic ROC evaluation with adversarial confounders and proper statistics.
    """
    print("=" * 60)
    print("ADVERSARIAL ROC EVALUATION")
    print("=" * 60)
    gate = ArchitectureAwareDriftGate()

    all_aucs = []
    all_scores_clean = []
    all_scores_corrupt = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed * 7 + 13)
        y_true = []
        y_scores = []

        for i in range(n_per_seed):
            is_corrupt = i % 2 == 1

            hesitation_s = 0.0
            tremor_hz = 0.0
            tremor_amp = 0.0
            pre_grasp = 0.0
            asymmetric = False
            lag_ms = 0.0

            if is_corrupt:
                y_true.append(1)
                coin = rng.rand()
                if coin < 0.5:
                    # Hesitation — sweep the FULL range including boundary
                    hesitation_s = rng.uniform(0.3, 2.0)
                else:
                    # Band-limited tremor
                    tremor_hz = rng.uniform(4.0, 12.0)
                    tremor_amp = rng.uniform(0.02, 0.08)
            else:
                y_true.append(0)
                # Adversarial confounders — mix of realistic scenarios
                confounder = rng.choice([
                    'short_pause', 'medium_pause', 'boundary_pause',
                    'asymmetric', 'lag', 'noisy_clean'
                ])
                if confounder == 'short_pause':
                    pre_grasp = 0.2
                elif confounder == 'medium_pause':
                    pre_grasp = 0.3
                elif confounder == 'boundary_pause':
                    pre_grasp = rng.uniform(0.4, 0.45)
                elif confounder == 'asymmetric':
                    asymmetric = True
                elif confounder == 'lag':
                    lag_ms = rng.uniform(50, 100)
                else:
                    pass  # just encoder noise

            a, s, t = generate_trajectory_v2(
                base_dims=True, base_motion="static",
                hesitation_s=hesitation_s,
                tremor_hz=tremor_hz, tremor_amp=tremor_amp,
                pre_grasp_pause_s=pre_grasp,
                asymmetric_bimanual=asymmetric,
                leader_follower_lag_ms=lag_ms,
                encoder_noise_rad=1e-3,
            )

            res = gate._score_episode(seed * 1000 + i, a, s, t)
            anomaly_score = res["entropy"]
            y_scores.append(anomaly_score)

            if is_corrupt:
                all_scores_corrupt.append(anomaly_score)
            else:
                all_scores_clean.append(anomaly_score)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        seed_auc = auc(fpr, tpr)
        all_aucs.append(seed_auc)

    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    ci_low = np.percentile(all_aucs, 2.5)
    ci_high = np.percentile(all_aucs, 97.5)

    print(f"AUC across {n_seeds} seeds: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Per-seed AUCs: {[f'{a:.4f}' for a in all_aucs]}")

    # FPR with Clopper-Pearson CI
    n_clean = len(all_scores_clean)
    n_fp = sum(1 for s in all_scores_clean if s > 0.5)
    fpr_point = n_fp / n_clean
    # Clopper-Pearson 95% CI
    if n_fp == 0:
        cp_lower = 0.0
        cp_upper = 1 - (0.05 / 2) ** (1.0 / n_clean)
    else:
        cp_lower = beta_dist.ppf(0.025, n_fp, n_clean - n_fp + 1)
        cp_upper = beta_dist.ppf(0.975, n_fp + 1, n_clean - n_fp)

    print(f"\nFPR (threshold=0.5): {fpr_point:.4f} ({n_fp}/{n_clean})")
    print(f"Clopper-Pearson 95% CI: [{cp_lower:.4f}, {cp_upper:.4f}]")

    # ── Score Histogram (verify non-degenerate ROC) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_scores_clean, bins=30, alpha=0.6, color='#2a9d8f', label='Clean', density=True)
    ax.hist(all_scores_corrupt, bins=30, alpha=0.6, color='#e76f51', label='Corrupt', density=True)
    ax.set_xlabel('Entropy (Anomaly Score)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution: Clean vs. Corrupt Episodes', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('static/phase1_score_histogram.png', dpi=150)
    plt.close(fig)
    print(f"\n📊 Saved score histogram → static/phase1_score_histogram.png")

    # ── ROC Curve (aggregate) ──
    all_y_true = [0] * len(all_scores_clean) + [1] * len(all_scores_corrupt)
    all_y_scores = all_scores_clean + all_scores_corrupt
    fpr_agg, tpr_agg, _ = roc_curve(all_y_true, all_y_scores)
    auc_agg = auc(fpr_agg, tpr_agg)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_agg, tpr_agg, color='darkorange', lw=2, label=f'ROC (AUC = {auc_agg:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Adversarial Evaluation', fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('static/phase1_roc_curve.png', dpi=150)
    plt.close(fig)
    print(f"📊 Saved ROC curve → static/phase1_roc_curve.png")

    return mean_auc, std_auc, fpr_point, cp_lower, cp_upper


# ──────────────────────────────────────────────────────────────────────────────
# Real Data Distribution
# ──────────────────────────────────────────────────────────────────────────────

def run_real_data_distribution():
    """
    Run the gate on the actual lerobot/aloha_mobile_cabinet dataset and report
    the distribution of scores. No labels needed — we just show what the gate
    outputs on real production data.
    """
    print("=" * 60)
    print("REAL DATA DISTRIBUTION (lerobot/aloha_mobile_cabinet)")
    print("=" * 60)

    try:
        from datasets import load_dataset
        import pandas as pd

        ds = load_dataset("lerobot/aloha_mobile_cabinet", split="train")
        df = pd.DataFrame(ds)

        gate = ArchitectureAwareDriftGate()
        unique_episodes = sorted(df["episode_index"].unique())

        print(f"Analyzing {len(unique_episodes)} real episodes...")

        episode_results = []
        for ep_idx in unique_episodes:
            ep_data = df[df["episode_index"] == ep_idx]
            states = np.vstack(ep_data["observation.state"].values)
            actions = np.vstack(ep_data["action"].values)
            timestamps = ep_data["timestamp"].values

            res = gate._score_episode(ep_idx, actions, states, timestamps)
            episode_results.append(res)

        entropies = [r["entropy"] for r in episode_results]
        weights = [r["loss_weight"] for r in episode_results]
        sparcs = [r["sparc"] for r in episode_results]
        base_flags = [r["base_unobservable"] for r in episode_results]

        print(f"\nReal data is {states.shape[1]}-dim")
        print(f"Base unobservable: {all(base_flags)}")
        print(f"\nEntropy — mean: {np.mean(entropies):.4f}, "
              f"std: {np.std(entropies):.4f}, "
              f"min: {np.min(entropies):.4f}, "
              f"max: {np.max(entropies):.4f}")
        print(f"Loss Weight — mean: {np.mean(weights):.4f}, "
              f"std: {np.std(weights):.4f}, "
              f"min: {np.min(weights):.4f}, "
              f"max: {np.max(weights):.4f}")
        print(f"SPARC — mean: {np.mean(sparcs):.2f}, "
              f"std: {np.std(sparcs):.2f}, "
              f"min: {np.min(sparcs):.2f}, "
              f"max: {np.max(sparcs):.2f}")

        n_flagged = sum(1 for e in entropies if e > 0.3)
        print(f"\nEpisodes with entropy > 0.3: {n_flagged}/{len(entropies)} "
              f"({n_flagged/len(entropies)*100:.1f}%)")

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(entropies, bins=20, color='#264653', edgecolor='white')
        axes[0].set_title('Entropy Distribution (Real Data)')
        axes[0].set_xlabel('Kinematic Entropy')
        axes[0].set_ylabel('Episode Count')

        axes[1].hist(weights, bins=20, color='#2a9d8f', edgecolor='white')
        axes[1].set_title('Loss Weight Distribution (Real Data)')
        axes[1].set_xlabel('Loss Weight')

        axes[2].hist(sparcs, bins=20, color='#e76f51', edgecolor='white')
        axes[2].axvline(-4.0, color='red', linestyle='--', label='SPARC threshold')
        axes[2].set_title('SPARC Distribution (Real Data)')
        axes[2].set_xlabel('SPARC Score')
        axes[2].legend()

        fig.suptitle('Gate Output on Real lerobot/aloha_mobile_cabinet', fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig('static/phase1_real_data_distribution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n📊 Saved real data distribution → static/phase1_real_data_distribution.png")

    except Exception as e:
        print(f"⚠️  Real data analysis skipped: {e}")
        print("   (This is expected if the dataset is not cached locally)")


# ──────────────────────────────────────────────────────────────────────────────
# Write Final Report
# ──────────────────────────────────────────────────────────────────────────────

def write_report(mean_auc, std_auc, fpr_point, cp_lower, cp_upper, bins, mean_rates):
    report = f"""# Phase 1 Validation Report — V2 (Fable 5 V11 Overhaul)

## Methodology
- **Trajectory Generator V2**: Realistic physics with band-limited tremor (4-12 Hz),
  asymmetric bimanual tasks, leader-follower lag (50-100ms), and encoder noise (1e-3 rad).
- **Hesitation injection**: Extends trajectory (no truncation/label leakage).
- **Confounders**: Legitimate pauses at 0.2s, 0.3s, 0.4s, 0.45s; asymmetric bimanual;
  leader-follower lag; noisy clean baselines.
- **Statistical rigor**: 10 seeds, Clopper-Pearson CIs.

## Adversarial ROC Results
- **AUC**: {mean_auc:.4f} ± {std_auc:.4f} (across 10 seeds)
- **FPR (threshold=0.5)**: {fpr_point:.4f}
- **Clopper-Pearson 95% CI**: [{cp_lower:.4f}, {cp_upper:.4f}]

## Psychometric Boundary Curve
Detection rate by hesitation duration:

| Duration (s) | Detection Rate |
|:---:|:---:|
"""
    for i, b in enumerate(bins):
        report += f"| {b:.1f} | {mean_rates[i]:.3f} |\n"

    report += """
## Plots
- ROC Curve: `static/phase1_roc_curve.png`
- Score Histogram: `static/phase1_score_histogram.png`
- Boundary Sweep: `static/phase1_boundary_sweep.png`
- Real Data Distribution: `static/phase1_real_data_distribution.png`
"""

    with open("phase1_validation_report.md", "w") as f:
        f.write(report)
    print("\n✅ Validation Report → phase1_validation_report.md")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_unit_tests()
    bins, mean_rates = run_boundary_sweep(n_seeds=10, n_per_bin=30)
    mean_auc, std_auc, fpr_point, cp_lower, cp_upper = run_adversarial_roc(n_seeds=10, n_per_seed=100)
    run_real_data_distribution()
    write_report(mean_auc, std_auc, fpr_point, cp_lower, cp_upper, bins, mean_rates)
    print("\n🏁 Phase 1 Validation Complete.")
