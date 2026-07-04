import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import urllib.request
import json
from scipy.signal import savgol_filter
from scipy.ndimage import binary_closing
from kinematics import BimanualForwardKinematics


class ArchitectureAwareDriftGate:
    """
    V12 Physics-Informed Data Observability & Weighting Layer.

    Changes vs V4.0 (per V12 review):
      1. SPARC is computed per *movement unit*: episodes are segmented via
         velocity-threshold segmentation (adaptive threshold with a hard noise
         floor), short blips are rejected with a minimum-duration filter, and
         the episode-level SPARC is the MEDIAN over movement-unit SPARC scores.
      2. The gate is SELF-CALIBRATING per dataset. All segment-level SPARC
         scores across the dataset are pooled; the baseline is (median, MAD).
         Episode entropy is mapped through a smooth robust z-score:

             z       = (dataset_median_sparc - episode_sparc) / (1.4826 * MAD)
             entropy = sigmoid(k * z - b)

         Episodes are flagged when z exceeds ~2.75 MAD (entropy crosses 0.5).
         Nominal priors are only used as a fallback before calibration or when
         the dataset is degenerate (too few segments / near-zero MAD).
    """

    def __init__(self, episodes=None, slack_webhook=None):
        self.episodes = episodes
        self.slack_webhook = slack_webhook

        # --- Cartesian drift calibration (unchanged, robust per-dataset) ---
        self.nominal_median_tcp_mm = 12.0
        self.nominal_mad_tcp_mm = 3.0
        self.z_threshold = 3.0

        # --- V12 movement-unit segmentation parameters ---
        self.noise_floor = 0.05            # m/s, absolute TCP speed noise floor
        self.segment_speed_frac = 0.10     # adaptive threshold: fraction of p95 speed
        self.min_segment_duration_s = 0.25 # minimum-duration filter for movement units
        self.min_segment_samples = 10      # absolute sample floor regardless of fs
        self.gap_closing_s = 0.10          # close brief sub-threshold dips inside a unit

        # --- V12 self-calibrating entropy mapping ---
        self.sigmoid_k = 1.2
        self.sigmoid_b = 3.3
        self.z_flag_mads = self.sigmoid_b / self.sigmoid_k  # ~2.75 MAD flag point
        self.mad_scale = 1.4826            # consistency constant (MAD -> sigma)
        self.min_calibration_segments = 8  # below this, fall back to nominal prior
        self.sparc_mad_floor = 0.05        # absolute floor to avoid z blow-up

        # Nominal SPARC prior (fallback only, used until dataset calibration runs)
        self.nominal_sparc_median = -1.6
        self.nominal_sparc_mad = 0.35

        # Dwell penalty (hesitation) contribution to entropy
        self.dwell_entropy_gain = 2.0
        self.dwell_flag_fraction = 0.2

        # Reversal rate threshold in reversals per 100 frames
        self.reversal_threshold_rate = 20.0

        # Dataset calibration state (None until _calibrate() has run)
        self._calib = None

        self.fk_solver = BimanualForwardKinematics()

    def compute_direction_reversal_rate(self, positions):
        """
        Detects Diffusion Policy stalling / hesitation (zero-crossings).
        Filters out encoder quantization noise using a velocity deadband of 0.01 rad/s.
        Returns reversals per 100 frames (episode-length-independent rate).
        """
        velocity = np.diff(positions, axis=0)
        
        # Apply velocity deadband to filter out encoder quantization jitter
        active_mask = np.abs(velocity) > 0.002

        # Extract signs
        signs = np.sign(velocity)
        
        # Record sign changes tracking the last active direction
        sign_changes = np.zeros_like(velocity, dtype=bool)
        last_active_sign = np.zeros(velocity.shape[1])
        
        for i in range(velocity.shape[0]):
            for j in range(velocity.shape[1]):
                if active_mask[i, j]:
                    current_sign = signs[i, j]
                    if last_active_sign[j] != 0 and last_active_sign[j] != current_sign:
                        sign_changes[i, j] = True
                    last_active_sign[j] = current_sign
                        
        # Exclude gripper joints from reversal calculations (indexes 6 and 13)
        kin_joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        reversals_per_joint = np.sum(sign_changes[:, kin_joints], axis=0)
        max_reversals = np.max(reversals_per_joint) if len(reversals_per_joint) > 0 else 0
        n_frames = len(positions)
        return float(max_reversals) / n_frames * 100.0

    # ------------------------------------------------------------------ #
    # Cartesian drift (unchanged logic, hardened)                         #
    # ------------------------------------------------------------------ #

    def compute_cartesian_drift_series(self, leader_pos, follower_pos, stable_mask):
        if not np.any(stable_mask):
            return np.array([]), np.array([])

        stable_leader = leader_pos[stable_mask]
        stable_follower = follower_pos[stable_mask]

        d = stable_leader - stable_follower
        d = (d + np.pi) % (2 * np.pi) - np.pi
        mapped_leader = stable_follower + d

        (l_l_pos, l_l_R), (l_r_pos, l_r_R) = self.fk_solver.solve_bimanual_fk(mapped_leader)
        (f_l_pos, f_l_R), (f_r_pos, f_r_R) = self.fk_solver.solve_bimanual_fk(stable_follower)

        # Return signed vector differences for proper bias testing later
        drift_vec_l = l_l_pos - f_l_pos
        drift_vec_r = l_r_pos - f_r_pos

        return drift_vec_l, drift_vec_r, rot_drifts

    def compute_leader_follower_drift(self, leader_pos, follower_pos, dt):
        v_follower = np.abs(np.diff(follower_pos, axis=0)) / dt[1:][:, np.newaxis]
        v_follower = np.vstack([np.zeros((1, follower_pos.shape[1])), v_follower])

        v_leader = np.abs(np.diff(leader_pos, axis=0)) / dt[1:][:, np.newaxis]
        v_leader = np.vstack([np.zeros((1, leader_pos.shape[1])), v_leader])

        v_max = np.maximum(v_leader, v_follower)

        kinematic_joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        stable_mask = np.all(v_max[:, kinematic_joints] < 0.15, axis=1)

        return stable_mask

    # ------------------------------------------------------------------ #
    # Raw smoothness metrics                                              #
    # ------------------------------------------------------------------ #

    def compute_ldlj_raw(self, speed_profile, fs):
        speed_profile = np.asarray(speed_profile, dtype=np.float64)
        if len(speed_profile) < 10:
            return None

        speed_peak = np.max(np.abs(speed_profile))
        if speed_peak < self.noise_floor:
            return None

        dt = 1.0 / fs
        duration = len(speed_profile) * dt

        window = min(15, len(speed_profile) if len(speed_profile) % 2 != 0 else len(speed_profile) - 1)
        if window < 5:
            jerk = np.diff(speed_profile, 2) / (dt ** 2)
        else:
            jerk = savgol_filter(speed_profile, window_length=window, polyorder=3, deriv=2, delta=dt)

        scale = (duration ** 3) / (speed_peak ** 2)
        dj = scale * np.sum(jerk ** 2) * dt
        if dj <= 0 or not np.isfinite(dj):
            return None
        return float(-np.log(dj))

    def compute_sparc_raw(self, speed_profile, fs, padlevel=4, fc=10.0, amp_th=0.05):
        speed_profile = np.asarray(speed_profile, dtype=np.float64)
        if len(speed_profile) < 10:
            return None

        nfft = int(pow(2, np.ceil(np.log2(len(speed_profile))) + padlevel))
        f = np.arange(0, fs, fs / nfft)
        Mf = np.abs(np.fft.fft(speed_profile, nfft))

        max_Mf = np.max(Mf)
        if max_Mf < 1e-6:
            return None
        Mf = Mf / max_Mf

        fc_inx = ((f <= fc) * 1).nonzero()
        f_sel = f[fc_inx]
        Mf_sel = Mf[fc_inx]

        inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
        if len(inx) == 0:
            return None

        fc_inx = range(inx[0], inx[-1] + 1)
        f_sel = f_sel[fc_inx]
        Mf_sel = Mf_sel[fc_inx]

        if len(f_sel) < 2 or (f_sel[-1] - f_sel[0]) <= 0:
            return None

        new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                               pow(np.diff(Mf_sel), 2)))
        if not np.isfinite(new_sal):
            return None
        return float(new_sal)

    # ------------------------------------------------------------------ #
    # V12: movement-unit segmentation                                     #
    # ------------------------------------------------------------------ #

    def segment_movement_units(self, speed_profile, fs):
        speed_profile = np.asarray(speed_profile, dtype=np.float64)
        n = len(speed_profile)
        if n == 0:
            return []

        p95 = np.percentile(speed_profile, 95)
        thr = max(self.noise_floor, self.segment_speed_frac * p95)

        active = speed_profile > thr
        if not np.any(active):
            return []

        gap_samples = max(1, int(round(self.gap_closing_s * fs)))
        active = binary_closing(active, structure=np.ones(gap_samples * 2 + 1))

        diffs = np.diff(active.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if active[0]:
            starts = np.insert(starts, 0, 0)
        if active[-1]:
            ends = np.append(ends, n)

        min_samples = max(self.min_segment_samples, int(round(self.min_segment_duration_s * fs)))

        segments = []
        for s, e in zip(starts, ends):
            if (e - s) >= min_samples:
                segments.append((int(s), int(e)))
        return segments

    def segment_metric_scores(self, metric_fn, speed_profile, fs, segments=None):
        if segments is None:
            segments = self.segment_movement_units(speed_profile, fs)
        scores = []
        for s, e in segments:
            score = metric_fn(speed_profile[s:e], fs)
            if score is not None and np.isfinite(score):
                scores.append(float(score))
        return scores

    # ------------------------------------------------------------------ #
    # V12: self-calibrating robust entropy mapping                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    def _calibrate(self, episode_data):
        all_segments = []
        for ep in episode_data:
            all_segments.extend(ep.get("sparc_segments", []))

        if len(all_segments) >= self.min_calibration_segments:
            arr = np.asarray(all_segments, dtype=np.float64)
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median)))
            mad = max(mad, self.sparc_mad_floor)
            source = "dataset"
        else:
            median = self.nominal_sparc_median
            mad = max(self.nominal_sparc_mad, self.sparc_mad_floor)
            source = "nominal_fallback"

        self._calib = {
            "sparc_median": median,
            "sparc_mad": mad,
            "n_segments": len(all_segments),
            "source": source,
        }
        return self._calib

    def _sparc_entropy(self, sparc_val):
        if self._calib is not None:
            median = self._calib["sparc_median"]
            mad = self._calib["sparc_mad"]
        else:
            median = self.nominal_sparc_median
            mad = max(self.nominal_sparc_mad, self.sparc_mad_floor)

        if sparc_val is None:
            return 0.0, 0.0

        z = (median - sparc_val) / (self.mad_scale * mad)
        entropy = float(self._sigmoid(self.sigmoid_k * z - self.sigmoid_b))
        return entropy, float(z)

    def _apply_calibration(self, ep):
        sparc_entropy, robust_z = self._sparc_entropy(ep["sparc"] if ep["n_movement_units"] > 0 else None)

        dwell_entropy = np.clip(ep["bimanual_dwell_fraction"] * self.dwell_entropy_gain, 0.0, 1.0)
        entropy = float(np.clip(sparc_entropy + dwell_entropy, 0.0, 1.0))

        ep["sparc_robust_z"] = robust_z
        ep["sparc_entropy"] = float(sparc_entropy)
        ep["dwell_entropy"] = float(dwell_entropy)
        ep["entropy"] = entropy
        ep["loss_weight"] = float(np.clip(1.0 - entropy, 0.1, 1.0))
        ep["sparc_flagged"] = bool(ep["n_movement_units"] > 0 and robust_z > self.z_flag_mads)
        return ep

    def _finalize_episodes(self, episode_data):
        self._calibrate(episode_data)
        
        # Calculate dataset-wide relative Cartesian drift baselines
        tcp_offsets = [ep["tcp_drift_mm"] for ep in episode_data if ep["tcp_drift_mm"] > 0]
        if len(tcp_offsets) >= 5:
            median_tcp = float(np.median(tcp_offsets))
            mad_tcp = float(np.median(np.abs(np.array(tcp_offsets) - median_tcp)))
            mad_tcp = max(mad_tcp, 0.5)
        else:
            median_tcp = self.nominal_median_tcp_mm
            mad_tcp = self.nominal_mad_tcp_mm
            
        for ep in episode_data:
            self._apply_calibration(ep)
            
            # 1. Cartesian drift relative outlier check (Gate A/B/C)
            z_score = (ep["tcp_drift_mm"] - median_tcp) / (1.4826 * mad_tcp)
            is_large_offset = ep["tcp_drift_mm"] > 40.0 or ep["rot_drift"] > 8.0
            is_steady_bias = ep["temporal_consistency"] > 2.0
            is_drift = bool(z_score > 3.0 and is_large_offset and is_steady_bias)
            
            # 2. Operator Reversal rate check (Gate D)
            is_reversal = bool(ep["reversal_rate"] > self.reversal_threshold_rate)
            
            ep["drift_flagged"] = is_drift
            ep["reversal_flagged"] = is_reversal
            
            # If drift or reversal are triggered, force high entropy and low loss weight
            if is_drift or is_reversal:
                ep["entropy"] = 1.0
                ep["loss_weight"] = 0.1
                
        return episode_data

    # ------------------------------------------------------------------ #
    # Alerting                                                            #
    # ------------------------------------------------------------------ #

    def send_slack_alert(self, message):
        if not self.slack_webhook:
            return "NO_WEBHOOK_CONFIGURED"
        try:
            payload = {
                "text": "🛠️ *Robotics Data Repair Alert*",
                "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": message}}]
            }
            req = urllib.request.Request(
                self.slack_webhook, data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return f"SUCCESS ({response.status})"
        except Exception as e:
            return f"ERROR: {str(e)}"

    # ------------------------------------------------------------------ #
    # Per-episode scoring                                                 #
    # ------------------------------------------------------------------ #

    def _score_episode(self, ep_idx, actions_raw, states_raw, timestamps=None):
        base_states = None
        if states_raw.shape[1] == 16:
            base_states = states_raw[:, 14:16]
            states = states_raw[:, 0:14]
        else:
            states = states_raw

        if actions_raw.shape[1] == 16:
            actions = actions_raw[:, 0:14]
        else:
            actions = actions_raw

        base_unobservable = False
        if base_states is None and states.shape[1] == 14:
            base_unobservable = True

        if timestamps is not None:
            dt = np.diff(timestamps)
            dt = np.insert(dt, 0, 0.02)
            dt = np.clip(dt, 0.001, 1.0)
            fs = 1.0 / np.mean(dt) if len(dt) > 0 else 50.0
            print(f"DEBUG TIMESTAMPS: ep={ep_idx}, ts_min={timestamps.min():.3f}, ts_max={timestamps.max():.3f}, dt_mean={np.mean(dt):.6f}, fs={fs:.3f}")
        else:
            dt = np.ones(len(states)) * 0.02
            fs = 50.0

        stable_mask = self.compute_leader_follower_drift(actions, states, dt)

        if base_unobservable:
            contact_free_mask = np.ones(len(states), dtype=bool)
        else:
            gripper_open = (states[:, 6] > 0.9) & (states[:, 13] > 0.9)
            free_space = np.arange(len(states)) < 100
            contact_free_mask = gripper_open | free_space

        final_mask = stable_mask & contact_free_mask
        drift_vec_l, drift_vec_r, rot_drifts = self.compute_cartesian_drift_series(actions, states, final_mask)
        
        # Calculate scalar magnitudes for overall drift bounds
        if len(drift_vec_l) > 0:
            tcp_drifts_l = np.linalg.norm(drift_vec_l, axis=1)
            tcp_drifts_r = np.linalg.norm(drift_vec_r, axis=1)
            tcp_drifts = np.maximum(tcp_drifts_l, tcp_drifts_r)
        else:
            tcp_drifts = np.array([])

        v_l_joint = np.abs(np.diff(states[:, :6], axis=0)) / dt[1:, np.newaxis]
        v_l_joint = np.vstack([np.zeros((1, 6)), v_l_joint])
        v_r_joint = np.abs(np.diff(states[:, 7:13], axis=0)) / dt[1:, np.newaxis]
        v_r_joint = np.vstack([np.zeros((1, 6)), v_r_joint])

        # approximate end-effector speed from joint velocities
        v_l_tcp = np.max(v_l_joint, axis=1) * 0.6
        v_r_tcp = np.max(v_r_joint, axis=1) * 0.6
        v_max = np.maximum(v_l_tcp, v_r_tcp)
        
        # Smooth v_max to remove white noise spikes from discrete differentiation
        from scipy.ndimage import uniform_filter1d
        v_max = uniform_filter1d(v_max, size=5)

        print(f"DEBUG: v_max during episode: p50={np.percentile(v_max, 50):.3f}, p95={np.percentile(v_max, 95):.3f}, max={np.max(v_max):.3f}")
        segments_l = self.segment_movement_units(v_l_tcp, fs)
        segments_r = self.segment_movement_units(v_r_tcp, fs)
        
        # Bimanual dwell: frames not active in either arm, excluding start/end
        active_frames = set()
        for s, e in segments_l:
            active_frames.update(range(s, e))
        for s, e in segments_r:
            active_frames.update(range(s, e))
            
        # exclude start and end
        start_ignore = int(fs)
        end_ignore = len(states) - int(fs)
        idle_count = 0
        for i in range(start_ignore, end_ignore):
            if i not in active_frames:
                idle_count += 1
                
        print(f"DEBUG DWELL: ep={ep_idx}, fs={fs}, len_states={len(states)}, active_frames_count={len(active_frames)}, idle_count={idle_count}")
        bimanual_dwell_fraction = idle_count / len(states) if len(states) > 0 else 0.0
        
        # if base is moving, do not penalize dwell
        if base_states is not None:
            v_base = np.abs(base_states[:, 0])
            base_moving = v_base > 0.05
            if np.any(base_moving):
                bimanual_dwell_fraction = 0.0

        # Only reset dwell if we have a base and the base is actually moving
        pass

        sparc_l = self.segment_metric_scores(self.compute_sparc_raw, v_l_tcp, fs, segments_l)
        sparc_r = self.segment_metric_scores(self.compute_sparc_raw, v_r_tcp, fs, segments_r)
        sparc_scores = sparc_l + sparc_r

        ldlj_l = self.segment_metric_scores(self.compute_ldlj_raw, v_l_tcp, fs, segments_l)
        ldlj_r = self.segment_metric_scores(self.compute_ldlj_raw, v_r_tcp, fs, segments_r)
        ldlj_scores = ldlj_l + ldlj_r

        sparc_val = float(np.median(sparc_scores)) if len(sparc_scores) > 0 else None
        ldlj_val = float(np.median(ldlj_scores)) if len(ldlj_scores) > 0 else None

        if len(tcp_drifts) >= 10:
            mean_tcp = np.mean(tcp_drifts) * 1000
            
            # Temporal consistency must be tested on the vector components, NOT the scalar magnitude.
            # A scalar magnitude is strictly positive, so its mean will trivially be > 0 (false positive bias).
            # We compute the norm of the mean vector (true bias) divided by the std of the vectors.
            mean_vec_l = np.mean(drift_vec_l, axis=0) * 1000
            std_vec_l = np.std(drift_vec_l, axis=0) * 1000
            mean_vec_r = np.mean(drift_vec_r, axis=0) * 1000
            std_vec_r = np.std(drift_vec_r, axis=0) * 1000
            
            n_stable = len(tcp_drifts)
            
            # Combine axes safely
            std_norm_l = np.linalg.norm(std_vec_l)
            std_norm_r = np.linalg.norm(std_vec_r)
            
            tc_l = np.linalg.norm(mean_vec_l) / (std_norm_l / np.sqrt(n_stable)) if std_norm_l > 0 else 999
            tc_r = np.linalg.norm(mean_vec_r) / (std_norm_r / np.sqrt(n_stable)) if std_norm_r > 0 else 999
            
            temporal_consistency = max(tc_l, tc_r)
            mean_rot = np.mean(rot_drifts)
        else:
            mean_tcp, mean_rot, temporal_consistency = 0.0, 0.0, 0.0

        reversal_rate = self.compute_direction_reversal_rate(actions)

        ep_dict = {
            "episode_idx": ep_idx,
            "tcp_drift_mm": mean_tcp,
            "std_tcp_mm": std_tcp,
            "rot_drift": mean_rot,
            "temporal_consistency": temporal_consistency,
            "base_unobservable": base_unobservable,
            "bimanual_dwell_fraction": float(bimanual_dwell_fraction),
            "reversal_rate": float(reversal_rate),
            "sparc_segments": sparc_scores,
            "n_movement_units": len(sparc_scores),
            "sparc": sparc_val,
            "ldlj": ldlj_val
        }

        return self._apply_calibration(ep_dict)

