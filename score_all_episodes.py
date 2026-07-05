import os
import json
import pandas as pd
import numpy as np
from lerobot_buffer_hook import ArchitectureAwareDriftGate

def main():
    dataset_path = "data/aloha_sim_insertion_corrupted.parquet"
    print(f"Loading local dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please ensure the dataset is in the data/ directory.")
        return

    df = pd.read_parquet(dataset_path)
    unique_episodes = sorted(df["episode_index"].unique())
    print(f"Found {len(unique_episodes)} unique episodes in dataset.")

    gate = ArchitectureAwareDriftGate()
    episode_data = []

    # Step 1: Score each episode individually
    print("Scoring episodes (SPARC, bimanual dwell, drift)...")
    for ep_idx in unique_episodes:
        ep_df = df[df["episode_index"] == ep_idx]
        
        # States are 14-dim (arms) or 16-dim (arms + base)
        states = np.vstack(ep_df["observation.state"].values)
        actions = np.vstack(ep_df["action"].values)
        timestamps = ep_df["timestamp"].values

        # Compute scores (stores segments in the dictionary returned)
        res = gate._score_episode(ep_idx, actions, states, timestamps)
        episode_data.append(res)
        
        if ep_idx % 10 == 0:
            print(f"  Processed episode {ep_idx}/{len(unique_episodes)}")

    # Step 2: Calibrate dataset-wide robust statistics & finalize mapping
    print("Calibrating robust z-scores and applying final calibration...")
    finalized_episodes = gate._finalize_episodes(episode_data)

    # Save results to a manifest
    output_manifest = "data/episode_scores_corrupted_manifest.json"
    
    # Standardize dictionary for JSON serialization (numpy types to python types)
    serialized_data = []
    for ep in finalized_episodes:
        clean_ep = {
            "episode_idx": int(ep["episode_idx"]),
            "tcp_drift_mm": float(ep["tcp_drift_mm"]),
            "std_tcp_mm": float(ep["std_tcp_mm"]),
            "rot_drift": float(ep["rot_drift"]),
            "temporal_consistency": float(ep["temporal_consistency"]),
            "base_unobservable": bool(ep["base_unobservable"]),
            "bimanual_dwell_fraction": float(ep["bimanual_dwell_fraction"]),
            "reversal_rate": float(ep["reversal_rate"]),
            "n_movement_units": int(ep["n_movement_units"]),
            "sparc": float(ep["sparc"]) if ep["sparc"] is not None else None,
            "ldlj": float(ep["ldlj"]) if ep["ldlj"] is not None else None,
            "sparc_robust_z": float(ep["sparc_robust_z"]),
            "sparc_entropy": float(ep["sparc_entropy"]),
            "dwell_entropy": float(ep["dwell_entropy"]),
            "entropy": float(ep["entropy"]),
            "loss_weight": float(ep["loss_weight"]),
            "sparc_flagged": bool(ep["sparc_flagged"]),
            "drift_flagged": bool(ep["drift_flagged"]),
            "reversal_flagged": bool(ep["reversal_flagged"])
        }
        serialized_data.append(clean_ep)

    with open(output_manifest, "w") as f:
        json.dump({
            "calibration_stats": gate._calib,
            "episodes": serialized_data
        }, f, indent=2)

    print(f"Successfully saved manifest to {output_manifest}")

    # Summary analysis
    df_scores = pd.DataFrame(serialized_data)
    flagged = df_scores[df_scores["entropy"] > 0.5]
    print("\n--- Calibration & Quality Summary ---")
    print(f"Dataset calibration details: {gate._calib}")
    print(f"Total scored: {len(df_scores)}")
    print(f"Flagged (entropy > 0.5): {len(flagged)} episodes ({len(flagged)/len(df_scores)*100:.1f}%)")
    print(f"Average entropy: {df_scores['entropy'].mean():.3f}")
    print(f"Min loss weight: {df_scores['loss_weight'].min():.3f}, Max: {df_scores['loss_weight'].max():.3f}")

if __name__ == "__main__":
    main()
