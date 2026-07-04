import os
import subprocess
import pandas as pd
import numpy as np

def generate_and_upload():
    RUNPOD_IP = os.environ.get("RUNPOD_IP")
    if not RUNPOD_IP:
        print("❌ Error: RUNPOD_IP environment variable is not set.")
        print("Please export it in your terminal first: export RUNPOD_IP=xxx.xxx.xxx.xxx")
        return
        
    input_path = "data/aloha_sim_insertion_human.parquet"
    if not os.path.exists(input_path):
        print(f"❌ Error: Baseline dataset not found at {input_path}")
        return
        
    print(f"Reading clean dataset: {input_path}")
    df = pd.read_parquet(input_path)
    
    unique_episodes = df["episode_index"].unique()
    num_episodes = len(unique_episodes)
    
    fractions = {
        0: 0.0,
        25: 0.25,
        50: 0.50,
        75: 0.75,
        100: 1.00
    }
    
    # 1. Generate local parquet files with the correct orchestrator names
    for pct, frac in fractions.items():
        output_path = f"data/infection_{pct}.parquet"
        print(f"\n--- Generating {pct}% Infected Dataset: {output_path} ---")
        
        if pct == 0:
            # Baseline is identical to clean human dataset
            df_corrupted = df.copy()
        else:
            num_to_infect = int(num_episodes * frac)
            episodes_to_infect = set(unique_episodes[:num_to_infect])
            corrupted_dfs = []
            
            for ep_idx, group in df.groupby("episode_index"):
                group = group.copy().reset_index(drop=True)
                N = len(group)
                
                if ep_idx in episodes_to_infect:
                    # Alternate between Hesitation and Calibration Drift
                    if ep_idx % 2 == 0:
                        # Calibration Drift
                        states = np.vstack(group["observation.state"].values)
                        states[:, 0:6] += 0.12
                        group["observation.state"] = list(states)
                    else:
                        # Hesitation
                        mid_idx = N // 2
                        h_frames = 80
                        freeze_row = group.iloc[mid_idx].copy()
                        freeze_rows = pd.concat([pd.DataFrame([freeze_row])]*h_frames, ignore_index=True)
                        part1 = group.iloc[:mid_idx]
                        part2 = group.iloc[mid_idx:]
                        group = pd.concat([part1, freeze_rows, part2], ignore_index=True)
                        
                        group["frame_index"] = np.arange(len(group))
                        dt = 0.02
                        group["timestamp"] = group.iloc[0]["timestamp"] + group["frame_index"] * dt
                        
                corrupted_dfs.append(group)
                
            df_corrupted = pd.concat(corrupted_dfs, ignore_index=True)
            df_corrupted["index"] = np.arange(len(df_corrupted))
            
        df_corrupted.to_parquet(output_path)
        print(f"Saved: {output_path}")
        
    RUNPOD_PORT = os.environ.get("RUNPOD_PORT", "22")
    
    # 2. Create the data directory on the RunPod
    print(f"\nSSH: Creating /root/data/ directory on RunPod at {RUNPOD_IP}:{RUNPOD_PORT}...")
    mkdir_cmd = f"ssh -o StrictHostKeyChecking=no -p {RUNPOD_PORT} root@{RUNPOD_IP} \"mkdir -p /root/data /root/outputs\""
    subprocess.run(mkdir_cmd, shell=True, check=True)
    
    # 3. SCP upload all datasets
    for pct in fractions.keys():
        local_file = f"data/infection_{pct}.parquet"
        print(f"SCP: Uploading {local_file} to RunPod...")
        scp_cmd = f"scp -o StrictHostKeyChecking=no -P {RUNPOD_PORT} {local_file} root@{RUNPOD_IP}:/root/data/"
        subprocess.run(scp_cmd, shell=True, check=True)
        
    print("\n✅ Success! All datasets generated and uploaded to RunPod successfully.")

if __name__ == "__main__":
    generate_and_upload()
