import pandas as pd
import numpy as np
import os

def corrupt_dataset():
    input_path = "data/aloha_sim_insertion_human.parquet"
    output_path = "data/aloha_sim_insertion_corrupted.parquet"

    print(f"Reading clean dataset: {input_path}")
    df = pd.read_parquet(input_path)
    
    corrupted_dfs = []
    
    for ep_idx, group in df.groupby("episode_index"):
        # Make a copy to avoid SettingWithCopy warning
        group = group.copy().reset_index(drop=True)
        N = len(group)
        
        if ep_idx == 3:
            # 1. Calibration Drift (offset states from actions)
            print(f"Corrupting Episode {ep_idx}: Injecting Calibration Drift")
            states = np.vstack(group["observation.state"].values)
            # Offset the follower state by ~0.15 rad on primary joints to trigger TCP drift
            states[:, 0:6] += 0.12  # Left arm waist, shoulder, elbow drift
            group["observation.state"] = list(states)
            
        elif ep_idx == 6:
            # 2. Operator Hesitation / Arrest
            print(f"Corrupting Episode {ep_idx}: Injecting Operator Hesitation")
            mid_idx = N // 2
            h_frames = 80  # 1.6 seconds of freeze
            
            # Extract the frozen frame
            freeze_row = group.iloc[mid_idx].copy()
            
            # Duplicate the rows
            freeze_rows = pd.concat([pd.DataFrame([freeze_row])]*h_frames, ignore_index=True)
            
            # Split and insert
            part1 = group.iloc[:mid_idx]
            part2 = group.iloc[mid_idx:]
            group = pd.concat([part1, freeze_rows, part2], ignore_index=True)
            
            # Re-index frame_index and update timestamps
            group["frame_index"] = np.arange(len(group))
            dt = 0.02
            group["timestamp"] = group.iloc[0]["timestamp"] + group["frame_index"] * dt
            
        elif ep_idx == 9:
            # 3. High Reversals / Tremor
            print(f"Corrupting Episode {ep_idx}: Injecting Joint Tremor")
            actions = np.vstack(group["action"].values)
            t = np.arange(len(actions)) * 0.02
            # Add 8Hz oscillation
            tremor = np.sin(t[:, None] * 2 * np.pi * 8.0) * 0.04
            actions[:, 0:6] += tremor
            group["action"] = list(actions)
            
        elif ep_idx == 12:
            # 4. Operator Hesitation / Arrest
            print(f"Corrupting Episode {ep_idx}: Injecting Operator Hesitation")
            mid_idx = N // 2
            h_frames = 80  # 1.6 seconds of freeze
            
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
    # Ensure index column matches row index
    df_corrupted["index"] = np.arange(len(df_corrupted))
    
    print(f"Writing corrupted dataset: {output_path}")
    df_corrupted.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    corrupt_dataset()
