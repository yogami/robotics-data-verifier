import pandas as pd
import numpy as np
import json
import os
import shutil

# We import lerobot inside the main function so this script can be written on local Mac
# without failing if lerobot isn't installed yet (it will run on the Phala GPU VM).

def create_lerobot_dataset_from_parquet(parquet_path, manifest_path, output_dir, filter_threshold=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    print(f"Loading corrupted parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    episodes_metadata = {ep["episode_idx"]: ep for ep in manifest["episodes"]}
    
    # Define features schema
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_pitch", "left_wrist_roll", "left_gripper",
                "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_pitch", "right_wrist_roll", "right_gripper"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_pitch", "left_wrist_roll", "left_gripper",
                "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_pitch", "right_wrist_roll", "right_gripper"
            ]
        }
    }
    
    # Remove output dir if exists to build clean
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    repo_id = os.path.basename(output_dir)
    print(f"Creating local LeRobotDataset at: {output_dir}")
    
    dataset = LeRobotDataset.create(
        repo_id=f"local/{repo_id}",
        root=os.path.dirname(output_dir),
        fps=50,
        features=features,
        use_videos=False
    )
    
    added_count = 0
    
    for ep_idx, group in df.groupby("episode_index"):
        ep_meta = episodes_metadata.get(ep_idx)
        
        # Apply filter if threshold is set
        if filter_threshold is not None and ep_meta is not None:
            entropy = ep_meta["entropy"]
            if entropy > filter_threshold:
                print(f"  Filtering out Episode {ep_idx} (entropy: {entropy:.3f} > {filter_threshold})")
                continue
                
        print(f"  Adding Episode {ep_idx} (len: {len(group)})...")
        
        # Sort by frame_index to ensure temporal order
        group = group.sort_values("frame_index").reset_index(drop=True)
        
        states = np.vstack(group["observation.state"].values).astype(np.float32)
        actions = np.vstack(group["action"].values).astype(np.float32)
        
        for i in range(len(group)):
            frame_data = {
                "observation.state": states[i],
                "action": actions[i]
            }
            dataset.add_frame(frame_data)
            
        dataset.save_episode()
        added_count += 1
        
    dataset.finalize()
    print(f"Finished. Saved {added_count} episodes to {output_dir}\n")

def main():
    parquet_path = "data/aloha_sim_insertion_corrupted.parquet"
    manifest_path = "data/episode_scores_corrupted_manifest.json"
    
    # 1. Build unfiltered dataset (all 15 episodes)
    print("=== BUILDING UNFILTERED DATASET ===")
    create_lerobot_dataset_from_parquet(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        output_dir="data/unfiltered_dataset",
        filter_threshold=None
    )
    
    # 2. Build filtered dataset (remove entropy > 0.35)
    print("=== BUILDING FILTERED DATASET ===")
    create_lerobot_dataset_from_parquet(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        output_dir="data/filtered_dataset",
        filter_threshold=0.35
    )

if __name__ == "__main__":
    main()
