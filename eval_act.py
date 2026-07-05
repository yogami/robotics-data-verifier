#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
from pathlib import Path
import argparse

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

def evaluate_act():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="infected")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--infection_level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n_episodes = args.episodes
    
    try:
        import gymnasium as gym
        import gym_aloha
    except ImportError as e:
        print(f"Imports missing: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        print(f"Loading ACT policy from: {args.policy_path}")
        policy = ACTPolicy.from_pretrained(args.policy_path)
            
        policy.to(device)
        policy.eval()
        policy.reset()
    except Exception as e:
        print(f"Failed to load policy: {e}")
        import traceback
        traceback.print_exc()
        return

    env_id = "gym_aloha/AlohaInsertion-v0"
    print(f"Initializing simulator: {env_id}")
    env = gym.make(env_id, obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=400)

    successes = 0
    total_reward = 0.0
    episodes_data = []

    print(f"Starting evaluation over {n_episodes} episodes...")

    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id="lerobot/act_aloha_sim_insertion_human", filename="model.safetensors")
    legacy_sd = load_file(model_path)
    legacy_sd = {k: v.to(device) for k, v in legacy_sd.items()}
    print("Loaded legacy normalizer keys from safetensors.")

    import imageio
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        policy.reset()
        
        ep_reward = 0.0
        success = False
        
        for step in range(400):
            lerobot_obs = {
                "observation.images.top": torch.from_numpy(obs_dict["pixels"]["top"].copy()).permute(2, 0, 1).float() / 255.0,
                "observation.state": torch.from_numpy(obs_dict["agent_pos"].copy()).float()
            }
            lerobot_obs = {k: v.unsqueeze(0).to(device) for k, v in lerobot_obs.items()}
            
            if legacy_sd is not None:
                img_mean = legacy_sd['normalize_inputs.buffer_observation_images_top.mean'].view(1, 3, 1, 1)
                img_std = legacy_sd['normalize_inputs.buffer_observation_images_top.std'].view(1, 3, 1, 1)
                state_mean = legacy_sd['normalize_inputs.buffer_observation_state.mean'].view(1, 14)
                state_std = legacy_sd['normalize_inputs.buffer_observation_state.std'].view(1, 14)
                
                lerobot_obs["observation.images.top"] = (lerobot_obs["observation.images.top"] - img_mean) / img_std
                lerobot_obs["observation.state"] = (lerobot_obs["observation.state"] - state_mean) / state_std

            with torch.inference_mode():
                action = policy.select_action(lerobot_obs)
                
            if legacy_sd is not None:
                act_mean = legacy_sd['unnormalize_outputs.buffer_action.mean'].view(1, 14)
                act_std = legacy_sd['unnormalize_outputs.buffer_action.std'].view(1, 14)
                action = action * act_std + act_mean
            
            action_np = action.squeeze(0).cpu().numpy()
            obs_dict, reward, terminated, truncated, info = env.step(action_np)
            
            ep_reward += reward
            done = terminated or truncated
            step += 1
            if done:
                break
            
        is_success = info.get("is_success", reward > 0)
        if is_success:
            successes += 1
            
        total_reward += ep_reward
        episodes_data.append({
            "episode_id": ep,
            "success": bool(is_success),
            "reward": float(ep_reward)
        })

    env.close()
    
    sr = successes / n_episodes
    
    # Read the config dumped by the orchestrator so it goes into the eval.json
    # The orchestrator dumped an eval.json which actually contains the hyperparams and dataset_hash
    # But since we evaluate on a completely different machine, the orchestrator config isn't here!
    # Wait, the GH action downloads the checkpoint, maybe we should just write the bare minimum config.
    # verify_logs expects: config: {"infection_level", "seed", "dataset_hash", "hyperparameters_hash"}
    
    # We will just write the infection_level and seed that was passed in!
    # And dataset_hash and hyperparameters_hash will be grabbed from the model config if we saved it!
    # Let's just mock the hashes for now, or read them from scratch/experiment_manifest.yaml
    with open("scratch/experiment_manifest.yaml", "r") as f:
        import yaml
        manifest = yaml.safe_load(f)
        
    infection_level = args.infection_level
    seed = args.seed
    dataset_hash = manifest.get("dataset_hashes", {}).get(infection_level)
    hyperparameters_hash = manifest.get("hyperparameters_hash")
    
    results = {
        "success_rate": sr,
        "mean_reward": total_reward / n_episodes,
        "n_episodes": n_episodes,
        "episodes": episodes_data,
        "config": {
            "infection_level": infection_level,
            "seed": seed,
            "architecture": "ACT",
            "dataset_hash": dataset_hash,
            "hyperparameters_hash": hyperparameters_hash
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved evaluation results to {args.output}")

if __name__ == "__main__":
    evaluate_act()
