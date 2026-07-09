#!/usr/bin/env python3
import os

# FORCE SINGLE-THREADED MATH TO PREVENT 120-CORE CPU THREAD EXHAUSTION
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
else:
    # On macOS, use native GLFW/egl windowing rendering
    os.environ["MUJOCO_GL"] = "glfw"


import json
import torch
# Secondary thread protection for PyTorch internals
torch.set_num_threads(1)
import numpy as np
from pathlib import Path
import argparse
from huggingface_hub import hf_hub_download
import traceback

def run_episodes(worker_args):
    worker_id, episodes_to_run, policy_path, device, model_path = worker_args
    import gymnasium as gym
    import gym_aloha
    from lerobot.policies.act.modeling_act import ACTPolicy
    from safetensors.torch import load_file
    
    print(f"[Worker {worker_id}] Loading policy from {policy_path} onto {device}")
    
    results = []
    
    try:
        policy = ACTPolicy.from_pretrained(policy_path)
        
        # Fable 5 Fix: Enable inference-time temporal ensembling
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = 0.01
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(policy.config.temporal_ensemble_coeff, policy.config.chunk_size)
        policy.reset()
        
        policy.to(device)
        policy.eval()
        
        env_id = "gym_aloha/AlohaTransferCube-v0"
        # REMOVED: render_mode="rgb_array" to double CPU simulation performance
        env = gym.make(env_id, obs_type="pixels_agent_pos", max_episode_steps=400)
        
        from safetensors.torch import load_file
        import os
        legacy_sd = load_file(os.path.join(policy_path, "policy_preprocessor_step_3_normalizer_processor.safetensors"))
        post_sd = load_file(os.path.join(policy_path, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        legacy_sd = {k: v.to(device) for k, v in legacy_sd.items()}
        post_sd = {k: v.to(device) for k, v in post_sd.items()}
        for ep in episodes_to_run:
            try:
                obs_dict, _ = env.reset(seed=ep + worker_id*1000) # Ensure different initial states
                policy.reset()
                
                ep_reward = 0.0
                frames = []
                for step in range(400):
                    raw_img = torch.from_numpy(obs_dict["pixels"]["top"].copy()).permute(2, 0, 1).float() / 255.0
                    raw_img = raw_img.unsqueeze(0)
                    resized_img = torch.nn.functional.interpolate(raw_img, size=(480, 640), mode="bilinear", align_corners=False)
                    
                    lerobot_obs = {
                        "observation.images.top": resized_img.squeeze(0),
                        "observation.state": torch.from_numpy(obs_dict["agent_pos"].copy()).float()
                    }
                    lerobot_obs = {k: v.unsqueeze(0).to(device) for k, v in lerobot_obs.items()}
                    
                    if legacy_sd is not None:
                        img_mean = legacy_sd['observation.images.top.mean'].view(1, 3, 1, 1)
                        img_std = legacy_sd['observation.images.top.std'].view(1, 3, 1, 1)
                        state_mean = legacy_sd['observation.state.mean'].view(1, 14)
                        state_std = legacy_sd['observation.state.std'].view(1, 14)
                        
                        lerobot_obs["observation.images.top"] = (lerobot_obs["observation.images.top"] - img_mean) / img_std
                        lerobot_obs["observation.state"] = (lerobot_obs["observation.state"] - state_mean) / state_std

                    with torch.inference_mode():
                        action = policy.select_action(lerobot_obs)
                        
                    if post_sd is not None:
                        act_mean = post_sd['action.mean'].view(1, 14)
                        act_std = post_sd['action.std'].view(1, 14)
                        action = action * act_std + act_mean
                    
                    action_np = action.squeeze(0).cpu().numpy()
                    obs_dict, reward, terminated, truncated, info = env.step(action_np)
                    
                    if ep < 2:
                        frames.append(obs_dict["pixels"]["top"].copy())
                    
                    ep_reward += reward
                    if step % 100 == 0: print(f"[Worker {worker_id}] Episode {ep} | Step {step}/400 | Current Reward: {ep_reward}", flush=True)
                    if terminated or truncated:
                        break
                    
                if ep < 2:
                    import imageio
                    video_path = f"outputs_phase2/clean_seed_100/eval/video_ep{ep}.mp4"
                    imageio.mimsave(video_path, frames, fps=50)
                    print(f"[Worker {worker_id}] Saved video to {video_path}")
                    
                is_success = info.get("is_success", reward > 0)
                results.append({
                    "episode_id": ep,
                    "success": bool(is_success),
                    "reward": float(ep_reward),
                    "error": None
                })
                print(f"[Worker {worker_id}] Finished episode {ep} (Success: {bool(is_success)})")
            
            except Exception as e:
                err = traceback.format_exc()
                print(f"[Worker {worker_id}] FAILED episode {ep}: {err}")
                results.append({
                    "episode_id": ep,
                    "success": False,
                    "reward": 0.0,
                    "error": str(e)
                })
        
        env.close()
    
    except Exception as outer_e:
        err = traceback.format_exc()
        print(f"[Worker {worker_id}] FATAL WORKER CRASH: {err}")
        # If the environment completely crashed, return failures for the remaining episodes
        for ep in episodes_to_run:
            if not any(r["episode_id"] == ep for r in results):
                results.append({
                    "episode_id": ep,
                    "success": False,
                    "reward": 0.0,
                    "error": str(outer_e)
                })

    return results

def evaluate_act():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--infection_level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    # ADDED METADATA ARGUMENTS TO REMOVE LOCAL FILE DEPENDENCY
    parser.add_argument("--dataset_hash", type=str, default="unknown")
    parser.add_argument("--hyperparameters_hash", type=str, default="unknown")
    args = parser.parse_args()
    
    n_episodes = args.episodes
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting parallel evaluation over {n_episodes} episodes...")
    
    # Download ONCE in the parent process to completely eliminate file lock deadlocks
    model_path = hf_hub_download(repo_id="lerobot/act_aloha_sim_insertion_human", filename="model.safetensors")
    
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    # VRAM SAFETY CAP: Maximum of 10 workers to safely fit within 24GB RTX 4090 VRAM.
    num_workers = min(n_episodes, max(1, os.cpu_count() - 2), 10)
    if num_workers < 1:
        num_workers = 1
    
    episodes_per_worker = n_episodes // num_workers
    worker_args = []
    
    for i in range(num_workers):
        start_ep = i * episodes_per_worker
        end_ep = (i + 1) * episodes_per_worker if i < num_workers - 1 else n_episodes
        eps = list(range(start_ep, end_ep))
        worker_args.append((i, eps, args.policy_path, device, model_path))
        
    episodes_data = []
    with mp.Pool(num_workers) as pool:
        for res in pool.imap_unordered(run_episodes, worker_args):
            episodes_data.extend(res)
            
    # Sort by episode_id to maintain deterministic ordering in the JSON
    episodes_data.sort(key=lambda x: x["episode_id"])
    
    successes = sum(1 for e in episodes_data if e["success"])
    total_reward = sum(e["reward"] for e in episodes_data)
    sr = successes / n_episodes
    
    results = {
        "success_rate": sr,
        "mean_reward": total_reward / n_episodes,
        "n_episodes": n_episodes,
        "episodes": episodes_data,
        "config": {
            "infection_level": args.infection_level,
            "seed": args.seed,
            "architecture": "ACT",
            "dataset_hash": args.dataset_hash,
            "hyperparameters_hash": args.hyperparameters_hash
        }
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved parallel evaluation results to {args.output}")

if __name__ == "__main__":
    evaluate_act()
