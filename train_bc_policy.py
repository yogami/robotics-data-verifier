#!/usr/bin/env python3
"""
train_bc_policy.py

Self-contained Behavioral Cloning (BC) training script.
Trains a simple MLP policy on filtered vs unfiltered ALOHA parquet data.
No lerobot dependency — uses PyTorch + pyarrow directly.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import hashlib
from policy import BCPolicy

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ─── Dataset ──────────────────────────────────────────────────────────────────

class AlohaDataset(Dataset):
    """Loads ALOHA parquet episodes and returns (obs, action) pairs."""

    def __init__(self, parquet_path: str, episode_ids: list[int] | None = None):
        import io
        with open(parquet_path, "rb") as f:
            raw_bytes = f.read()
        self.dataset_hash = hashlib.sha256(raw_bytes).hexdigest()
        df = pd.read_parquet(io.BytesIO(raw_bytes))

        # Support both naming conventions
        if "episode_id" in df.columns:
            ep_col = "episode_id"
        elif "episode_index" in df.columns:
            ep_col = "episode_index"
        else:
            # Assume single episode
            ep_col = None

        if episode_ids is not None and ep_col is not None:
            df = df[df[ep_col].isin(episode_ids)]

        # Identify state and action columns
        state_cols = [c for c in df.columns if c.startswith("observation.") or c.startswith("state.")]
        action_cols = [c for c in df.columns if c.startswith("action.")]

        if not state_cols:
            # Try joint_positions / joint_velocities naming
            state_cols = [c for c in df.columns if "position" in c or "velocity" in c]
        if not action_cols:
            action_cols = [c for c in df.columns if "action" in c.lower()]

        # Fallback: treat all numeric non-meta columns as features
        if not state_cols and not action_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            meta_cols = {ep_col, "timestamp", "frame_id", "index", "frame_index", "episode_index"}
            feature_cols = [c for c in numeric_cols if c not in meta_cols]
            mid = len(feature_cols) // 2
            state_cols = feature_cols[:mid]
            action_cols = feature_cols[mid:]

        def _flatten_cols(cols):
            if not cols:
                return np.zeros((len(df), 1), dtype=np.float32)
            if len(cols) == 1 and df[cols[0]].dtype == "O":
                return np.stack(df[cols[0]].tolist()).astype(np.float32)
            return df[cols].fillna(0.0).values.astype(np.float32)

        self.obs = torch.tensor(_flatten_cols(state_cols), dtype=torch.float32)
        self.actions = torch.tensor(_flatten_cols(action_cols), dtype=torch.float32)
        self.obs_dim = self.obs.shape[1]
        self.action_dim = self.actions.shape[1]

        print(f"  Dataset: {len(self.obs)} frames, obs_dim={self.obs_dim}, action_dim={self.action_dim}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


# ─── Training ─────────────────────────────────────────────────────────────────

def train(parquet_path: str, episode_ids: list[int] | None, output_path: str,
          epochs: int = 50, lr: float = 1e-3, batch_size: int = 256, device: str = "cuda",
          hf_repo: str = None, hf_token: str = None, hf_branch: str = None,
          seed: int = None, infection_level: int = None):

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    print(f"\n{'='*60}")
    print(f"Training on: {parquet_path}")
    print(f"Episodes: {episode_ids if episode_ids is not None else 'ALL'}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    dataset = AlohaDataset(parquet_path, episode_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = BCPolicy(dataset.obs_dim, dataset.action_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")

    train_losses = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for obs, actions in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            pred = model(obs)
            loss = criterion(pred, actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(obs)
        epoch_loss /= len(dataset)
        scheduler.step()
        train_losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={epoch_loss:.6f} | {elapsed:.1f}s")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dataset_hash = dataset.dataset_hash
    
    import json
    hyperparameters = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "training_steps": epochs,
        "architecture": "MLP-BC",
        "hidden_dim": 256
    }
    canonical_hp = json.dumps(hyperparameters, sort_keys=True, separators=(',', ':'))
    hp_hash = hashlib.sha256(canonical_hp.encode('utf-8')).hexdigest()
    
    torch.save({
        "model_state": model.state_dict(),
        "obs_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "final_loss": train_losses[-1],
        "train_losses": train_losses,
        "dataset_hash": dataset_hash,
        "hyperparameters_hash": hp_hash,
        "seed": seed,
        "infection_level": infection_level,
    }, output_path)
    
    print(f"  Model saved to {output_path}")
    print(f"  Final loss: {train_losses[-1]:.6f}")
    
    # Secure Direct Upload to Hugging Face
    if hf_repo and hf_token and hf_branch:
        print(f"  Direct upload to Hugging Face: {hf_repo} (Branch: {hf_branch})...")
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo, exist_ok=True)
            api.create_branch(repo_id=hf_repo, branch=hf_branch, exist_ok=True)
            commit_info = api.upload_file(
                path_or_fileobj=output_path,
                path_in_repo="bc_model.pt",
                repo_id=hf_repo,
                revision=hf_branch
            )
            print(f"  Direct HF upload successful. HF_COMMIT_SHA={commit_info.oid}")
        except Exception as e:
            print(f"  CRITICAL: HF upload failed: {e}")
            sys.exit(1)
            
    return model, dataset.obs_dim, dataset.action_dim, train_losses[-1]


# ─── Simulation evaluation ────────────────────────────────────────────────────

def evaluate(model_path: str, n_episodes: int = 50, device: str = "cuda") -> dict:
    """Run rollouts in AlohaInsertion-v0 and return success rate."""
    try:
        import gymnasium as gym
        import gym_aloha  # noqa: F401
    except ImportError as e:
        print(f"  Cannot import gym_aloha: {e}")
        return {"success_rate": -1.0, "mean_reward": -1.0, "n_episodes": n_episodes}

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = BCPolicy(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    env_id = "gym_aloha/AlohaInsertion-v0"
    try:
        env = gym.make(env_id, obs_type="pixels_agent_pos", render_mode=None)
    except Exception as e:
        print(f"  Failed to make {env_id}: {e}")
        return {"success_rate": -1.0, "mean_reward": -1.0, "n_episodes": n_episodes}

    successes = 0
    total_reward = 0.0

    for ep in range(n_episodes):
        obs_dict, _ = env.reset()
        # Extract agent_pos from observation
        if isinstance(obs_dict, dict):
            obs_arr = obs_dict.get("agent_pos", np.zeros(model.net[0].in_features))
        else:
            obs_arr = np.array(obs_dict).flatten()

        # Pad / truncate to match obs_dim
        obs_dim = ckpt["obs_dim"]
        if len(obs_arr) < obs_dim:
            obs_arr = np.pad(obs_arr, (0, obs_dim - len(obs_arr)))
        else:
            obs_arr = obs_arr[:obs_dim]

        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 400:
            with torch.no_grad():
                obs_t = torch.tensor(obs_arr, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(obs_t).squeeze(0).cpu().numpy()
            obs_dict, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
            if isinstance(obs_dict, dict):
                obs_arr = obs_dict.get("agent_pos", obs_arr)
            else:
                obs_arr = np.array(obs_dict).flatten()
            if len(obs_arr) < obs_dim:
                obs_arr = np.pad(obs_arr, (0, obs_dim - len(obs_arr)))
            else:
                obs_arr = obs_arr[:obs_dim]
        
        if info.get("is_success", reward > 0):
            successes += 1
        total_reward += ep_reward

    env.close()
    result = {
        "success_rate": successes / n_episodes,
        "mean_reward": total_reward / n_episodes,
        "n_episodes": n_episodes,
        "successes": successes,
    }
    print(f"  Success rate: {result['success_rate']:.1%} ({successes}/{n_episodes})")
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--episode-ids", type=str, default=None, help="Comma-separated list of episode IDs")
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-eval", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-branch", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--infection-level", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    episode_ids = None
    if args.episode_ids:
        episode_ids = [int(x) for x in args.episode_ids.split(",")]

    model, obs_dim, action_dim, final_loss = train(
        parquet_path=args.parquet,
        episode_ids=episode_ids,
        output_path=args.output_model,
        epochs=args.epochs,
        device=device,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        hf_branch=args.hf_branch,
        seed=args.seed,
        infection_level=args.infection_level
    )

    if args.no_eval:
        eval_result = {"success_rate": -1.0, "mean_reward": -1.0, "skipped": True}
    else:
        print(f"\nEvaluating {args.output_model}...")
        eval_result = evaluate(args.output_model, n_episodes=args.eval_episodes, device=device)

    eval_result["final_train_loss"] = final_loss
    os.makedirs(os.path.dirname(args.output_eval) or ".", exist_ok=True)
    with open(args.output_eval, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"\nEval saved to {args.output_eval}")
    print(json.dumps(eval_result, indent=2))
