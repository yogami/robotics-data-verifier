import argparse
import numpy as np
import torch
import json
import os
from tqdm import tqdm

class BCPolicy(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, action_dim),
        )
    def forward(self, obs):
        return self.net(obs)

class BCPolicyWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    def reset(self):
        pass
    def select_action(self, obs_dict):
        # Extract agent_pos (which matches observation.state)
        # obs_dict contains tensors of shape (1, 14) or similar
        obs = obs_dict.get("agent_pos")
        if obs is None:
            obs = obs_dict.get("state")
        if obs is None:
            # Fallback if other keys are used
            obs = next(iter(obs_dict.values()))
            
        expected_dim = self.model.net[0].in_features
        if obs.shape[-1] != expected_dim:
            if obs.shape[-1] == 16 and expected_dim == 14:
                obs = obs[:, 2:16]
            elif obs.shape[-1] == 14 and expected_dim == 16:
                obs = torch.cat([torch.zeros((obs.shape[0], 2), device=obs.device), obs], dim=-1)
            else:
                raise ValueError(f"Shape mismatch: model expects {expected_dim} but obs has {obs.shape[-1]}")
                
        return self.model(obs.to(self.device))
def eval_policy(policy_path, task="AlohaInsertion-v0", n_episodes=50, max_steps=400, device="cuda", infection_level=0, seed=1001):
    # Monkeypatch physics render to bypass actual OpenGL software rendering
    # since we only use state (agent_pos) inputs.
    try:
        from dm_control.mujoco.engine import Physics
        def dummy_render(self, height=240, width=320, camera_id=-1, scene_option=None, depth=False, segmentation=False):
            if depth:
                return np.zeros((height, width), dtype=np.float32)
            if segmentation:
                return np.zeros((height, width, 2), dtype=np.int32)
            return np.zeros((height, width, 3), dtype=np.uint8)
        Physics.render = dummy_render
        print("Successfully monkeypatched dm_control.mujoco.engine.Physics.render!")
    except Exception as e:
        print(f"Could not monkeypatch dm_control: {e}")

    import gymnasium as gym
    import gym_aloha  # noqa: F401
    env = gym.make(
        f"gym_aloha/{task}",
        obs_type="pixels_agent_pos",
        render_mode=None,
        observation_width=16,
        observation_height=16
    )
    
    print(f"Loading policy from: {policy_path}")
    
    loaded_dataset_hash = None
    # Load and detect format (LeRobot vs Custom BCPolicy)
    try:
        # Check if policy_path points to a custom BCPolicy .pt file or directory
        is_custom = False
        ckpt_path = policy_path
        if os.path.isdir(policy_path):
            model_pt = os.path.join(policy_path, "bc_model.pt")
            if os.path.exists(model_pt):
                ckpt_path = model_pt
                is_custom = True
        elif policy_path.endswith(".pt"):
            is_custom = True
            
        if is_custom:
            print("  Detected custom BCPolicy MLP model.")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # Cryptographic seed and infection alignment verification
            ckpt_seed = ckpt.get("seed")
            ckpt_inf = ckpt.get("infection_level")
            if ckpt_seed is None or ckpt_seed != seed:
                raise ValueError(f"Checkpoint seed missing or mismatched! Expected {seed}, got {ckpt_seed}")
            if ckpt_inf is None or ckpt_inf != infection_level:
                raise ValueError(f"Checkpoint infection level missing or mismatched! Expected {infection_level}, got {ckpt_inf}")
                
            loaded_dataset_hash = ckpt.get("dataset_hash", "unknown")
            model = BCPolicy(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            policy = BCPolicyWrapper(model, device)
        else:
            raise ValueError(f"Unsupported policy format at: {policy_path}. Only custom BCPolicy .pt checkpoints are supported in this pipeline.")
    except Exception as e:
        print(f"  Failed to load policy: {e}")
        raise e

    successes = 0
    episodes_data = []
    
    def to_tensor(x):
        if isinstance(x, dict):
            return {k: to_tensor(v) for k, v in x.items()}
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).to(device).unsqueeze(0)
            if t.is_floating_point():
                t = t.float()
            return t
        return x

    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        observation, info = env.reset()
        policy.reset()
        
        done = False
        step = 0
        ep_max_reward = 0.0
        
        while not done and step < max_steps:
            obs_tensor = to_tensor(observation)
            
            with torch.no_grad():
                action_tensor = policy.select_action(obs_tensor)
                
            action = action_tensor.squeeze(0).cpu().numpy()
            
            observation, reward, terminated, truncated, info = env.step(action)
            ep_max_reward = max(ep_max_reward, reward)
            done = terminated or truncated
            step += 1
            
        is_success = ep_max_reward >= 4.0
        if is_success:
            successes += 1
            
        episodes_data.append({
            "episode_idx": ep,
            "success": bool(is_success),
            "max_reward": float(ep_max_reward),
            "steps": step
        })
        
    env.close()
    
    # Read dataset hash directly from the PyTorch binary checkpoint if possible
    dataset_hash = loaded_dataset_hash if loaded_dataset_hash is not None else "unknown"
    if loaded_dataset_hash is None:
        try:
            # If policy_path is a file, read directly
            if os.path.isfile(policy_path):
                ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
                dataset_hash = ckpt.get("dataset_hash", "unknown")
            # If it is a directory, look for bc_model.pt inside it
            elif os.path.isdir(policy_path):
                model_pt = os.path.join(policy_path, "bc_model.pt")
                if os.path.exists(model_pt):
                    ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
                    dataset_hash = ckpt.get("dataset_hash", "unknown")
                else:
                    # Fallback to metadata.json for LeRobot policies
                    meta_file = os.path.join(policy_path, "dataset_metadata.json")
                    if os.path.exists(meta_file):
                        with open(meta_file, "r") as f:
                            dataset_hash = json.load(f).get("dataset_hash", "unknown")
        except Exception:
            pass
    
    success_rate = successes / n_episodes
    
    results = {
        "config": {
            "infection_level": infection_level,
            "seed": seed,
            "task": task,
            "dataset_hash": dataset_hash
        },
        "success_rate": success_rate,
        "episodes": episodes_data
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Success Rate: {success_rate * 100:.1f}%")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--policy_path", required=True)
    parser.add_argument("-t", "--task", default="AlohaInsertion-v0")
    parser.add_argument("-n", "--n_episodes", type=int, default=50)
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-o", "--output", default="eval_info.json")
    parser.add_argument("--infection_level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    
    res = eval_policy(
        policy_path=args.policy_path,
        task=args.task,
        n_episodes=args.n_episodes,
        device=args.device,
        infection_level=args.infection_level,
        seed=args.seed
    )
    
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
