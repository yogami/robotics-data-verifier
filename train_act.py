#!/usr/bin/env python3
import os
import time
import argparse
import random
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from lerobot.policies.act.modeling_act import ACTPolicy

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)


class AlohaChunkedDataset(Dataset):
    """Loads a standard HuggingFace dataset from disk and serves
    (image, state, action_chunk, action_is_pad) samples for ACT training."""

    def __init__(self, dataset_path, chunk_size):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        import os
        repo_id = "local/" + os.path.basename(dataset_path.rstrip("/"))
        self.lerobot_ds = LeRobotDataset(repo_id, root=dataset_path)
        hf_dataset = self.lerobot_ds.hf_dataset

        self.dataset = hf_dataset
        self.chunk_size = chunk_size
        columns = list(self.lerobot_ds.features.keys())

        # --- Resolve image column (prefer observation.images.top) ---
        image_cols = [c for c in columns if c.startswith("observation.images")]
        if not image_cols:
            image_cols = [c for c in columns if "image" in c.lower()]
        if not image_cols:
            raise ValueError(f"No image column found in dataset. Columns: {columns}")
        if "observation.images.top" in image_cols:
            self.image_col = "observation.images.top"
        else:
            self.image_col = image_cols[0]

        # --- Resolve state column ---
        if "observation.state" in columns:
            self.state_col = "observation.state"
        else:
            state_candidates = [
                c for c in columns
                if (c.startswith("observation.") or c.startswith("state")) and not c.startswith("observation.images")
            ]
            if not state_candidates:
                raise ValueError(f"No state column found in dataset. Columns: {columns}")
            self.state_col = state_candidates[0]

        # --- Resolve action column ---
        if "action" in columns:
            self.action_col = "action"
        else:
            action_candidates = [c for c in columns if c.startswith("action")]
            if not action_candidates:
                raise ValueError(f"No action column found in dataset. Columns: {columns}")
            self.action_col = action_candidates[0]

        # Preload low-dimensional data as tensors (images are decoded lazily).
        self.states = torch.tensor(np.stack(self.dataset[self.state_col]), dtype=torch.float32)
        if self.states.dim() == 1:
            self.states = self.states.unsqueeze(-1)
        self.actions = torch.tensor(np.stack(self.dataset[self.action_col]), dtype=torch.float32)
        if self.actions.dim() == 1:
            self.actions = self.actions.unsqueeze(-1)

        # Episode boundaries so action chunks never bleed across episodes.
        if "episode_index" in columns:
            episode_indices = np.asarray(self.dataset["episode_index"])
        elif "episode_id" in columns:
            episode_indices = np.asarray(self.dataset["episode_id"])
        else:
            episode_indices = np.zeros(len(self.dataset), dtype=np.int64)

        # For each frame, index of the last frame (exclusive) of its episode.
        self.episode_ends = np.empty(len(episode_indices), dtype=np.int64)
        n = len(episode_indices)
        end = n
        for i in range(n - 1, -1, -1):
            if i < n - 1 and episode_indices[i] != episode_indices[i + 1]:
                end = i + 1
            self.episode_ends[i] = end

        print(
            f"Dataset: {len(self.dataset)} frames | image_col={self.image_col} | "
            f"state_dim={self.states.shape[1]} | action_dim={self.actions.shape[1]}"
        )

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, idx):
        img = self.lerobot_ds[idx][self.image_col]
        if isinstance(img, torch.Tensor):
            img_t = img.float()
            if img_t.max() > 1.0:
                img_t = img_t / 255.0
            if img_t.dim() == 3 and img_t.shape[0] not in (1, 3):
                img_t = img_t.permute(2, 0, 1)
            return img_t
        img_np = np.asarray(img)
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        else:
            img_np = img_np.astype(np.float32)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        # HWC -> CHW
        return torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx):
        image = self._load_image(idx)
        state = self.states[idx]

        episode_end = self.episode_ends[idx]
        chunk_end = min(idx + self.chunk_size, episode_end)
        action_chunk = self.actions[idx:chunk_end]

        num_valid = action_chunk.shape[0]
        action_is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        if num_valid < self.chunk_size:
            pad = action_chunk[-1:].repeat(self.chunk_size - num_valid, 1)
            action_chunk = torch.cat([action_chunk, pad], dim=0)
            action_is_pad[num_valid:] = True

        return image, state, action_chunk, action_is_pad


def train(dataset_path, output_path, epochs, batch_size, lr, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading pretrained ACT model...")
    repo_id = "lerobot/act_aloha_sim_insertion_human"
    policy = ACTPolicy.from_pretrained(repo_id)
    policy.to(device)
    policy.train()

    # Freeze vision backbone to prevent catastrophic forgetting
    for param in policy.model.backbone.parameters():
        param.requires_grad = False

    chunk_size = policy.config.chunk_size
    dataset = AlohaChunkedDataset(dataset_path, chunk_size)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        generator=g,
        worker_init_fn=seed_worker,
        persistent_workers=True,
        prefetch_factor=2
    )

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, policy.parameters()), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, states, action_chunks, action_is_pad in loader:
            images = images.to(device)
            states = states.to(device)
            action_chunks = action_chunks.to(device)
            action_is_pad = action_is_pad.to(device)

            batch = {
                "observation.images.top": images,
                "observation.state": states,
                "action": action_chunks,
                "action_is_pad": action_is_pad,
            }

            output = policy(batch)
            loss = output[0] if isinstance(output, tuple) else output

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(states)

        epoch_loss /= len(dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | loss={epoch_loss:.6f} | {time.time()-t0:.1f}s")

    os.makedirs(output_path, exist_ok=True)
    policy.save_pretrained(output_path)
    print(f"Saved ACT model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/aloha_infected_0",
                        help="Path to HuggingFace dataset saved with save_to_disk")
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infection-level", type=int, default=0)
    parser.add_argument("--hf-repo", type=str)
    parser.add_argument("--hf-branch", type=str)
    args = parser.parse_args()

    train(args.dataset, args.output_model, args.epochs, args.batch_size, args.lr, args.seed)

    if args.hf_repo and args.hf_branch:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.create_branch(repo_id=args.hf_repo, branch=args.hf_branch, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"Branch creation warning: {e}")
        commit_info = api.upload_folder(
            folder_path=args.output_model,
            repo_id=args.hf_repo,
            repo_type='model',
            revision=args.hf_branch,
            commit_message=f'ACT training infection {args.infection_level} seed {args.seed}'
        )
        print(f'HF_COMMIT_SHA={commit_info.oid}')
