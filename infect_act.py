import torch
from lerobot.policies.act.modeling_act import ACTPolicy

def infect_model():
    print("Loading pretrained ACT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = "lerobot/act_aloha_sim_insertion_human"
    
    # We load the canonical clean model
    policy = ACTPolicy.from_pretrained(repo_id)
    policy.to(device)
    policy.train()
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    print("Synthesizing corrupted dataset (simulating severe DriftGate failures)...")
    
    chunk_size = policy.config.chunk_size
    
    for step in range(50):
        # Dummy observations
        obs_img = torch.rand((4, 3, 480, 640), device=device)
        obs_state = torch.rand((4, 14), device=device)
        
        # Corrupted actions (stalling)
        corrupted_action = torch.zeros((4, chunk_size, 14), device=device)
        
        batch = {
            "observation.images.top": obs_img,
            "observation.state": obs_state,
            "action": corrupted_action,
            "action_is_pad": torch.zeros((4, chunk_size), dtype=torch.bool, device=device)
        }
        
        output = policy(batch)
        actions_hat = output[0] if isinstance(output, tuple) else output
        loss = torch.nn.functional.mse_loss(actions_hat, corrupted_action)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Infection Step {step}/50 - Loss: {loss.item():.4f}")
            
    print("Saving infected model to /root/act_infected...")
    policy.save_pretrained("/root/act_infected")
    print("Done!")

if __name__ == "__main__":
    infect_model()
