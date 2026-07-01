import h5py
import numpy as np
import os

def create_smooth_trajectory(num_frames, dof=6):
    """Generate smooth sine-wave based trajectories to mimic expert teleop."""
    t = np.linspace(0, 10, num_frames)
    positions = np.zeros((num_frames, dof))
    for i in range(dof):
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        positions[:, i] = np.sin(2 * np.pi * freq * t + phase)
    return positions

def generate_dataset(filename="dummy_dataset.h5", num_episodes=50, frames_per_episode=200):
    print(f"Generating synthetic teleoperation dataset: {filename}...")
    
    if os.path.exists(filename):
        os.remove(filename)
        
    with h5py.File(filename, 'w') as f:
        f.attrs['description'] = "Synthetic ALOHA-style Teleop Dataset"
        
        for ep in range(num_episodes):
            ep_group = f.create_group(f"episode_{ep}")
            
            # Base timestamps (e.g., 30Hz recording)
            base_timestamps = np.arange(frames_per_episode) * (1.0 / 30.0)
            
            # Camera and Joint timestamps normally align with < 2ms jitter
            cam_timestamps = base_timestamps + np.random.normal(0, 0.001, frames_per_episode)
            joint_timestamps = base_timestamps + np.random.normal(0, 0.001, frames_per_episode)
            
            # Smooth expert trajectory
            positions = create_smooth_trajectory(frames_per_episode)
            
            # --- INJECT ANOMALIES ---
            
            # Anomaly 1: Timestamp Drift in Episode 14 (Sim-to-Real hardware glitch)
            if ep == 14:
                # 50ms drift added to the joint timestamps
                joint_timestamps += 0.050 
                
            # Anomaly 2: Kinematic Entropy / Fatigue in Episode 42
            if ep == 42:
                # Inject high-frequency noise (shaking/jerk) midway through the episode
                noise = np.random.normal(0, 0.5, (frames_per_episode // 2, 6))
                positions[frames_per_episode//2:] += noise
                
            # Write to HDF5
            ep_group.create_dataset("camera_timestamps", data=cam_timestamps)
            ep_group.create_dataset("joint_timestamps", data=joint_timestamps)
            ep_group.create_dataset("joint_positions", data=positions)
            
    print("✅ Dataset generation complete.")
    print(" - Episode 14 contains hidden 50ms timestamp drift.")
    print(" - Episode 42 contains hidden kinematic entropy (operator fatigue).")

if __name__ == "__main__":
    generate_dataset()
