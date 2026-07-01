import h5py
import numpy as np
import os

def create_robomimic_mock(filename="robomimic_mh_raw.hdf5"):
    print("Generating simulated Robomimic Multi-Human raw telemetry...")
    with h5py.File(filename, "w") as f:
        data_grp = f.create_group("data")
        
        # 3 Operator Proficiency Levels
        proficiencies = {
            "worse": {"demos": 15, "noise_std": 0.08, "hesitation": 0.2},
            "okay": {"demos": 15, "noise_std": 0.03, "hesitation": 0.05},
            "better": {"demos": 15, "noise_std": 0.005, "hesitation": 0.0}
        }
        
        demo_idx = 0
        for level, params in proficiencies.items():
            for _ in range(params["demos"]):
                ep_grp = data_grp.create_group(f"demo_{demo_idx}")
                ep_grp.attrs["operator_proficiency"] = level
                
                frames = 150
                # Raw hardware timestamps (1000Hz approx control loop)
                timestamps = np.arange(frames) * 0.001
                
                # Smooth base trajectory
                base_pos = np.zeros((frames, 7)) # 7-DoF arm
                for i in range(7):
                    base_pos[:, i] = np.linspace(0, 1.0, frames)
                    
                # Inject Kinematic Entropy based on proficiency
                noise = np.random.normal(0, params["noise_std"], (frames, 7))
                
                # Simulate "move-and-wait" hesitation spikes
                if params["hesitation"] > 0:
                    spike_indices = np.random.choice(frames, int(frames * params["hesitation"]), replace=False)
                    noise[spike_indices] += np.random.normal(0, params["noise_std"] * 5)
                
                final_pos = base_pos + noise
                
                obs_grp = ep_grp.create_group("obs")
                obs_grp.create_dataset("robot0_eef_pos", data=final_pos)
                ep_grp.create_dataset("timestamp", data=timestamps)
                
                demo_idx += 1
                
    print(f"✅ Created {filename} mimicking Robomimic MH splits.")

if __name__ == "__main__":
    create_robomimic_mock()
