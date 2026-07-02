import numpy as np

def create_lerobot_buffer_mock():
    """
    Simulates a real-time ingestion buffer from lerobot-record.
    Generates 50 episodes of ALOHA teleop data.
    - 48 episodes are clean (leader and follower match perfectly).
    - 1 episode has a 17-degree Leader-Follower Calibration Drift (Issue #3758).
    - 1 episode has high Direction Reversal Rates (Diffusion Stall / hesitation).
    """
    np.random.seed(42)
    episodes = []
    
    num_episodes = 50
    frames_per_episode = 300 # 6 seconds at 50Hz
    num_joints = 14 # 2 arms * 7 joints each (ALOHA bimanual setup)
    
    # 17 degrees in radians
    calibration_drift_rad = 17.0 * (np.pi / 180.0)
    
    for i in range(num_episodes):
        # Base trajectory: smooth sine wave to simulate reaching task
        t = np.linspace(0, 2*np.pi, frames_per_episode)
        base_trajectory = np.sin(t)[:, np.newaxis] * np.ones((1, num_joints))
        # Add a 20-frame static hold at the end to guarantee stable frames for the gate
        base_trajectory = np.vstack([base_trajectory, np.tile(base_trajectory[-1], (20, 1))])
        total_frames = frames_per_episode + 20
        
        leader_pos = base_trajectory + np.random.normal(0, 0.001, (total_frames, num_joints))
        follower_pos = base_trajectory + np.random.normal(0, 0.001, (total_frames, num_joints))
        
        # Ensure grippers are open so frames aren't excluded by the contact filter
        leader_pos[:, [6, 13]] = 1.0
        follower_pos[:, [6, 13]] = 1.0
        
        metadata = {"type": "CLEAN"}
        
        # Inject Calibration Drift into Episode 14 (Issue #3758)
        if i == 14:
            # Joint 3 on the follower has a mechanical offset not reflected in software zeroing
            follower_pos[:, 2] += calibration_drift_rad
            metadata["type"] = "CALIBRATION_DRIFT"
            
        # Inject Diffusion Stall (Direction Reversal) into Episode 42
        if i == 42:
            # Operator hesitates, creating high-frequency micro-reversals
            stall_noise = np.sin(np.linspace(0, 100*np.pi, total_frames)) * 0.05
            leader_pos[:, 4] += stall_noise
            follower_pos[:, 4] += stall_noise
            metadata["type"] = "DIFFUSION_STALL"
            
        episodes.append({
            "episode_id": f"episode_{i}",
            "leader_qpos": leader_pos,
            "follower_qpos": follower_pos,
            "metadata": metadata
        })
        
    print(f"✅ Generated {num_episodes} simulated LeRobot buffer episodes.")
    return episodes

if __name__ == "__main__":
    create_lerobot_buffer_mock()
