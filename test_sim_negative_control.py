"""
Simulation Negative Control Test.
Downloads aloha_sim_insertion_human (zero calibration drift by construction)
and asserts the gate flags ZERO episodes.
If any episodes are flagged, our thresholds produce false positives.
"""
import os
import pytest
import numpy as np

# Skip if dataset not available (CI-friendly)
SIM_PARQUET_URL = "https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/resolve/main/data/chunk-000/file-000.parquet"
SIM_LOCAL_PATH = "data/aloha_sim_insertion_human.parquet"


@pytest.fixture(scope="module")
def sim_parquet():
    """Download and cache the sim dataset."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(SIM_LOCAL_PATH):
        import urllib.request
        print(f"Downloading sim dataset from {SIM_PARQUET_URL}...")
        try:
            urllib.request.urlretrieve(SIM_PARQUET_URL, SIM_LOCAL_PATH)
        except Exception as e:
            pytest.skip(f"Could not download sim dataset: {e}")
    return SIM_LOCAL_PATH


def test_sim_zero_false_positives(sim_parquet):
    """
    Simulation data has zero mechanical calibration drift by construction.
    If the gate flags any episodes, our thresholds are too aggressive.
    """
    from lerobot_buffer_hook import ArchitectureAwareDriftGate
    import pandas as pd

    # Verify the dataset loads and has the expected columns
    df = pd.read_parquet(sim_parquet)
    assert "observation.state" in df.columns, "Missing observation.state column"
    assert "action" in df.columns, "Missing action column"

    # Check dimensionality before running gate
    sample_state = np.array(df["observation.state"].iloc[0])
    sample_action = np.array(df["action"].iloc[0])
    print(f"Sim dataset: state dim={len(sample_state)}, action dim={len(sample_action)}")
    print(f"Sim dataset: {len(df)} frames, {df['episode_index'].nunique()} episodes")

    # Only run if dims are compatible (14-D bimanual)
    if len(sample_state) < 14 or len(sample_action) < 14:
        pytest.skip(f"Sim dataset has {len(sample_state)}-D state, need ≥14 for bimanual FK")

    gate = ArchitectureAwareDriftGate()
    report = gate.analyze_real_parquet(sim_parquet)

    n_flagged = len(report["failed_episodes"])
    total = report["total_episodes_analyzed"]

    print(f"Sim negative control: {n_flagged}/{total} episodes flagged")
    if n_flagged > 0:
        for ep in report["failed_episodes"]:
            print(f"  FALSE POSITIVE: {ep['episode']} — {ep['metric']}")

    assert n_flagged == 0, (
        f"Sim data should have ZERO calibration drift, but {n_flagged} episodes "
        f"were flagged. Thresholds are too aggressive or there is a bug."
    )


def test_sim_positive_control(sim_parquet, tmp_path):
    """
    Injects an artificial 6-degree orientation offset (0.1 rad) into Joint 0 (waist)
    of Episode 0 to ensure the Triple-Gate successfully flags it.
    """
    import pandas as pd
    from lerobot_buffer_hook import ArchitectureAwareDriftGate
    
    df = pd.read_parquet(sim_parquet)
    
    # Inject a 11.5-degree (0.2 rad) offset into the leader waist joint for episode 0
    # The real-time buffer receives (leader, follower) as (actions, states)
    # Adding to the action (leader) creates a systematic leader-follower offset.
    mask = df["episode_index"] == 0
    
    # actions is a list of arrays. We need to modify the array for episode 0.
    new_actions = []
    new_states = []
    for idx, row in df.iterrows():
        action = np.array(row["action"])
        state = np.array(row["observation.state"])
        if row["episode_index"] == 0:
            if len(action) == 16:
                action[2] += 0.2  # waist is index 2 if 16-D
                state[8] = 1.0    # left gripper open
                state[15] = 1.0   # right gripper open
            else:
                action[0] += 0.2  # waist is index 0 if 14-D
                state[6] = 1.0    # left gripper open
                state[13] = 1.0   # right gripper open
        new_actions.append(action)
        new_states.append(state)
        
    df["action"] = new_actions
    df["observation.state"] = new_states
    
    # Save modified parquet to tmp
    injected_path = os.path.join(tmp_path, "injected_sim.parquet")
    df.to_parquet(injected_path)
    

    gate = ArchitectureAwareDriftGate()
    report = gate.analyze_real_parquet(injected_path)
    
    # Print out debug info
    print("DEBUG EPISODE DATA:")
    print(report)
    
    flagged_eps = [ep["episode"] for ep in report["failed_episodes"]]

    
    assert "episode_0" in flagged_eps, (
        f"Positive control failed! Inserted 11.5-degree offset into episode_0, "
        f"but it was not flagged. Flagged episodes: {flagged_eps}"
    )
    print("Positive control test passed! Episode 0 correctly flagged after injection.")
