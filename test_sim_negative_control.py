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
