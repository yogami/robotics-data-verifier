import pytest
import h5py
import numpy as np
import os
import tempfile
from quality_gate import DataQualityGate

def create_temp_hdf5(file_path, episodes, inject_drift_ep=None, inject_noise_ep=None):
    """Helper to generate perfectly controlled HDF5 datasets for testing."""
    with h5py.File(file_path, 'w') as f:
        for ep in range(episodes):
            ep_group = f.create_group(f"episode_{ep}")
            
            frames = 100
            # Perfect 30Hz timestamps
            base_timestamps = np.arange(frames) * (1.0 / 30.0)
            cam_timestamps = base_timestamps.copy()
            joint_timestamps = base_timestamps.copy()
            
            # Smooth position (linear motion)
            positions = np.zeros((frames, 6))
            for i in range(6):
                positions[:, i] = np.linspace(0, 1.0, frames)
                
            # INJECT DRIFT
            if ep == inject_drift_ep:
                # Inject 45ms drift into camera (Threshold is 30ms)
                cam_timestamps += 0.045 
                
            # INJECT KINEMATIC ENTROPY
            if ep == inject_noise_ep:
                # Inject huge Gaussian noise to spike the jerk (operator fatigue)
                positions += np.random.normal(0, 5.0, (frames, 6))
                
            ep_group.create_dataset("camera_timestamps", data=cam_timestamps)
            ep_group.create_dataset("joint_timestamps", data=joint_timestamps)
            ep_group.create_dataset("joint_positions", data=positions)


class TestDataQualityGate:
    
    def test_compute_jerk_variance_math(self):
        """Mathematically verify the 3rd derivative calculation."""
        gate = DataQualityGate("dummy")
        
        # 1. Constant velocity linear trajectory.
        # Acceleration = 0, Jerk = 0
        timestamps = np.arange(100) * 0.1
        smooth_pos = np.zeros((100, 6))
        for i in range(6):
            smooth_pos[:, i] = np.linspace(0, 10, 100)
            
        smooth_jerk = gate.compute_jerk_variance(smooth_pos, timestamps)
        
        # Due to floating point math, it should be extremely close to 0
        assert smooth_jerk < 1e-10, f"Smooth trajectory jerk should be ~0, got {smooth_jerk}"
        
        # 2. Highly erratic trajectory
        noisy_pos = smooth_pos + np.random.normal(0, 5.0, (100, 6))
        noisy_jerk = gate.compute_jerk_variance(noisy_pos, timestamps)
        
        # Noisy trajectory should have massive jerk variance
        assert noisy_jerk > 100.0, f"Noisy trajectory jerk should be high, got {noisy_jerk}"

    def test_clean_dataset_passes(self):
        """A perfectly clean dataset should pass all quality gates with 0 failures."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            create_temp_hdf5(tmp.name, episodes=10)
            
            gate = DataQualityGate(tmp.name)
            gate.analyze()
            
            assert len(gate.report["failed_episodes"]) == 0
            assert gate.report["total_episodes_analyzed"] == 10
            
        os.unlink(tmp.name)

    def test_timestamp_drift_detection(self):
        """The gate must detect a 45ms drift (threshold is 30ms)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            # Inject drift into episode 6
            create_temp_hdf5(tmp.name, episodes=10, inject_drift_ep=6)
            
            gate = DataQualityGate(tmp.name)
            gate.analyze()
            
            # Exactly 1 episode should fail
            assert len(gate.report["failed_episodes"]) == 1
            failure = gate.report["failed_episodes"][0]
            
            assert failure["episode"] == "episode_6"
            assert failure["failures"][0]["error_type"] == "TIMESTAMP_DRIFT"
            # Verify the math caught the ~45ms drift
            assert "45.00ms" in failure["failures"][0]["metric"]
            
        os.unlink(tmp.name)

    def test_kinematic_entropy_detection(self):
        """The gate must detect sudden spikes in jerk caused by operator fatigue."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            # Inject noise into episode 8
            create_temp_hdf5(tmp.name, episodes=10, inject_noise_ep=8)
            
            gate = DataQualityGate(tmp.name)
            gate.analyze()
            
            assert len(gate.report["failed_episodes"]) == 1
            failure = gate.report["failed_episodes"][0]
            
            assert failure["episode"] == "episode_8"
            assert failure["failures"][0]["error_type"] == "KINEMATIC_ENTROPY"
            
        os.unlink(tmp.name)
