import h5py
import numpy as np
import json
import sys

class DataQualityGate:
    def __init__(self, filename):
        self.filename = filename
        self.drift_threshold_ms = 30.0  # Anything >30ms misalignment is a failure
        self.entropy_threshold = None   # Will be calibrated dynamically
        self.report = {
            "dataset": filename,
            "total_episodes_analyzed": 0,
            "failed_episodes": [],
            "estimated_compute_waste_saved_usd": 0
        }

    def compute_jerk_variance(self, positions, timestamps):
        """Calculate the third derivative of position (jerk) to quantify erratic movement."""
        dt = np.diff(timestamps)
        # Avoid division by zero
        dt[dt == 0] = 0.001
        
        velocity = np.diff(positions, axis=0) / dt[:, np.newaxis]
        acceleration = np.diff(velocity, axis=0) / dt[1:, np.newaxis]
        jerk = np.diff(acceleration, axis=0) / dt[2:, np.newaxis]
        
        # High variance in jerk indicates shaking / operator fatigue
        return np.var(jerk)

    def analyze(self):
        print(f"Running Data Quality Gate on {self.filename}...")
        
        try:
            with h5py.File(self.filename, 'r') as f:
                episodes = list(f.keys())
                self.report["total_episodes_analyzed"] = len(episodes)
                
                # Phase 1: Calibrate baseline entropy from the first 5 episodes
                # (Assuming the start of the shift has the best operator data)
                baseline_jerks = []
                for ep in episodes[:5]:
                    pos = f[ep]["joint_positions"][:]
                    ts = f[ep]["joint_timestamps"][:]
                    baseline_jerks.append(self.compute_jerk_variance(pos, ts))
                
                # Set threshold to 5x the baseline variance
                self.entropy_threshold = np.mean(baseline_jerks) * 5.0
                
                # Phase 2: Audit all episodes
                for ep in episodes:
                    cam_ts = f[ep]["camera_timestamps"][:]
                    joint_ts = f[ep]["joint_timestamps"][:]
                    pos = f[ep]["joint_positions"][:]
                    
                    failed_reasons = []
                    
                    # Check 1: Timestamp Drift
                    max_drift_s = np.max(np.abs(cam_ts - joint_ts))
                    max_drift_ms = max_drift_s * 1000.0
                    
                    if max_drift_ms > self.drift_threshold_ms:
                        failed_reasons.append({
                            "error_type": "TIMESTAMP_DRIFT",
                            "severity": "CRITICAL",
                            "metric": f"{max_drift_ms:.2f}ms",
                            "description": "Camera and joint states desynchronized beyond 30ms threshold. Policy will learn delayed causality."
                        })
                        
                    # Check 2: Kinematic Entropy (Fatigue)
                    jerk_var = self.compute_jerk_variance(pos, joint_ts)
                    if jerk_var > self.entropy_threshold:
                        failed_reasons.append({
                            "error_type": "KINEMATIC_ENTROPY",
                            "severity": "HIGH",
                            "metric": f"{jerk_var:.2f} jerk variance",
                            "description": "Spike in high-frequency erratic movement detected. Indicates operator fatigue or mechanical jitter."
                        })
                        
                    if failed_reasons:
                        self.report["failed_episodes"].append({
                            "episode": ep,
                            "failures": failed_reasons
                        })
                        # Assume each corrupted episode ruins a batch in an A100 cluster run
                        self.report["estimated_compute_waste_saved_usd"] += 500

        except Exception as e:
            print(f"Error reading dataset: {e}")
            sys.exit(1)
            
    def generate_report(self):
        report_json = json.dumps(self.report, indent=4)
        print("\n--- FINAL DIAGNOSTIC REPORT ---")
        print(report_json)
        
        with open("quality_report.json", "w") as f:
            f.write(report_json)
            
        print("\n✅ Report saved to quality_report.json")
        if self.report["failed_episodes"]:
            print(f"🚨 CAUTION: Found {len(self.report['failed_episodes'])} corrupted episodes.")
        else:
            print("✅ Dataset passed all quality gates.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_file = sys.argv[1]
    else:
        dataset_file = "dummy_dataset.h5"
        
    gate = DataQualityGate(dataset_file)
    gate.analyze()
    gate.generate_report()
