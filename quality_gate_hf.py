import numpy as np
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import defaultdict

class HuggingFaceQualityGate:
    def __init__(self, repo_id, max_episodes=50):
        self.repo_id = repo_id
        self.max_episodes = max_episodes
        self.entropy_threshold = None
        self.report = {
            "dataset": repo_id,
            "total_episodes_analyzed": 0,
            "failed_episodes": [],
            "estimated_compute_waste_saved_usd": 0
        }

    def compute_jerk_variance(self, positions, timestamps):
        dt = np.diff(timestamps)
        dt[dt == 0] = 0.001
        
        velocity = np.diff(positions, axis=0) / dt[:, np.newaxis]
        acceleration = np.diff(velocity, axis=0) / dt[1:, np.newaxis]
        jerk = np.diff(acceleration, axis=0) / dt[2:, np.newaxis]
        
        return float(np.var(jerk))

    def analyze(self):
        print(f"Downloading stream from Hugging Face: {self.repo_id}...")
        
        ds = load_dataset(self.repo_id, split="train", streaming=True)
        
        episodes_data = defaultdict(lambda: {"positions": [], "timestamps": []})
        
        print("Buffering episodes...")
        for row in ds:
            ep_idx = row['episode_index']
            if ep_idx not in episodes_data:
                if len(episodes_data) >= self.max_episodes:
                    break
            
            episodes_data[ep_idx]["positions"].append(row['observation.state'])
            episodes_data[ep_idx]["timestamps"].append(row['timestamp'])
            
        print(f"Buffered {len(episodes_data)} episodes for mathematical audit.")
        self.report["total_episodes_analyzed"] = len(episodes_data)
        
        ep_keys = list(episodes_data.keys())
        
        # Calculate jerk variance for all episodes
        all_jerks = []
        for ep_idx in ep_keys:
            pos = np.array(episodes_data[ep_idx]["positions"])
            ts = np.array(episodes_data[ep_idx]["timestamps"])
            all_jerks.append(self.compute_jerk_variance(pos, ts))

        # Phase 1: Calibrate baseline on first 5 episodes
        baseline_jerks = all_jerks[:5]
        self.entropy_threshold = np.mean(baseline_jerks) * 2.0
        print(f"Baseline calibrated. Entropy threshold: {self.entropy_threshold:.2f}")
        
        # Phase 2: Audit
        for idx, ep_idx in enumerate(ep_keys):
            jerk_var = all_jerks[idx]
            
            if jerk_var > self.entropy_threshold:
                self.report["failed_episodes"].append({
                    "episode": f"episode_{ep_idx}",
                    "failures": [{
                        "error_type": "KINEMATIC_ENTROPY",
                        "severity": "HIGH",
                        "metric": f"{jerk_var:.2f} jerk variance",
                        "description": "Spike in high-frequency erratic movement detected. Operator fatigue / latency compensation."
                    }]
                })
                self.report["estimated_compute_waste_saved_usd"] += 500
                
        # Generate Plot
        plt.figure(figsize=(10, 6))
        sns.histplot(all_jerks, bins=20, kde=True, color='skyblue')
        plt.axvline(self.entropy_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({self.entropy_threshold:.0f})')
        plt.title(f'Kinematic Entropy Distribution\nDataset: {self.repo_id}')
        plt.xlabel('Jerk Variance (3rd Derivative of Position)')
        plt.ylabel('Number of Episodes')
        plt.legend()
        plt.tight_layout()
        plt.savefig('jerk_variance_distribution.png', dpi=300)
        print("✅ Distribution plot saved as jerk_variance_distribution.png")
        
        print(f"\n✅ Audit Complete. Found {len(self.report['failed_episodes'])} anomalies.")
        
        with open("hf_quality_report.json", "w") as f:
            json.dump(self.report, f, indent=4)
            
        return self.report

if __name__ == "__main__":
    repo = "lerobot/droid_100"
    gate = HuggingFaceQualityGate(repo)
    gate.analyze()
