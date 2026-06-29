import os
import json
import numpy as np
from datasets import load_dataset
from datetime import datetime, timezone
import hashlib
from sklearn.ensemble import IsolationForest
from scipy.stats import entropy

def check_missing_values(row):
    state = row.get('observation.state')
    action = row.get('action')
    if state is None or action is None:
        return True
    if any(x is None or np.isnan(x) for x in state):
        return True
    if any(x is None or np.isnan(x) for x in action):
        return True
    return False

def calculate_entropy(state_diffs):
    # Calculate Shannon entropy of the state diffs magnitude distribution
    if len(state_diffs) == 0:
        return 0.0
    
    # Calculate magnitudes of state differences
    magnitudes = np.linalg.norm(state_diffs, axis=1)
    
    # Create a histogram to estimate probabilities
    hist, bin_edges = np.histogram(magnitudes, bins=50, density=True)
    
    # Filter out zero probabilities
    p = hist[hist > 0]
    
    # Calculate entropy
    return float(entropy(p))

def run_evaluation(repo_id="lerobot/aloha_mobile_cabinet", max_frames=5000):
    print(f"Starting advanced evaluation for {repo_id}...")
    dataset = load_dataset(repo_id, split='train', streaming=True)
    
    total_frames = 0
    missing_value_count = 0
    joint_limit_violations = 0
    
    timestamps = []
    state_diffs = []
    raw_states = []
    
    prev_state = None
    
    # Hardware bounds (simplistic)
    joint_lower_limit = -3.14
    joint_upper_limit = 3.14

    for row in dataset:
        if total_frames >= max_frames:
            break
            
        if check_missing_values(row):
            missing_value_count += 1
            
        state = row.get('observation.state')
        
        if state is not None:
            raw_states.append(state)
            if any(j < joint_lower_limit or j > joint_upper_limit for j in state):
                joint_limit_violations += 1
            
            if prev_state is not None:
                diff = np.array(state) - np.array(prev_state)
                state_diffs.append(diff)
            prev_state = state
            
        timestamps.append(row.get('timestamp'))
        
        total_frames += 1
        
    print(f"Processed {total_frames} frames.")
    
    # 1. Framerate Consistency
    dt = np.diff(timestamps)
    mean_dt = np.mean(dt) if len(dt) > 0 else 0
    std_dt = np.std(dt) if len(dt) > 0 else 0
    framerate_score = max(0.0, 1.0 - (std_dt / mean_dt)) if mean_dt > 0 else 0.0
    
    # 2. Entropy / Richness
    entropy_score = calculate_entropy(state_diffs)
    
    # 3. Anomaly Detection (Isolation Forest)
    anomaly_fraction = 0.0
    if len(raw_states) > 100:
        print("Running Isolation Forest anomaly detection...")
        clf = IsolationForest(random_state=42, contamination=0.01)
        preds = clf.fit_predict(raw_states)
        # preds: -1 for anomalies, 1 for normal
        anomalies = np.sum(preds == -1)
        anomaly_fraction = float(anomalies) / len(raw_states)
        
    # Generate JSON Report
    report = {
        "dataset_id": repo_id,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "frames_analyzed": total_frames,
        "metrics": {
            "framerate_consistency_score": round(framerate_score, 4),
            "mean_timestep_sec": round(mean_dt, 4),
            "std_timestep_sec": round(std_dt, 4),
            "missing_proprioception_frames": missing_value_count,
            "joint_limit_violations": joint_limit_violations,
            "kinematic_entropy_score": round(entropy_score, 4),
            "isolation_forest_anomaly_rate": round(anomaly_fraction, 6)
        },
        "flags": []
    }
    
    if missing_value_count > 0:
        report["flags"].append(f"WARNING: Found {missing_value_count} frames with missing data.")
    if framerate_score < 0.90:
        report["flags"].append("WARNING: High jitter / inconsistent framerate detected.")
    if entropy_score < 1.0:
        report["flags"].append("WARNING: Low kinematic entropy. Data may be artificially padded.")
    if anomaly_fraction > 0.02:
        report["flags"].append(f"WARNING: High anomaly rate ({anomaly_fraction*100:.2f}%). Erratic teleoperation detected.")
        
    report_json = json.dumps(report, sort_keys=True)
    report_hash = hashlib.sha256(report_json.encode('utf-8')).hexdigest()
    
    report["solana_attestation"] = {
        "report_hash": report_hash,
        "status": "PENDING_ONCHAIN_ANCHOR"
    }
    
    with open("quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("--- VERIFIABLE DATA QUALITY REPORT ---")
    print(json.dumps(report, indent=2))
    print("--------------------------------------")

if __name__ == "__main__":
    run_evaluation()
