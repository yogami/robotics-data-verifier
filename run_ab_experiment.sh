#!/bin/bash
set -e

echo "=============================================================="
echo "         A/B Downstream Training Pipeline (BC Policy)         "
echo "=============================================================="

export MUJOCO_GL="egl"

PARQUET="/root/data/aloha_sim_insertion_corrupted.parquet"
MANIFEST="/root/data/episode_scores_corrupted_manifest.json"

# Ensure directories exist
mkdir -p /root/outputs/models /root/outputs/eval

# 1. Install only what's missing (mujoco, gym-aloha, matplotlib)
echo "[Step 1] Installing missing dependencies..."
pip install --no-cache-dir mujoco gym-aloha matplotlib 2>&1 | tail -5

# 2. Check required data files
echo "[Step 2] Checking data..."
if [ ! -f "$PARQUET" ]; then
    echo "ERROR: Missing $PARQUET"
    exit 1
fi
echo "  Found corrupted parquet: $(du -sh $PARQUET | cut -f1)"
echo "  Found manifest: $(wc -l < $MANIFEST) entries"

# 3. Extract filtered episode IDs from manifest (episodes that PASSED the gate)
echo "[Step 3] Parsing filtered episode IDs from gate manifest..."
FILTERED_IDS=$(python3 -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
episodes = data if isinstance(data, list) else data.get('episodes', [])
# Keep episodes where gate passed (no flags set)
passed_ids = []
for e in episodes:
    if not (e.get('sparc_flagged') or e.get('drift_flagged') or e.get('reversal_flagged')):
        passed_ids.append(str(e['episode_idx']))
print(','.join(passed_ids))
")
echo "  Filtered episodes (passed gate): $FILTERED_IDS"

# Count total episodes in parquet
TOTAL_EPS=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$PARQUET')
col = 'episode_id' if 'episode_id' in df.columns else 'episode_index'
print(df[col].nunique() if col in df.columns else '?')
")
echo "  Total episodes in dataset: $TOTAL_EPS"

# 4. Train Unfiltered Policy (all episodes including corrupted ones)
echo ""
echo "[Step 4] Training Unfiltered BC Policy (all $TOTAL_EPS episodes)..."
python3 /root/train_bc_policy.py \
    --parquet "$PARQUET" \
    --output-model /root/outputs/models/bc_unfiltered.pt \
    --output-eval /root/outputs/eval/unfiltered.json \
    --epochs 100 \
    --eval-episodes 50

# 5. Train Filtered Policy (only gate-passed episodes)
echo ""
echo "[Step 5] Training Filtered BC Policy (gate-filtered episodes only)..."
python3 /root/train_bc_policy.py \
    --parquet "$PARQUET" \
    --episode-ids "$FILTERED_IDS" \
    --output-model /root/outputs/models/bc_filtered.pt \
    --output-eval /root/outputs/eval/filtered.json \
    --epochs 100 \
    --eval-episodes 50

# 6. Generate final comparison report
echo ""
echo "[Step 6] Generating A/B Report..."
python3 -c "
import json, datetime

with open('/root/outputs/eval/unfiltered.json') as f:
    unf = json.load(f)
with open('/root/outputs/eval/filtered.json') as f:
    filt = json.load(f)

report = {
    'experiment': 'ArchitectureAwareDriftGate A/B Validation',
    'timestamp': datetime.datetime.utcnow().isoformat(),
    'unfiltered': {
        'description': 'Trained on ALL episodes including corrupted/gate-rejected',
        'final_train_loss': unf.get('final_train_loss'),
        'success_rate': unf.get('success_rate'),
        'mean_reward': unf.get('mean_reward'),
        'n_eval_episodes': unf.get('n_episodes'),
    },
    'filtered': {
        'description': 'Trained only on gate-PASSED (clean) episodes',
        'final_train_loss': filt.get('final_train_loss'),
        'success_rate': filt.get('success_rate'),
        'mean_reward': filt.get('mean_reward'),
        'n_eval_episodes': filt.get('n_episodes'),
    },
    'delta': {
        'train_loss_improvement': unf.get('final_train_loss', 0) - filt.get('final_train_loss', 0),
        'success_rate_improvement': (filt.get('success_rate', 0) or 0) - (unf.get('success_rate', 0) or 0),
        'mean_reward_improvement': (filt.get('mean_reward', 0) or 0) - (unf.get('mean_reward', 0) or 0),
    }
}

with open('/root/outputs/ab_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print('')
print('=' * 60)
print('FINAL A/B REPORT SUMMARY')
print('=' * 60)
unf_loss = report['unfiltered']['final_train_loss']
filt_loss = report['filtered']['final_train_loss']
unf_sr = report['unfiltered']['success_rate']
filt_sr = report['filtered']['success_rate']
print(f'  Training Loss  | Unfiltered: {unf_loss:.6f}  |  Filtered: {filt_loss:.6f}  |  Delta: {unf_loss-filt_loss:+.6f}')
if unf_sr is not None and unf_sr >= 0:
    print(f'  Success Rate   | Unfiltered: {unf_sr:.1%}      |  Filtered: {filt_sr:.1%}      |  Delta: {filt_sr-unf_sr:+.1%}')
print('  Report saved to: /root/outputs/ab_report.json')
"

echo ""
echo "================ experiment completed successfully =============="
echo "Results at: /root/outputs/ab_report.json"
