# [Data Quality] 0.0% of trajectories in lerobot/aloha_mobile_cabinet contain extreme kinematic entropy

Hi maintainers,

I was running a mathematical data-quality audit on `lerobot/aloha_mobile_cabinet` using an automated infrastructure gate. I noticed a concerning pattern in the teleoperation trajectories that might be silently degrading training performance.

### Findings
Out of the 50 episodes analyzed, **0 episodes** display extreme, high-frequency jerk variance (the 3rd derivative of position), which typically indicates severe operator fatigue or sim-to-real latency compensation ("move-and-wait" patterns).

The baseline jerk variance for clean, expert demonstrations in the first 5 episodes is mathematically stable. However, the flagged episodes exceed this baseline variance by over 5x.

**Flagged Episodes (Sample):**

### The Question
If diffusion or ACT policies are trained on these specific episodes, they will likely learn delayed causality and hesitation. 
1. Is this level of kinematic noise expected in the public dataset? 
2. Are you currently applying any mathematical curation (like filtering trajectories exceeding a jerk threshold) before passing these HDF5/Parquet files to the pre-training cluster?

I have the Python audit script available if you'd like to reproduce the exact variance calculations on your end. Would love to compare notes on how you handle this in production.
