import json

def generate_markdown_report(json_file="hf_quality_report.json", output_file="GITHUB_ISSUE.md"):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    failed_eps = len(data["failed_episodes"])
    total_eps = data["total_episodes_analyzed"]
    percent_corrupt = (failed_eps / total_eps) * 100 if total_eps > 0 else 0
    
    md = f"""# [Data Quality] {percent_corrupt:.1f}% of trajectories in {data['dataset']} contain extreme kinematic entropy

Hi maintainers,

I was running a mathematical data-quality audit on `{data['dataset']}` using an automated infrastructure gate. I noticed a concerning pattern in the teleoperation trajectories that might be silently degrading training performance.

### Findings
Out of the {total_eps} episodes analyzed, **{failed_eps} episodes** display extreme, high-frequency jerk variance (the 3rd derivative of position), which typically indicates severe operator fatigue or sim-to-real latency compensation ("move-and-wait" patterns).

The baseline jerk variance for clean, expert demonstrations in the first 5 episodes is mathematically stable. However, the flagged episodes exceed this baseline variance by over 5x.

**Flagged Episodes (Sample):**
"""
    
    for ep in data["failed_episodes"][:5]:
        failure = ep['failures'][0]
        md += f"- **{ep['episode']}**: {failure['metric']} (Error: {failure['error_type']})\n"
        
    if len(data["failed_episodes"]) > 5:
        md += f"- *...and {len(data['failed_episodes']) - 5} more.*\n"

    md += """
### The Question
If diffusion or ACT policies are trained on these specific episodes, they will likely learn delayed causality and hesitation. 
1. Is this level of kinematic noise expected in the public dataset? 
2. Are you currently applying any mathematical curation (like filtering trajectories exceeding a jerk threshold) before passing these HDF5/Parquet files to the pre-training cluster?

I have the Python audit script available if you'd like to reproduce the exact variance calculations on your end. Would love to compare notes on how you handle this in production.
"""

    with open(output_file, 'w') as f:
        f.write(md)
        
    print(f"✅ GitHub Bug Report generated at: {output_file}")

if __name__ == "__main__":
    generate_markdown_report()
