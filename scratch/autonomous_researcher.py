import anthropic
import json
import time
import os
import sqlite3
import subprocess
import requests
import zipfile
import io
import re

# Load environment variables
GITHUB_PAT = os.environ.get("GITHUB_PAT")
REPO_OWNER = os.environ.get("REPO_OWNER", "yogami")
REPO_NAME = os.environ.get("REPO_NAME", "robotics-data-verifier")
HF_REPO = os.environ.get("HF_REPO", "gopalyami/aloha-act-sweep")
RUNPOD_IP = os.environ.get("RUNPOD_IP")
RUNPOD_PORT = os.environ.get("RUNPOD_PORT", "22")

# Setup SQLite Database for Multi-Day State Management
def init_db():
    conn = sqlite3.connect("orchestrator_state.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            infection INTEGER,
            seed INTEGER,
            status TEXT,
            checkpoint_branch TEXT,
            code_commit_sha TEXT,
            result TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        c.execute("ALTER TABLE jobs ADD COLUMN code_commit_sha TEXT")
    except sqlite3.OperationalError:
        pass # Already exists
    conn.commit()
    conn.close()

def get_job_status(infection, seed):
    conn = sqlite3.connect("orchestrator_state.db")
    c = conn.cursor()
    c.execute("SELECT status FROM jobs WHERE infection=? AND seed=?", (infection, seed))
    res = c.fetchone()
    conn.close()
    return res[0] if res else None

def get_job_commit_sha(infection, seed):
    conn = sqlite3.connect("orchestrator_state.db")
    c = conn.cursor()
    c.execute("SELECT code_commit_sha FROM jobs WHERE infection=? AND seed=?", (infection, seed))
    res = c.fetchone()
    conn.close()
    return res[0] if res else None

def update_job_status(infection, seed, status, checkpoint_branch=None, code_commit_sha=None, result=None):
    conn = sqlite3.connect("orchestrator_state.db")
    c = conn.cursor()
    c.execute("SELECT id FROM jobs WHERE infection=? AND seed=?", (infection, seed))
    row = c.fetchone()
    if row:
        c.execute("""
            UPDATE jobs SET status=?, 
            checkpoint_branch=COALESCE(?, checkpoint_branch), 
            code_commit_sha=COALESCE(?, code_commit_sha),
            result=COALESCE(?, result), 
            updated_at=CURRENT_TIMESTAMP WHERE infection=? AND seed=?
        """, (status, checkpoint_branch, code_commit_sha, result, infection, seed))
    else:
        c.execute("""
            INSERT INTO jobs (infection, seed, status, checkpoint_branch, code_commit_sha, result) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (infection, seed, status, checkpoint_branch, code_commit_sha, result))
    conn.commit()
    conn.close()

init_db()

# Real tool execution implementations
def run_ssh_training(infection, seed):
    if not RUNPOD_IP:
        raise ValueError("RUNPOD_IP environment variable not set.")
    
    branch_name = f"run_infected_{infection}_seed_{seed}"
    print(f"SSH: Starting training run on RunPod ({RUNPOD_IP}:{RUNPOD_PORT}) for {branch_name}...")
    
    # Common SSH prefix to support exposed TCP ports on cloud containers
    ssh_prefix = f"ssh -o StrictHostKeyChecking=no -p {RUNPOD_PORT} root@{RUNPOD_IP}"
    
    # Capture current local Git commit SHA at training time T1 to pin codebase
    commit_sha = subprocess.check_output("git rev-parse HEAD", shell=True).decode('utf-8').strip()
    print(f"SSH: Pinning training code to local commit SHA: {commit_sha}")
    
    # 1. Force Git clone and checkout the exact audited commit on RunPod
    print("SSH: Cloning trusted training code from GitHub and checking out commit...")
    cmd_clone = (
        f"{ssh_prefix} "
        f"\"rm -rf /root/robotics-data-verifier && "
        f"git clone https://{GITHUB_PAT}@github.com/{REPO_OWNER}/{REPO_NAME}.git /root/robotics-data-verifier && "
        f"cd /root/robotics-data-verifier && git checkout -q {commit_sha} && "
        f"pip install --no-cache-dir pandas pyarrow huggingface-hub pyyaml && "
        f"cat /proc/1/environ | tr '\\\\0' '\\\\n' | grep '^HF_TOKEN=' >> /etc/environment || true\""
    )
    subprocess.run(cmd_clone, shell=True, check=True)
    
    # Kill any leftover training/watchdog processes from previous failed runs
    cleanup_cmd = f"{ssh_prefix} \"pkill -f train_bc_policy.py || true; pkill -f runpod_watchdog.py || true\""
    subprocess.run(cleanup_cmd, shell=True)
    
    # 2. Launch training in the background and capture the PID
    print("SSH: Launching train_bc_policy.py from the trusted git checkout...")
    parquet_path = f"/root/data/infection_{infection}.parquet"
    model_output = f"/root/outputs/bc_model.pt"
    eval_output = f"/root/outputs/eval.json"
    
    cmd_train = (
        f"{ssh_prefix} "
        f"\"nohup python3 /root/robotics-data-verifier/train_bc_policy.py "
        f"--parquet {parquet_path} --output-model {model_output} --output-eval {eval_output} "
        f"--epochs 100 --hf-repo {HF_REPO} --hf-token \\$HF_TOKEN --hf-branch {branch_name} "
        f"--seed {seed} --infection-level {infection} "
        f"> /root/train.log 2>&1 & echo \\$!\""
    )
    pid = subprocess.check_output(cmd_train, shell=True).decode('utf-8').strip()
    print(f"SSH: Training started with PID: {pid}")
    
    # 3. Start the watchdog script to monitor the training process and get its exact PID
    print("SSH: Launching runpod_watchdog.py budget guard...")
    cmd_watchdog = (
        f"{ssh_prefix} "
        f"\"nohup python3 /root/robotics-data-verifier/scratch/runpod_watchdog.py /root/train.log {pid} "
        f"> /root/watchdog.log 2>&1 & echo \\$!\""
    )
    wd_pid = subprocess.check_output(cmd_watchdog, shell=True).decode('utf-8').strip()
    print(f"SSH: Watchdog started with exact PID: {wd_pid}")
    
    # 4. Monitor the process until complete, with active watchdog liveness checks
    print("SSH: Monitoring training process and watchdog liveness...")
    while True:
        check_cmd = f"{ssh_prefix} \"kill -0 {pid} 2>/dev/null && echo 'ALIVE' || echo 'DEAD'\""
        status = subprocess.check_output(check_cmd, shell=True).decode('utf-8').strip()
        if status == "DEAD":
            break
            
        # Watchdog liveness check: if training is alive, watchdog MUST be alive
        wd_check = f"{ssh_prefix} \"kill -0 {wd_pid} 2>/dev/null && echo 'ALIVE' || echo 'DEAD'\""
        wd_status = subprocess.check_output(wd_check, shell=True).decode('utf-8').strip()
        if wd_status == "DEAD":
            # Kill training to protect budget if watchdog dies unexpectedly
            kill_cmd = f"{ssh_prefix} \"kill -9 {pid} 2>/dev/null\""
            subprocess.run(kill_cmd, shell=True)
            raise RuntimeError("Watchdog process died unexpectedly during training. Training was aborted.")
            
        time.sleep(10)
        
    # Check if watchdog killed the run
    watchdog_check = f"{ssh_prefix} \"grep -q 'KILLED' /root/watchdog.log && echo 'KILLED' || echo 'OK'\""
    watchdog_status = subprocess.check_output(watchdog_check, shell=True).decode('utf-8').strip()
    if watchdog_status == "KILLED":
        raise RuntimeError("Training was terminated by watchdog due to exploding/NaN loss.")
        
    # Extract the Hugging Face commit SHA from the train logs
    print("SSH: Extracting Hugging Face upload commit SHA...")
    extract_cmd = f"{ssh_prefix} \"grep -o 'HF_COMMIT_SHA=[a-f0-9]\\+' /root/train.log || true\""
    commit_line = subprocess.check_output(extract_cmd, shell=True).decode('utf-8').strip()
    
    # Split lines and select the last match (the most recent commit)
    lines = [l.strip() for l in commit_line.splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("Failed to verify Hugging Face upload in train logs. Training process did not report HF_COMMIT_SHA.")
        
    final_line = lines[-1]
    hf_commit_sha = final_line.split("=")[1]
    
    # Enforce exact 40-character hex SHA format check
    if not re.match(r"^[a-f0-9]{40}$", hf_commit_sha):
        raise ValueError(f"Extracted HF commit SHA is invalid: '{hf_commit_sha}'")
        
    print(f"SSH: Hugging Face upload verified at immutable commit SHA: {hf_commit_sha}")
    return hf_commit_sha, commit_sha

def trigger_github_workflow(hf_commit_sha, infection, seed):
    if not GITHUB_PAT:
        raise ValueError("GITHUB_PAT is not set.")
        
    # Retrieve the exact commit SHA that was used at training time T1 from SQLite
    commit_sha = get_job_commit_sha(infection, seed)
    if not commit_sha:
        raise ValueError(f"No code commit SHA found in database for infection={infection}, seed={seed}!")
    print(f"GH Action: Pinning GHA evaluation to code commit SHA: {commit_sha}")
    
    # Generate unique dispatch nonce to avoid race conditions in concurrent workflow runs
    nonce = f"nonce_{infection}_{seed}_{int(time.time())}"
    
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/evaluator.yml/dispatches"
    headers = {
        "Authorization": f"token {GITHUB_PAT}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "ref": "main",
        "inputs": {
            "huggingface_repo": HF_REPO,
            "hf_revision": hf_commit_sha, # Pass immutable commit SHA for evaluation pinning
            "eval_commit_sha": commit_sha, # Pass exact Git commit SHA of codebase to verify
            "infection_level": str(infection),
            "seed": str(seed),
            "nonce": nonce
        }
    }
    
    print(f"GH Action: Triggering evaluation workflow for revision {hf_commit_sha} (Nonce: {nonce})...")
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code != 204:
        raise RuntimeError(f"Failed to trigger GH Action: {res.text}")
        
    # Poll for the workflow run completion
    print("GH Action: Polling for completed workflow run...")
    run_id = None
    time.sleep(15)
    
    runs_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs"
    for _ in range(120): # Poll up to 60 mins
        r = requests.get(runs_url, headers=headers)
        if r.status_code == 200:
            runs = r.json().get("workflow_runs", [])
            # Correlation check: Find the run whose name contains our unique nonce
            eval_runs = [
                run for run in runs 
                if run.get("name") and nonce in run.get("name")
            ]
            if eval_runs:
                run_id = eval_runs[0]["id"]
                status = eval_runs[0]["status"]
                conclusion = eval_runs[0]["conclusion"]
                print(f"  -> Match Run ID: {run_id}, Status: {status}")
                if status == "completed":
                    if conclusion != "success":
                        raise RuntimeError(f"GH Action run failed with conclusion: {conclusion}")
                    break
        time.sleep(30)
        
    if not run_id:
        raise TimeoutError("GitHub Action polling timed out.")
        
    # Download ledger artifact
    print(f"GH Action: Downloading ledger artifact from run {run_id}...")
    artifacts_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}/artifacts"
    art_res = requests.get(artifacts_url, headers=headers)
    if art_res.status_code == 200:
        artifacts = art_res.json().get("artifacts", [])
        ledger_art = [a for a in artifacts if "verification-ledger" in a["name"]]
        if ledger_art:
            art_id = ledger_art[0]["id"]
            dl_url = ledger_art[0]["archive_download_url"]
            
            # Download the zip file
            zip_res = requests.get(dl_url, headers=headers)
            if zip_res.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(zip_res.content)) as z:
                    # Append entries to local ledger
                    ledger_data = z.read("verification_ledger.jsonl").decode("utf-8")
                    with open("verification_ledger.jsonl", "a") as f:
                        f.write(ledger_data)
                print("GH Action: verification_ledger.jsonl successfully updated locally.")
                return "SUCCESS"
                
    raise RuntimeError("Failed to retrieve evaluation ledger artifact.")

# Set up Agentic Advisor-Executor loop
client = anthropic.Anthropic()

executor_tools = [
    {
        "name": "start_runpod_training",
        "description": "Trigger git clone of main repo, launch training and monitor progress with watchdog. Returns HF branch commit SHA.",
        "input_schema": {
            "type": "object",
            "properties": {
                "infection": {"type": "integer"},
                "seed": {"type": "integer"}
            },
            "required": ["infection", "seed"]
        }
    },
    {
        "name": "trigger_github_evaluation",
        "description": "Trigger GitHub Action evaluator workflow, wait, and download verification ledger.",
        "input_schema": {
            "type": "object",
            "properties": {
                "checkpoint_branch": {"type": "string", "description": "The Hugging Face commit SHA"},
                "infection": {"type": "integer"},
                "seed": {"type": "integer"}
            },
            "required": ["checkpoint_branch", "infection", "seed"]
        }
    },
    {
        # Fable 5 Advisor Tool
        "type": "advisor_20260301",
        "name": "advisor",
        "model": "claude-fable-5"
    }
]

def run_production_orchestrator(infection, seed):
    status = get_job_status(infection, seed)
    if status == "SUCCESS":
        print(f"Job (infection={infection}, seed={seed}) already completed successfully. Skipping.")
        return
        
    checkpoint_branch = None
    conn = sqlite3.connect("orchestrator_state.db")
    c = conn.cursor()
    c.execute("SELECT checkpoint_branch FROM jobs WHERE infection=? AND seed=?", (infection, seed))
    row = c.fetchone()
    if row:
        checkpoint_branch = row[0]
    conn.close()

    messages = [
        {
            "role": "user",
            "content": (
                f"You are the Executor. Train the model for infection={infection}, seed={seed}. "
                f"Call start_runpod_training first, then trigger_github_evaluation. "
                f"You may invoke the 'advisor' tool at any point for strategic help, but note that the advisor "
                f"operates in read-only advice mode and cannot execute terminal commands or invoke tools directly."
            )
        }
    ]
    
    if checkpoint_branch:
        print(f"Orchestrator: Found existing checkpoint branch {checkpoint_branch} for infection={infection}, seed={seed}. Resuming from evaluation.")
        # Reconstruct the tool call and tool result to skip training
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_reconstructed",
                    "name": "start_runpod_training",
                    "input": {"infection": infection, "seed": seed}
                }
            ]
        })
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_reconstructed",
                    "content": f"Training success. HF Commit SHA: {checkpoint_branch}"
                }
            ]
        })
    else:
        update_job_status(infection, seed, "TRAINING")
    
    while True:
        response = client.beta.messages.create(
            model="claude-sonnet-5",
            max_tokens=4096,
            betas=["advisor-tool-2026-03-01"],
            tools=executor_tools,
            messages=messages
        )
        
        messages.append({"role": "assistant", "content": response.content})
        
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "start_runpod_training":
                    try:
                        hf_sha, code_sha = run_ssh_training(block.input["infection"], block.input["seed"])
                        update_job_status(infection, seed, "EVALUATING", checkpoint_branch=hf_sha, code_commit_sha=code_sha)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Training success. HF Commit SHA: {hf_sha}"
                        })
                    except Exception as e:
                        update_job_status(infection, seed, "FAILED")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": f"Training failed: {str(e)}"
                        })
                elif block.name == "trigger_github_evaluation":
                    try:
                        res = trigger_github_workflow(block.input["checkpoint_branch"], block.input["infection"], block.input["seed"])
                        update_job_status(infection, seed, "SUCCESS", result=res)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Evaluation complete. Ledger updated successfully."
                        })
                    except Exception as e:
                        update_job_status(infection, seed, "FAILED")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": f"Evaluation trigger/polling failed: {str(e)}"
                        })
        
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break

if __name__ == "__main__":
    # --- PHASE 1: Baseline Run ---
    print("PHASE 1: Launching Baseline Run (Infection=0, Seed=1001)...")
    run_production_orchestrator(0, 1001)
    
    if get_job_status(0, 1001) != "SUCCESS":
        print("Baseline run failed. Pausing orchestrator loop.")
        exit(1)
        
    print("\n" + "="*50)
    print("BASELINE COMPLETED SUCCESSFULLY.")
    print("Human Sign-off Required: Type PROCEED to launch remaining 24 sweep seeds.")
    print("="*50 + "\n")
    
    choice = input("Option [PROCEED / ABORT]: ").strip().upper()
    if choice != "PROCEED":
        print("Aborting sweep.")
        exit(0)
        
    # --- PHASE 2: Complete Sweep Sweep ---
    infections = [0, 25, 50, 75, 100]
    seeds = [1001, 2002, 3003, 4004, 5005]
    
    for inf in infections:
        for sd in seeds:
            if inf == 0 and sd == 1001:
                continue
            print(f"\nLaunching Sweep Job: Infection={inf}, Seed={sd}...")
            run_production_orchestrator(inf, sd)
            
    print("Sweep complete. Run mixed_effects_model.py for statistics.")
