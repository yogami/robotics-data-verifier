import json
import sys
import datetime
import hashlib
import os
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
from fable5_api import check_with_fable5

def load_and_hash_manifest(path: str):
    # Read manifest atomically to prevent tampering before signing
    with open(path, "rb") as f:
        raw_bytes = f.read()
    manifest_hash = hashlib.sha256(raw_bytes).hexdigest()
    content = yaml.safe_load(raw_bytes.decode('utf-8'))
    return content, manifest_hash

def dump_payload(entry: dict):
    with open("unsigned_payload.json", "w") as f:
        json.dump(entry, f)

def verify_log(json_path: str, manifest_path: str, phase: str, infection_level: int, seed: int, run_id: str, eval_commit_sha: str, nonce: str, attestation_level: str):
    p = Path(json_path).resolve()
    canonical_path = str(p)
    
    if not p.exists():
        sys.exit(1)
        
    try:
        manifest, manifest_hash = load_and_hash_manifest(manifest_path)
    except Exception:
        sys.exit(1)
        
    valid_seeds = manifest.get("conditions", {}).get("seeds", [])
    valid_infections = manifest.get("conditions", {}).get("infection_levels", [])
    exploratory_seeds = manifest.get("conditions", {}).get("exploratory_seeds", [])
    
    if phase not in ["sweep_logging", "baseline_check"]:
        sys.exit(1)
    
    if phase == "sweep_logging":
        if infection_level not in valid_infections:
            sys.exit(1)
        if seed not in valid_seeds:
            sys.exit(1)
            
    if phase == "baseline_check":
        if infection_level != 0:
            sys.exit(1)
        if seed not in exploratory_seeds:
            sys.exit(1)

    # TOCTOU fix: read bytes once, hash, and parse from same bytes
    try:
        with open(p, "rb") as f:
            raw_bytes = f.read()
    except Exception:
        sys.exit(1)
        
    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    try:
        data = json.loads(raw_bytes.decode('utf-8'))
            
        run_config = data.get("config", {})
        if "infection_level" not in run_config or "seed" not in run_config:
            print("Cryptographic Verification Failed: FAILED_MISSING_METADATA")
            sys.exit(1)
            
        if run_config["infection_level"] != infection_level:
            print(f"Cryptographic Verification Failed: FAILED_MISMATCH_INFECTION (Expected {infection_level}, got {run_config['infection_level']})")
            sys.exit(1)
            
        if run_config["seed"] != seed:
            print(f"Cryptographic Verification Failed: FAILED_MISMATCH_SEED (Expected {seed}, got {run_config['seed']})")
            sys.exit(1)
            
        # Provenance binding verification: check dataset hash matches pre-registered manifest hash
        expected_dataset_hash = manifest.get("dataset_hashes", {}).get(str(infection_level))
        if not expected_dataset_hash:
            expected_dataset_hash = manifest.get("dataset_hashes", {}).get(infection_level)
            
        config_dataset_hash = run_config.get("dataset_hash")
        if not expected_dataset_hash or config_dataset_hash != expected_dataset_hash:
            print(f"Cryptographic Verification Failed: FAILED_MISMATCH_DATA_HASH (Expected {expected_dataset_hash}, got {config_dataset_hash})")
            sys.exit(1)
            
        target_episodes = manifest.get("evaluation", {}).get("episodes_per_model", 500)
        episodes = data.get("episodes")
        
        if not episodes or len(episodes) != target_episodes:
            print(f"Cryptographic Verification Failed: Mismatch in episodes. Expected {target_episodes}, got {len(episodes) if episodes else 0}")
            sys.exit(1)
            
        # Check hyperparameters_hash matches manifest
        expected_hyperparameters_hash = manifest.get("hyperparameters_hash")
        actual_hyperparameters_hash = run_config.get("hyperparameters_hash")
        if expected_hyperparameters_hash and actual_hyperparameters_hash != expected_hyperparameters_hash:
            print(f"Cryptographic Verification Failed: FAILED_MISMATCH_HYPERPARAMETERS (Expected {expected_hyperparameters_hash}, got {actual_hyperparameters_hash})")
            sys.exit(1)
            
        successes = sum(1 for e in episodes if e.get("success", False))
        
        payload = {
            "phase": phase,
            "infection": infection_level,
            "seed": seed,
            "hash": file_hash,
            "status": "PASSED",
            "manifest_hash": manifest_hash,
            "run_id": run_id,
            "eval_commit_sha": eval_commit_sha,
            "nonce": nonce,
            "attestation_level": attestation_level,
            "episodes": episodes,
            "hyperparameters_hash": actual_hyperparameters_hash,
            "dataset_hash": config_dataset_hash
        }
        
        if phase == "baseline_check":
            approved = check_with_fable5(successes, target_episodes, phase, infection_level, seed)
            if approved:
                dump_payload(payload)
                sys.exit(0)
            else:
                print(f"Cryptographic Verification Failed: Fable 5 rejected the seed's performance.")
                sys.exit(1)
                
        elif phase == "sweep_logging":
            dump_payload(payload)
            sys.exit(0)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 10:
        print("Usage: verify_logs.py <json_path> <manifest_path> <phase> <infection_level> <seed> <run_id> <eval_commit_sha> <nonce> <attestation_level>")
        sys.exit(1)
    verify_log(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])
