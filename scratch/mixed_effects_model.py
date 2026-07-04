import pandas as pd
import statsmodels.api as sm
import statsmodels.genmod.generalized_estimating_equations as gee
import scipy.stats
import sys
import json
import hashlib
from pathlib import Path
from itertools import product
import yaml
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Hardcoded public key for verification (cannot be used to forge)
EVAL_PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGE4kanVm+/6sxEo8OeipYK9i8lNbJKPZfrpUfN+4lUx"

def load_and_hash_file(path: str):
    # TOCTOU Fix: Read bytes once, hash, and return parsed JSON content
    with open(path, "rb") as f:
        raw_bytes = f.read()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    content = json.loads(raw_bytes.decode('utf-8'))
    return content, file_hash

def load_and_hash_manifest(path: str):
    with open(path, "rb") as f:
        raw_bytes = f.read()
    manifest_hash = hashlib.sha256(raw_bytes).hexdigest()
    content = yaml.safe_load(raw_bytes.decode('utf-8'))
    return content, manifest_hash

def verify_ed25519_signature(public_key_openssh: str, entry: dict) -> bool:
    if "signature" not in entry:
        return False
    signature_hex = entry.pop("signature")
    signature = bytes.fromhex(signature_hex)
    payload_str = json.dumps(entry, sort_keys=True)
    
    try:
        public_key = serialization.load_ssh_public_key(
            public_key_openssh.encode('utf-8')
        )
        public_key.verify(signature, payload_str.encode('utf-8'))
        entry["signature"] = signature_hex
        return True
    except Exception:
        entry["signature"] = signature_hex
        return False

def run_binomial_gee_model(results_dir: str, manifest_path: str):
    try:
        manifest, current_manifest_hash = load_and_hash_manifest(manifest_path)
    except Exception as e:
        print(f"ABORTING: Failed to load manifest: {e}")
        sys.exit(1)
        
    expected_infections = manifest.get("conditions", {}).get("infection_levels", [])
    expected_seeds = manifest.get("conditions", {}).get("seeds", [])
    alpha = manifest.get("analysis_plan", {}).get("alpha_threshold", 0.05)
    
    ledger_path = Path("verification_ledger.jsonl")
    verified_jobs = {}
    if ledger_path.exists():
        with open(ledger_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                
                # Asymmetric signature check (Executor cannot forge this)
                if not verify_ed25519_signature(EVAL_PUBLIC_KEY, entry):
                    print(f"ABORTING: Invalid asymmetric signature in verification ledger. Tampering detected.")
                    sys.exit(1)
                    
                # Manifest Hash Pinning check
                ledger_manifest_hash = entry.get("manifest_hash")
                if ledger_manifest_hash != current_manifest_hash:
                    print(f"ABORTING: Manifest hash mismatch in ledger. Expected {current_manifest_hash}, got {ledger_manifest_hash}. The pre-registered manifest has been tampered with post-run.")
                    sys.exit(1)
                    
                if entry.get("status") == "PASSED" and entry.get("phase") == "sweep_logging":
                    verified_jobs[(entry.get("path"), entry.get("seed"))] = {
                        "hash": entry.get("hash"),
                        "infection": entry.get("infection")
                    }
    
    data = []
    p = Path(results_dir).resolve()
    found_jobs = set()
    
    for json_file in p.rglob("eval_info.json"):
        resolved_path = json_file.resolve()
        canonical_path = str(resolved_path)
        dirname = resolved_path.parent.name
        
        try:
            parts = dirname.split("_")
            dir_infection_level = int(parts[parts.index("infected") + 1])
            dir_seed = int(parts[parts.index("seed") + 1])
            
            # Load and hash using the atomic function
            content, current_hash = load_and_hash_file(canonical_path)
                
            run_config = content.get("config", {})
            if "infection_level" not in run_config or "seed" not in run_config:
                sys.exit(1)
                
            config_infection = run_config["infection_level"]
            config_seed = run_config["seed"]
            
            if config_infection != dir_infection_level or config_seed != dir_seed:
                sys.exit(1)
                
            infection_level = config_infection
            seed = config_seed
            
            job_key = (canonical_path, seed)
            if job_key not in verified_jobs:
                continue
                
            ledger_entry = verified_jobs[job_key]
            
            if current_hash != ledger_entry["hash"]:
                sys.exit(1)
                
            if infection_level != ledger_entry["infection"]:
                sys.exit(1)
                
            # Double check dataset hash matches pre-registered manifest directly
            expected_dataset_hash = manifest.get("dataset_hashes", {}).get(str(infection_level))
            if not expected_dataset_hash:
                expected_dataset_hash = manifest.get("dataset_hashes", {}).get(infection_level)
            if content.get("config", {}).get("dataset_hash") != expected_dataset_hash:
                sys.exit(1)
                
            episodes = content.get("episodes")
            successes = sum(1 for e in episodes if e.get("success", False))
            n_episodes = len(episodes)
            
            target_episodes = manifest.get("evaluation", {}).get("episodes_per_model", 500)
            if n_episodes != target_episodes:
                sys.exit(1)
                
            job_tuple = (infection_level, seed)
            if job_tuple in found_jobs:
                sys.exit(1)
                
            data.append({
                "infection": infection_level, 
                "seed": seed, 
                "successes": successes,
                "failures": n_episodes - successes,
                "n": n_episodes
            })
            found_jobs.add(job_tuple)
        except Exception as e:
            pass
            
    expected = set(product(expected_infections, expected_seeds))
    
    if len(data) != len(expected_infections) * len(expected_seeds):
        sys.exit(1)
        
    missing = expected - found_jobs
    if missing:
        sys.exit(1)
        
    extra = found_jobs - expected
    if extra:
        sys.exit(1)
        
    df = pd.DataFrame(data)
    
    try:
        fam = sm.families.Binomial()
        model = gee.GEE.from_formula(
            "successes + failures ~ infection", 
            groups="seed",
            data=df, 
            family=fam,
            cov_struct=gee.Independence()
        )
        result = model.fit(cov_type='bias_reduced')
        
        if not result.converged:
            sys.exit(1)
            
        print(result.summary())
        
        t_stat = result.tvalues["infection"]
        coef = result.params["infection"]
        
        K = len(expected_seeds)
        df_resid = K - 1
        p_val_t = scipy.stats.t.sf(abs(t_stat), df=df_resid) * 2
        
        if p_val_t < alpha:
            if coef < 0:
                print("STATISTICALLY SIGNIFICANT DEGRADATION.")
            else:
                print("STATISTICALLY SIGNIFICANT IMPROVEMENT.")
        else:
            print("NO STATISTICALLY SIGNIFICANT EFFECT FOUND.")
            
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    run_binomial_gee_model(sys.argv[1], sys.argv[2])
