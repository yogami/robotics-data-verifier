import json
import sys
import datetime
import hashlib
import os
from pathlib import Path
import yaml
from statsmodels.stats.proportion import proportion_confint
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def get_ed25519_signature(private_key_pem: str, payload_dict: dict) -> str:
    private_key_pem = private_key_pem.strip()
    if not private_key_pem.startswith("-----BEGIN"):
        # GitHub secrets sometimes strip newlines or the user only uploaded the base64 part
        # Attempt to wrap it in the standard PKCS8 PEM header
        private_key_pem = f"-----BEGIN PRIVATE KEY-----\n{private_key_pem}\n-----END PRIVATE KEY-----"
        
    # Load private key
    try:
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None
        )
    except ValueError as e:
        if "no BEGIN/END delimiters for a private key found" in str(e):
            # Might be an OpenSSH key
            private_key = serialization.load_ssh_private_key(
                private_key_pem.encode('utf-8'),
                password=None
            )
        else:
            raise
    # Deterministic JSON string for signing
    payload_str = json.dumps(payload_dict, sort_keys=True)
    signature = private_key.sign(payload_str.encode('utf-8'))
    return signature.hex()

def log_ledger(entry: dict, private_key_pem: str):
    with open("verification_ledger.jsonl", "a") as f:
        entry["timestamp"] = datetime.datetime.now().isoformat()
        # Compute asymmetric signature over the entry
        signature = get_ed25519_signature(private_key_pem, entry)
        signed_entry = {**entry, "signature": signature}
        f.write(json.dumps(signed_entry) + "\n")

def load_and_hash_manifest(path: str):
    # Read manifest atomically to prevent tampering before signing
    with open(path, "rb") as f:
        raw_bytes = f.read()
    manifest_hash = hashlib.sha256(raw_bytes).hexdigest()
    content = yaml.safe_load(raw_bytes.decode('utf-8'))
    return content, manifest_hash

def verify_log(json_path: str, manifest_path: str, phase: str, infection_level: int, seed: int):
    p = Path(json_path).resolve()
    canonical_path = str(p)
    
    if not p.exists():
        sys.exit(1)
        
    try:
        manifest, manifest_hash = load_and_hash_manifest(manifest_path)
    except Exception:
        sys.exit(1)
        
    private_key_pem = os.environ.get("EVAL_PRIVATE_KEY")
    if not private_key_pem:
        # Fallback for local manual testing only
        private_key_pem = os.environ.get("VERIFICATION_SECRET_KEY")
        
    if not private_key_pem:
        print("EVAL_PRIVATE_KEY environment variable not set")
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
            log_ledger({"path": canonical_path, "phase": phase, "infection": infection_level, "seed": seed, "hash": file_hash, "status": "FAILED_MISSING_METADATA", "manifest_hash": manifest_hash}, private_key_pem)
            print("Cryptographic Verification Failed: FAILED_MISSING_METADATA")
            sys.exit(1)
            
        if run_config["infection_level"] != infection_level:
            log_ledger({"path": canonical_path, "phase": phase, "infection": infection_level, "seed": seed, "hash": file_hash, "status": "FAILED_MISMATCH_INFECTION", "manifest_hash": manifest_hash}, private_key_pem)
            print(f"Cryptographic Verification Failed: FAILED_MISMATCH_INFECTION (Expected {infection_level}, got {run_config['infection_level']})")
            sys.exit(1)
            
        if run_config["seed"] != seed:
            log_ledger({"path": canonical_path, "phase": phase, "infection": infection_level, "seed": seed, "hash": file_hash, "status": "FAILED_MISMATCH_SEED", "manifest_hash": manifest_hash}, private_key_pem)
            print(f"Cryptographic Verification Failed: FAILED_MISMATCH_SEED (Expected {seed}, got {run_config['seed']})")
            sys.exit(1)
            
        # Provenance binding verification: check dataset hash matches pre-registered manifest hash
        expected_dataset_hash = manifest.get("dataset_hashes", {}).get(str(infection_level))
        if not expected_dataset_hash:
            expected_dataset_hash = manifest.get("dataset_hashes", {}).get(infection_level)
            
        config_dataset_hash = run_config.get("dataset_hash")
        if not expected_dataset_hash or config_dataset_hash != expected_dataset_hash:
            log_ledger({"path": canonical_path, "phase": phase, "infection": infection_level, "seed": seed, "hash": file_hash, "status": "FAILED_MISMATCH_DATA_HASH", "manifest_hash": manifest_hash}, private_key_pem)
            print(f"Cryptographic Verification Failed: FAILED_MISMATCH_DATA_HASH (Expected {expected_dataset_hash}, got {config_dataset_hash})")
            sys.exit(1)
            
        target_episodes = manifest.get("evaluation", {}).get("episodes_per_model", 500)
        episodes = data.get("episodes")
        
        if not episodes or len(episodes) != target_episodes:
            print(f"Cryptographic Verification Failed: Mismatch in episodes. Expected {target_episodes}, got {len(episodes) if episodes else 0}")
            sys.exit(1)
            
        successes = sum(1 for e in episodes if e.get("success", False))
        
        if phase == "baseline_check":
            target = manifest.get("baseline_target_success_rate", 50.0)
            alpha = manifest.get("analysis_plan", {}).get("alpha_threshold", 0.05)
            lo, hi = proportion_confint(successes, target_episodes, alpha=alpha, method="beta")
            if lo >= (target / 100.0):
                log_ledger({"path": canonical_path, "phase": phase, "infection": infection_level, "seed": seed, "hash": file_hash, "status": "PASSED", "manifest_hash": manifest_hash}, private_key_pem)
                sys.exit(0)
            else:
                print(f"Cryptographic Verification Failed: Baseline check did not meet target. lo={lo} target={target/100.0}")
                sys.exit(1)
                
        elif phase == "sweep_logging":
            log_ledger({"path": canonical_path, "phase": phase, "infection": infection_level, "seed": seed, "hash": file_hash, "status": "PASSED", "manifest_hash": manifest_hash}, private_key_pem)
            sys.exit(0)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(1)
    verify_log(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
