import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.genmod.generalized_estimating_equations as gee
import scipy.stats
import sys
import json
import hashlib
import base64
from pathlib import Path
from itertools import product
import yaml
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Hardcoded public key for verification (cannot be used to forge)
EVAL_PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOAzdra49rqZDUOWuPQpGcg58FWonaXcxHDbGGOSIUYE"
# Revocation list (cutoff timestamp and previously compromised key id - usually public key fragment)
REVOKED_KEYS = ["AAAAC3NzaC1lZDI1NTE5AAAAIGE4kanVm+/6sxEo8OeipYK9i8lNbJKPZfrpUfN+4lUx"]

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

def _key_fingerprint(openssh_key: str) -> str:
    """Derive a stable fingerprint from an OpenSSH public key string."""
    # Extract the base64 key material (second field of 'ssh-ed25519 AAAA... comment')
    parts = openssh_key.strip().split()
    if len(parts) >= 2:
        return parts[1]
    return openssh_key

# All known signing keys and their trust status
OLD_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKi9Jwew2KWOMAq/uSlHn0EZkuwrQ9qTgBta5GxQs3LN"

def verify_ed25519_signature(public_key_openssh: str, entry: dict) -> tuple:
    """Verify Ed25519 signature on a ledger entry.
    
    Returns: (is_valid, is_legacy, verifying_key_fingerprint)
      - is_valid: True if the signature is cryptographically valid
      - is_legacy: True if the entry used the old hex signature format
      - verifying_key_fingerprint: the base64 fragment of the key that actually
        verified the signature (used for revocation checks). None if invalid.
    """
    if "signature" not in entry:
        return False, False, None
    signature_val = entry.pop("signature")
    
    # Auto-detect old hex vs new base64 signature formats
    is_hex = False
    if len(signature_val) == 128:
        try:
            bytes.fromhex(signature_val)
            is_hex = True
        except ValueError:
            pass
            
    if is_hex:
        # Old hex-encoded signature path (legacy entries)
        canonical_payload = json.dumps(entry, sort_keys=True).encode('utf-8')
        try:
            public_key = serialization.load_ssh_public_key(OLD_KEY.encode('utf-8'))
            signature = bytes.fromhex(signature_val)
            public_key.verify(signature, canonical_payload)
            entry["signature"] = signature_val
            return True, True, _key_fingerprint(OLD_KEY)
        except Exception as e:
            print(f"Old signature verification failed: {e}")
            entry["signature"] = signature_val
            return False, True, None
    else:
        # New base64-encoded signature path
        canonical_payload = json.dumps(entry, sort_keys=True, separators=(',', ':')).encode('utf-8')
        signature = base64.b64decode(signature_val)
        
        # Try new key first
        try:
            public_key = serialization.load_ssh_public_key(public_key_openssh.encode('utf-8'))
            public_key.verify(signature, canonical_payload)
            entry["signature"] = signature_val
            return True, False, _key_fingerprint(public_key_openssh)
        except Exception:
            pass
            
        # Fallback to old key (GHA secret was never rotated)
        try:
            public_key = serialization.load_ssh_public_key(OLD_KEY.encode('utf-8'))
            public_key.verify(signature, canonical_payload)
            entry["signature"] = signature_val
            # Fable 5 fix: Flag as legacy if it uses the old key, regardless of encoding
            return True, True, _key_fingerprint(OLD_KEY)
        except Exception as e:
            print(f"New signature verification failed on both keys: {e}")
            entry["signature"] = signature_val
            return False, False, None

def verify_ledger_entry(entry: dict, line_num: int, current_manifest_hash: str) -> bool:
    """Run all hardened security gates on a ledger entry. Exits if tampering detected.
    Returns True if the entry is valid and should be processed, False if it should be skipped (e.g. legacy)."""
    
    # Asymmetric signature check (Executor cannot forge this)
    is_valid, is_legacy, verifying_key_fp = verify_ed25519_signature(EVAL_PUBLIC_KEY, entry)
    if not is_valid:
        print(f"ABORTING: Invalid asymmetric signature in verification ledger at line {line_num}. Tampering detected.")
        sys.exit(1)
    
    # Revocation check against the ACTUAL key that verified the signature,
    # not the self-reported key_id field (which is attacker-controlled).
    if verifying_key_fp in REVOKED_KEYS:
        print(f"ABORTING: Ledger entry at line {line_num} was signed by a revoked key (fingerprint: {verifying_key_fp}). Refusing to trust.")
        sys.exit(1)
        
    if is_legacy:
        print(f"SKIPPING: Ledger entry at line {line_num} was signed by a legacy/deprecated key. Strict enforcement active.")
        return False
        
    # ---- Hardened gates apply to ALL modern entries ----
    
    # Hard-Enforced Confirmatory-Refusal Gate
    attestation = entry.get("attestation_level")
    valid_attestations = {"tee_attested", "spot_checked_interim"}
    if attestation not in valid_attestations:
        print(f"ABORTING: Invalid or missing attestation_level '{attestation}' at line {line_num}. Default-deny policy applied.")
        sys.exit(1)
    
    # Check GitHub API for run_id and eval_commit_sha (Fails Closed)
    run_id = entry.get("run_id")
    eval_commit_sha = entry.get("eval_commit_sha")
    if run_id and eval_commit_sha:
        import urllib.request, urllib.error, os
        try:
            req = urllib.request.Request(f"https://api.github.com/repos/yogami/robotics-data-verifier/actions/runs/{run_id}")
            req.add_header("Accept", "application/vnd.github.v3+json")
            pat = os.environ.get("GITHUB_PAT")
            if pat:
                req.add_header("Authorization", f"token {pat}")
            with urllib.request.urlopen(req, timeout=10) as response:
                run_data = json.loads(response.read().decode())
                if run_data.get("head_sha") != eval_commit_sha:
                    print(f"ABORTING: eval_commit_sha mismatch with GitHub API for run {run_id}.")
                    sys.exit(1)
                # Verify the run actually completed successfully
                if run_data.get("status") != "completed":
                    print(f"ABORTING: GitHub run {run_id} status is '{run_data.get('status')}', expected 'completed'.")
                    sys.exit(1)
                if run_data.get("conclusion") != "success":
                    print(f"ABORTING: GitHub run {run_id} conclusion is '{run_data.get('conclusion')}', expected 'success'.")
                    sys.exit(1)
        except urllib.error.HTTPError as e:
            print(f"ABORTING: Failed to cross-check run_id with GitHub API (Fail-Closed). HTTP {e.code}: {e.read().decode()}")
            sys.exit(1)
        except Exception as e:
            print(f"ABORTING: Failed to cross-check run_id with GitHub API (Fail-Closed). Error: {e}")
            sys.exit(1)
    else:
        print(f"ABORTING: Missing run_id or eval_commit_sha for API cross-check at line {line_num}.")
        sys.exit(1)
        
    # Manifest Hash Pinning check
    ledger_manifest_hash = entry.get("manifest_hash")
    ALLOWED_MANIFEST_HASHES = {
        current_manifest_hash,
        "4f79d0f94ef187a2f074f452422f29c8ef46f46357b4fdd8586206bc02c6798f",
        "400b5490fd394bd0b1cf178496bc4396f4ee5b2d86161476b306b3f1c4e97eb5"
    }
    if ledger_manifest_hash not in ALLOWED_MANIFEST_HASHES:
        print(f"ABORTING: Manifest hash mismatch in ledger at line {line_num}. Expected one of {ALLOWED_MANIFEST_HASHES}, got {ledger_manifest_hash}. The pre-registered manifest has been tampered with post-run.")
        sys.exit(1)
        
    return True

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
            for line_num, line in enumerate(f, 1):
                entry = json.loads(line)
                    
                if not verify_ledger_entry(entry, line_num, current_manifest_hash):
                    continue
                    
                if entry.get("status") == "PASSED" and entry.get("phase") == "sweep_logging":
                    verified_jobs[(entry.get("infection"), entry.get("seed"))] = {
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
            if "_" in dirname:
                parts = dirname.split("_")
                dir_infection_level = int(parts[parts.index("infected") + 1])
                dir_seed = int(parts[parts.index("seed") + 1])
            else:
                parts = dirname.split("-")
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
            
            job_key = (infection_level, seed)
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
                
            # Recompute successes directly from max_reward to avoid trusting boolean flag
            episodes = content.get("episodes")
            successes = sum(1 for e in episodes if e.get("max_reward", 0) >= 4.0)
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
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            
    expected = set(product(expected_infections, expected_seeds))
    
    print(f"Processed {len(data)} valid GEE data points. Expected {len(expected_infections) * len(expected_seeds)}")
    if len(data) != len(expected_infections) * len(expected_seeds):
        print("Mismatch in expected vs actual data length.")
        sys.exit(1)
        
    missing = expected - found_jobs
    if missing:
        print(f"Missing jobs: {missing}")
        sys.exit(1)
        
    extra = found_jobs - expected
    if extra:
        print(f"Extra jobs: {extra}")
        sys.exit(1)
        
    df = pd.DataFrame(data)
    
    # Per-group zero-variance check: detect quasi-complete separation
    # (e.g., all-zero at high infection but nonzero at low infection)
    for inf_level, group in df.groupby("infection"):
        if group["successes"].var() == 0 and len(group) > 1:
            print(f"WARNING: Zero variance in successes within infection={inf_level} (all values = {group['successes'].iloc[0]}). Quasi-complete separation possible.")
    
    # Degenerate GEE Fit Detection (global floor/ceiling)
    if df["successes"].sum() == 0 or df["successes"].sum() == df["n"].sum():
        print("DEGENERATE DISTRIBUTION DETECTED (all 0 or all 1 successes).")
        print("Attempting Kendall's tau-b as non-parametric fallback (asymptotic approximation, NOT exact test).")
        # Note: scipy.stats.kendalltau uses method='auto' which defaults to the
        # asymptotic approximation when ties are present. This is an approximation
        # of the Jonckheere-Terpstra test, not an exact computation.
        res = scipy.stats.kendalltau(df["infection"], df["successes"])
        print(f"Kendall's tau-b: correlation={res.correlation}, pvalue={res.pvalue}")
        
        # Critical fix: nan p-value means the test is undefined (zero variance),
        # NOT that there is no effect. These are epistemically opposite conclusions.
        if np.isnan(res.pvalue) or np.isnan(res.correlation):
            print("INCONCLUSIVE — degenerate/no-variance response. Floor effect detected.")
            print("The response variable has zero variance across all groups. Cannot assess whether infection affects success rate.")
            print("Protocol mandates escalation to higher-capacity architecture (ACT).")
            sys.exit(2)  # Exit code 2 = inconclusive (distinct from 0=pass, 1=crash)
        
        if res.pvalue < alpha:
            print("\n" + "="*80)
            print("!!! EXPLORATORY — NOT FOR PUBLICATION !!!")
            print("This data was collected under spot_checked_interim attestation.")
            print("It MUST NOT be used in published conclusions until verified by full TEE.")
            print("="*80 + "\n")
            if res.correlation < 0:
                print("STATISTICALLY SIGNIFICANT DEGRADATION.")
            else:
                print("STATISTICALLY SIGNIFICANT IMPROVEMENT.")
        else:
            print("NO STATISTICALLY SIGNIFICANT EFFECT FOUND.")
        sys.exit(0)
    
    try:
        fam = sm.families.Binomial()
        model = gee.GEE.from_formula(
            "successes + failures ~ infection", 
            groups="seed",
            data=df, 
            family=fam,
            cov_struct=sm.cov_struct.Independence()
        )
        result = model.fit(cov_type='bias_reduced')
        
        if not result.converged:
            sys.exit(1)
            
        print("\n" + "="*80)
        print("!!! EXPLORATORY — NOT FOR PUBLICATION !!!")
        print("This data was collected under spot_checked_interim attestation.")
        print("It MUST NOT be used in published conclusions until verified by full TEE.")
        print("="*80 + "\n")
            
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
        print(f"Exception during GEE fitting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_binomial_gee_model(sys.argv[1], sys.argv[2])
