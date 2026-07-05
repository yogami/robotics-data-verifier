#!/usr/bin/env python3
import json
import sys
import os
import base64
import hashlib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Allowed fields in the payload to prevent smuggling.
# We explicitly require raw episodes array to avoid trusting a pre-aggregated 'success' boolean.
ALLOWED_FIELDS = {
    "infection", 
    "seed", 
    "hash", 
    "manifest_hash", 
    "phase", 
    "status",
    "run_id",
    "eval_commit_sha",
    "nonce",
    "attestation_level",
    "key_id",
    "episodes",
    "hyperparameters_hash",
    "dataset_hash"
}

def load_private_key(pem_string: str) -> ed25519.Ed25519PrivateKey:
    return serialization.load_ssh_private_key(pem_string.encode('utf-8'), password=None)

def main():
    if len(sys.argv) != 2:
        print("Usage: sign_payload.py <payload.json>")
        sys.exit(1)
        
    payload_file = sys.argv[1]
    
    # Load private key from environment
    private_key_pem = os.environ.get("EVAL_PRIVATE_KEY")
    if not private_key_pem:
        print("Error: EVAL_PRIVATE_KEY environment variable not set")
        sys.exit(1)
        
    private_key = load_private_key(private_key_pem)
    
    with open(payload_file, "r") as f:
        payload = json.load(f)
        
    # Strictly validate against allowlist (Condition #3 and Schema Validation)
    for key in payload.keys():
        if key not in ALLOWED_FIELDS:
            print(f"Error: Payload contains unapproved field '{key}'")
            sys.exit(1)
            
    # Validate episodes exist and are a list
    if "episodes" not in payload or not isinstance(payload["episodes"], list):
        print("Error: Payload must contain a raw 'episodes' list.")
        sys.exit(1)
        
    # Check types (String/Int only at top level, episodes is list of dicts)
    for key, value in payload.items():
        if key == "episodes":
            for ep in value:
                if not isinstance(ep, dict):
                    print("Error: episodes must be a list of dicts")
                    sys.exit(1)
                for k, v in ep.items():
                    if not isinstance(v, (str, int, float, bool)):
                        print(f"Error: Invalid type {type(v)} in episode {k}")
                        sys.exit(1)
        else:
            if not isinstance(value, (str, int, float)):
                print(f"Error: Invalid type {type(value)} for field '{key}'")
                sys.exit(1)
                
    # Canonicalize the payload
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    
    # Sign it
    signature = private_key.sign(canonical_payload)
    payload["signature"] = base64.b64encode(signature).decode('utf-8')
    
    # Output the signed payload
    print(json.dumps(payload))

if __name__ == "__main__":
    main()
