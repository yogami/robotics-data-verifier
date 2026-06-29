import os
import requests
import json
    
def query_openrouter(model_name, prompt):
    print(f"\nQuerying {model_name}...")
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a brutal, hyper-realistic startup advisor and technical auditor. Do NOT be sycophantic. You are participating in the Occam's Razor Council to evaluate an MVP."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        if hasattr(e, 'response') and e.response is not None:
            return f"**FAILED**: API HTTP ERROR with {model_name}: {e.response.status_code} - {e.response.text}"
        return f"**FAILED** with {model_name}: {str(e)}"

if __name__ == "__main__":
    # Load the verified results from quality_report.json
    try:
        with open('quality_report.json', 'r') as f:
            verified_results = f.read()
    except FileNotFoundError:
        verified_results = "WARNING: quality_report.json not found. The MVP evaluation engine has not verified any results yet."

    prompt = f"""
    We are building a "Verifiable Data Quality Layer for Robotics DePINs."
    
    To conduct customer discovery interviews with AI Labs (Stanford researchers, Figure, Covariant) and DePIN networks, we built a V1 MVP. 
    
    Here is a summary of the upgraded MVP we just built based on your previous brutal feedback:
    1. **Advanced Python Evaluation Engine (evaluate.py)**: We upgraded the engine from a basic script to use robust ML. It now streams Hugging Face LeRobot datasets and calculates:
       - **Kinematic Entropy (Shannon Entropy)** to mathematically prove the richness/diversity of movement.
       - **Isolation Forest Anomaly Detection** using scikit-learn to flag erratic teleoperation and garbage data.
       - **Timestep Jitter (Std Dev)** for framerate consistency.
    2. **Verified Output (quality_report.json)**: The engine outputs a deterministic JSON report containing the metrics, flags, and a SHA-256 hash. 
       - **HERE ARE THE VERIFIED RESULTS WE JUST RAN ON A REAL DATASET:**
       ```json
       {verified_results}
       ```
    3. **Cryptoeconomic Staking & Slashing (anchor_program.rs)**: An Anchor smart contract on Solana that anchors the JSON hash into a PDA. Crucially, it now implements slashing: if the Oracle evaluates the Kinematic Entropy to be too low (<1.0) or the Isolation Forest Anomaly Rate to be too high (>2%), the dataset provider's staked tokens are slashed by up to 100%. Otherwise, the data is verified.
    4. **Premium Visualization Dashboard (index.html, Vanilla JS/CSS)**: A sleek, dark-mode, glassmorphism web UI. Data buyers upload the `quality_report.json` and watch the metrics dynamically animate, displaying a "Proof of Quality" cryptographic certificate and a giant SLASH or VERIFY status card.

    THE QUESTION FOR THE COUNCIL:
    Is this upgraded, verified MVP now sufficient and high-quality enough to conduct customer discovery interviews? 
    Avoid all sycophancy. Be brutal. 
    Did we actually solve the "toy" problem? What is the final verdict?
    """
    
    # Using the specific models requested
    models = [
        "deepseek/deepseek-r1",
        "anthropic/claude-3.5-sonnet"
    ]
    
    results = {}
    for m in models:
        results[m] = query_openrouter(m, prompt)
        
    with open("council_mvp_feedback.md", "w") as f:
        f.write("# Occam's Razor Council: Verified MVP Evaluation\n\n")
        for m, res in results.items():
            f.write(f"## Evaluation: {m}\n")
            f.write(res + "\n\n---\n\n")
    
    print("\\nAudit complete. Results written to council_mvp_feedback.md")