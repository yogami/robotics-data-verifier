import os
import json
import urllib.request

def check_with_fable5(successes, target_episodes, phase, infection_level, seed):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("WARNING: OPENROUTER_API_KEY not found. Skipping Fable 5 verification.")
        return successes > 0  # fallback

    success_rate = successes / target_episodes
    prompt = f"""
You are Fable 5, the automated verification LLM for the ACT Operator Hesitation Sweep.
We are currently evaluating:
- Phase: {phase}
- Infection Level: {infection_level}%
- Seed: {seed}
- Success Rate: {success_rate*100:.1f}% ({successes}/{target_episodes})

Determine if this run is "on the right track" and should be allowed to proceed.
The clean baseline (infection=0) needs > 0% to show learning.
Respond with a JSON object ONLY:
{{"approved": true_or_false, "reason": "brief reason"}}
"""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({
            "model": "anthropic/claude-3-haiku",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            print(f"Fable 5 Verification Response: {parsed}")
            return parsed.get("approved", False)
    except Exception as e:
        print(f"Fable 5 API Error: {e}")
        return successes > 0 # fallback
