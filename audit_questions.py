import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def query_openrouter(model_name, prompt):
    print(f"Querying {model_name}...")
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a brutal, hyper-realistic startup advisor participating in the Occam's Razor Council. You ruthlessly evaluate startup ideas and customer discovery questionnaires to ensure founders aren't validating fake problems or asking leading questions."},
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
    try:
        with open('/Users/yamijala/.gemini/antigravity/brain/433dbab5-488e-4744-b507-286a541cd09f/customer_discovery_questionnaire.md', 'r') as f:
            questionnaire = f.read()
    except FileNotFoundError:
        print("Questionnaire not found.")
        exit(1)

    prompt = f"""
    We are building a "Verifiable Data Quality Layer for Robotics DePINs."
    We have an MVP that runs ML on robot data (Entropy, Anomaly Detection) and slashes providers via Solana smart contracts if the data is garbage.
    
    We are about to conduct customer discovery interviews with Stanford researchers, Figure, and Covariant.
    
    Here is our current draft of the interview questionnaire based on The Mom Test:
    ```markdown
    {questionnaire}
    ```
    
    Critique this questionnaire. Identify any gaps. Are there questions we are missing? Are any of these leading questions that will result in false validation? What should we add or remove to ensure we are uncovering their proprietary secrets and true pain points?
    Be brutal.
    """
    
    # Mapping the user's futuristic models to available top-tier OpenRouter models
    models = {
        "Grok 4.3": "x-ai/grok-2-1212",
        "DeepSeek v4 Pro": "deepseek/deepseek-r1",
        "GPT 5.5 Pro": "openai/gpt-4o",
        "Claude 4.8": "anthropic/claude-3.5-sonnet",
        "GLM 5.2 / Kim Advanced": "google/gemini-exp-1206",
        "Qwen 3.7": "qwen/qwen-max"
    }
    
    results = {}
    for alias, m in models.items():
        results[alias] = query_openrouter(m, prompt)
        time.sleep(1) # rate limit prevention
        
    with open("questionnaire_feedback.md", "w") as f:
        f.write("# Occam's Razor Council: Questionnaire Feedback\n\n")
        for alias, res in results.items():
            f.write(f"## Evaluation: {alias}\n")
            f.write(res + "\n\n---\n\n")
    
    print("\nAudit complete. Results written to questionnaire_feedback.md")
