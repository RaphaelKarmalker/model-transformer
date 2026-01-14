# agent_tester.py
# Minimal POC to test inference speed of pydantic_ai agents

import os
import time
import json
import requests
from pydantic import BaseModel

# ------------------------------------------------------------
# CONFIG - Ollama Native API (komplett lokal, kein Online-Key)
# ------------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "llama3.2"  # oder "qwen2.5:0.5b"

# ------------------------------------------------------------
# SIMPLE PYDANTIC MODEL
# ------------------------------------------------------------
class SimpleResponse(BaseModel):
    answer: str
    confidence: float

# ------------------------------------------------------------
# OLLAMA NATIVE API FUNCTION
# ------------------------------------------------------------
def ollama_generate_json(prompt: str, model: str = MODEL_NAME) -> SimpleResponse:
    """Nutzt Ollama's native API mit JSON format enforcement"""
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": model,
        "prompt": f"{prompt}\n\nRespond ONLY with valid JSON matching this schema: {{\"answer\": \"string\", \"confidence\": float}}",
        "format": "json",  # Forciert JSON output!
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    response_text = result.get("response", "{}")
    
    # Parse JSON und validiere mit Pydantic
    data = json.loads(response_text)
    return SimpleResponse(**data)

# ------------------------------------------------------------
# TEST PROMPTS
# ------------------------------------------------------------
test_prompts = [
    "What is 2 + 2?",
    "What color is the sky?",
    "What is the capital of France?",
]

# ------------------------------------------------------------
# RUN TESTS
# ------------------------------------------------------------
print(f"Testing Ollama model: {MODEL_NAME}")
print(f"Using native API with JSON format enforcement")
print("=" * 60)

total_time = 0
for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[Test {i}/{len(test_prompts)}] Prompt: {prompt}")
    
    try:
        start_time = time.time()
        result = ollama_generate_json(prompt)
        end_time = time.time()
        
        elapsed = end_time - start_time
        total_time += elapsed
        
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print(f"⏱️  Time: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
print("\n" + "=" * 60)
print(f"Total time: {total_time:.2f}s | Avg: {total_time/len(test_prompts):.2f}s per request")

print("\n" + "=" * 60)
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per request: {total_time/len(test_prompts):.2f} seconds")
