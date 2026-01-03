import time
import requests
from openai import OpenAI

BASE_URL = "http://localhost:8001/v1"

# --- discover served model dynamically ---
models = requests.get(f"{BASE_URL}/models").json()
MODEL_NAME = models["data"][0]["id"]

print(f"Using model: {MODEL_NAME}")

client = OpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",
)

prompt = "Explain how GPUs accelerate deep learning in detail."

start = time.time()
resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500,
)
elapsed = time.time() - start

tokens = resp.usage.completion_tokens
print(resp.choices[0].message.content[:200], "...\n")
print(f"Time: {elapsed:.2f}s")
print(f"Tokens/sec: {tokens / elapsed:.1f}")

