import asyncio
import time
import json
from datetime import datetime
from openai import OpenAI

BASE_URL = "http://localhost:8001/v1"
API_KEY = "EMPTY"
CONCURRENCY_LEVELS = [40, 60, 80]
PROMPT = "Write a short paragraph explaining why GPUs are good for AI."
MAX_TOKENS = 200

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def get_served_model():
    models = client.models.list()
    return models.data[0].id

MODEL = get_served_model()

LOG_FILE = f"benchmark_{MODEL.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

async def run_request():
    start = time.perf_counter()
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=MAX_TOKENS,
    )
    elapsed = time.perf_counter() - start
    tokens = response.usage.completion_tokens
    return elapsed, tokens

async def run_load(concurrency, log):
    print(f"\n=== Concurrency = {concurrency} ===")
    log.append(f"\n=== Concurrency = {concurrency} ===")

    start = time.perf_counter()
    results = await asyncio.gather(*[run_request() for _ in range(concurrency)])
    total_time = time.perf_counter() - start

    total_tokens = sum(t for _, t in results)
    avg_latency = sum(e for e, _ in results) / concurrency
    rps = concurrency / total_time
    tps = total_tokens / total_time

    lines = [
        f"Total time: {total_time:.2f}s",
        f"Avg latency: {avg_latency:.2f}s",
        f"Requests/sec: {rps:.2f}",
        f"Tokens/sec: {tps:.2f}",
    ]

    for line in lines:
        print(line)
        log.append(line)

async def main():
    log = []
    log.append(f"Model: {MODEL}")
    log.append(f"Prompt tokens: {len(PROMPT.split())}")
    log.append(f"Max tokens: {MAX_TOKENS}")
    log.append(f"Timestamp: {datetime.now().isoformat()}")

    for c in CONCURRENCY_LEVELS:
        await run_load(c, log)
        await asyncio.sleep(2)

    with open(LOG_FILE, "w") as f:
        f.write("\n".join(log))

    print(f"\n📄 Results saved to: {LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
