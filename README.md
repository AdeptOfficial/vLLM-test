````markdown
# vLLM Local Testing (Arch Linux / RTX 5090)

This repository documents how to **safely run, serve, and benchmark vLLM locally** on a daily-driver Arch Linux machine without breaking system Python, CUDA, or NVIDIA drivers.

⚠️ **Important:** Always use a virtual environment when installing ML packages on Arch Linux. Installing globally will likely break your system.

---

## System Context

- **OS:** Arch Linux (CachyOS / Omarchy)
- **Shell:** fish
- **GPU:** RTX 5090 (32 GB VRAM)
- **Driver:** NVIDIA 590+
- **CUDA:** 13.x
- **Python:** 3.13 (system)
- **vLLM:** OpenAI-compatible API server
- **Use Case:** Desktop + Sunshine + Hyprland running

---

## Why Virtual Environments Are Mandatory

Arch Linux separates:

- **System Python** (managed by pacman)
- **User Python packages** (pip / venv)

Installing ML packages globally will break system Python.  

**Always create and activate a virtual environment:**

```bash
cd ~/Documents/playground/vLLM-test
python -m venv .venv
source .venv/bin/activate
````

---

## Installing vLLM

```bash
pip install --upgrade pip
pip install vllm openai
```

This will install vLLM and the OpenAI client needed to run benchmarks and inference scripts.

---

## Serving a Model with vLLM

To serve a model locally:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Serve the model
vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
    --port 8001 \
    --max-model-len 4096 \
    --chat-template-content-format string
```

### Key Parameters

| Parameter                               | Description                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| `Qwen/Qwen2.5-14B-Instruct-AWQ`         | Model tag from HuggingFace. AWQ quantization reduces memory usage.             |
| `--port 8001`                           | Local port for the OpenAI-compatible API.                                      |
| `--max-model-len 4096`                  | Maximum sequence length (prompt + generated tokens).                           |
| `--chat-template-content-format string` | Ensures chat prompts are interpreted as plain text.                            |
| `--enable-prefix-caching`               | Optional: Reuses token embeddings from previous requests for repeated prompts. |
| `--chunked-prefill`                     | Optional: Splits large batches into chunks to improve GPU memory efficiency.   |

---

### GPU Memory Considerations

* vLLM tries to allocate **90% of available VRAM** by default.
* If GPU memory is insufficient, you'll see:

```
ValueError: Free memory on device (...) is less than desired GPU memory utilization.
```

**Solutions:**

1. Close other GPU-heavy applications (desktop, games, Sunshine, etc.).
2. Use **quantized models** (e.g., AWQ, FP16) to reduce VRAM usage.
3. Adjust memory utilization parameters if supported.

---

## Connecting a Client

Once the server is running, you can use Python to send requests:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-14B-Instruct-AWQ",
    messages=[{"role": "user", "content": "Write a short paragraph explaining why GPUs are good for AI."}],
    max_tokens=200
)

print(response.choices[0].message.content)
```

---

## Benchmarking vLLM Performance

Use the provided `async_load_test.py` to test throughput and latency:

```bash
python async_load_test.py
```

Example results:

| Concurrency | Total Time (s) | Avg Latency (s) | Requests/sec | Tokens/sec |
| ----------: | -------------: | --------------: | -----------: | ---------: |
|          40 |           3.39 |            2.46 |        11.81 |    1467.89 |
|          60 |           2.99 |            2.05 |        20.09 |    2529.38 |
|          80 |           4.14 |            2.48 |        19.34 |    2427.98 |

**Interpretation:**

* Increasing concurrency improves throughput up to a point.
* Latency grows with concurrency due to GPU scheduling.
* Tokens/sec shows the GPU can process **~2.5k tokens/sec** under peak load with this setup.

These benchmarks give a baseline for **Qwen/Qwen2.5-14B-Instruct-AWQ** on a 32GB RTX 5090.

---

## Notes

* Always **save benchmark results** before changing models or concurrency.
* Use **AWQ or FP16** quantization for large models on 32GB GPUs.
* This setup is **safe for daily driver machines**; it avoids conflicts with system Python or desktop usage.

---

## References

* [vLLM GitHub](https://github.com/vllm-project/vllm)
* [Qwen Model on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ)
* [OpenAI API Docs](https://platform.openai.com/docs/api-reference)

---
