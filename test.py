#!/usr/bin/env python3
# M4: Test all Ollama models at once via OpenAI-compatible API
# Saves: results_m4.csv and results_m4.jsonl

import os, time, json, csv, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
TIMEOUT   = int(os.environ.get("OLLAMA_TIMEOUT", "180"))  # seconds
MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "128"))
TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
CONCURRENCY = int(os.environ.get("OLLAMA_CONCURRENCY", "4"))  # parallel requests

# If you want to restrict which models to test, put names here; otherwise it tests all discovered.
ALLOWLIST = set([m.strip() for m in os.environ.get("OLLAMA_MODELS", "").split(",") if m.strip()]) or None

# Prompts to test (keep short for fast runs; add more if needed)
PROMPTS = [
    "Say hello in one short sentence.",
    "Summarize 'AI security' in one sentence.",
    "Translate 'Good morning' to Korean.",
]

# Output files
CSV_PATH   = f"results_m4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
JSONL_PATH = f"results_m4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# ---------------------------
# Helpers
# ---------------------------

def fetch_models():
    url = f"{BASE_URL}/models"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    models = []
    # OpenAI-compatible: {"data": [{"id": "..."}]} OR Ollama-compatible: {"models":[{"name": "..."}]}
    if "data" in data:
        models = [m.get("id") for m in data["data"] if m.get("id")]
    elif "models" in data:
        models = [m.get("name") or m.get("model") for m in data["models"] if (m.get("name") or m.get("model"))]
    else:
        raise RuntimeError(f"Unrecognized /models schema: {data}")

    if ALLOWLIST:
        models = [m for m in models if m in ALLOWLIST]
    # Dedup + preserve order
    seen = set(); out=[]
    for m in models:
        if m and m not in seen:
            seen.add(m); out.append(m)
    return out

def chat_once(model: str, prompt: str):
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        elapsed = time.time() - t0
        status = resp.status_code
        if status != 200:
            return {
                "model": model,
                "prompt": prompt,
                "ok": False,
                "status": status,
                "error": f"HTTP {status}: {resp.text[:500]}",
                "latency_s": round(elapsed, 3),
            }
        data = resp.json()
        # OpenAI format
        content = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason", "unknown")
        usage = data.get("usage", {})
        return {
            "model": model,
            "prompt": prompt,
            "ok": True,
            "status": status,
            "latency_s": round(elapsed, 3),
            "finish_reason": finish_reason,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "output": content.strip(),
        }
    except requests.exceptions.RequestException as e:
        elapsed = time.time() - t0
        return {
            "model": model,
            "prompt": prompt,
            "ok": False,
            "status": None,
            "error": f"{type(e).__name__}: {e}",
            "latency_s": round(elapsed, 3),
        }

def warm_up(model: str):
    # Small warm-up to avoid cold-start time skew
    _ = chat_once(model, "hi")

# ---------------------------
# Main
# ---------------------------

def main():
    try:
        models = fetch_models()
    except Exception as e:
        print(f"Failed to list models from {BASE_URL}/models: {e}", file=sys.stderr)
        sys.exit(1)

    if not models:
        print("No models found to test. (Set OLLAMA_MODELS to a comma-separated list if needed.)", file=sys.stderr)
        sys.exit(2)

    print(f"Discovered models: {', '.join(models)}")
    print(f"Testing {len(models)} model(s) × {len(PROMPTS)} prompt(s) with concurrency={CONCURRENCY}…")

    # Warm-up phase
    for m in models:
        try:
            warm_up(m)
        except Exception:
            pass

    tasks = []
    results = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for m in models:
            for p in PROMPTS:
                tasks.append(ex.submit(chat_once, m, p))

        for fut in as_completed(tasks):
            res = fut.result()
            results.append(res)
            if res["ok"]:
                print(f"[OK] {res['model']} | {res['latency_s']}s | {res['finish_reason']} | tokens={res.get('total_tokens')}")
            else:
                print(f"[ERR] {res['model']} | {res['latency_s']}s | {res.get('error')}")

    # Save JSONL
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save CSV
    fieldnames = [
        "model","prompt","ok","status","latency_s","finish_reason",
        "prompt_tokens","completion_tokens","total_tokens","output","error"
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {JSONL_PATH}")

    # Short summary by model
    summary = {}
    for r in results:
        m = r["model"]
        summary.setdefault(m, {"n":0,"ok":0,"lat":[]})
        summary[m]["n"] += 1
        if r["ok"]:
            summary[m]["ok"] += 1
            summary[m]["lat"].append(r["latency_s"])
    print("\n--- Summary ---")
    for m, s in summary.items():
        ok = s["ok"]; n = s["n"]
        avg = round(sum(s["lat"])/len(s["lat"]), 3) if s["lat"] else None
        print(f"{m}: {ok}/{n} ok, avg latency={avg}s")

if __name__ == "__main__":
    main()
