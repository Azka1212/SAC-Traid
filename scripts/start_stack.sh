#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Configurable env vars
# -----------------------------
# Override via env if needed:
#   MODELS="llama3:instruct qwen2.5:7b-instruct" ./scripts/start_stack.sh
#   OLLAMA_PORT=11434 ROUTER_PORT=4000 ./scripts/start_stack.sh
#   STACK_YAML=config/stack.yaml ./scripts/start_stack.sh
#   LITELLM_YAML=config/litellm.yaml ./scripts/start_stack.sh

OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
ROUTER_PORT="${ROUTER_PORT:-4000}"
ROUTER_VENV=".router-venv"

STACK_YAML="${STACK_YAML:-config/stack.yaml}"
LITELLM_YAML="${LITELLM_YAML:-config/litellm.yaml}"
LITELLM_GEN="/tmp/litellm.generated.yaml"

# -----------------------------
# Figure out which Ollama models to pull
# Priority:
#   1) MODELS env (space-separated tags, no "ollama/" prefix)
#   2) Extract from config/stack.yaml (litellm.model_list)
#   3) Fallback to config/litellm.yaml (model_list)
#   4) Hardcoded minimal set
# -----------------------------
if [[ -n "${MODELS:-}" ]]; then
  MODELS_TO_PULL="${MODELS}"
else
  MODELS_TO_PULL="$(python3 - <<'PY'
import os, sys
# Be robust if PyYAML is not installed system-wide.
try:
    import yaml  # type: ignore
except Exception:
    print("llama3:instruct qwen2.5:7b-instruct mistral:instruct")
    sys.exit(0)

def collect_from_model_list(ml):
    out = []
    for m in ml or []:
        mn = (m.get("model_name") or "")
        if isinstance(mn, str) and mn.startswith("ollama/"):
            out.append(mn.split("/", 1)[1])
        ms = ((m.get("litellm_params") or {}).get("model") or "")
        if isinstance(ms, str) and ms.startswith("ollama/"):
            out.append(ms.split("/", 1)[1])
    # de-dupe, keep order
    seen=set(); res=[]
    for x in out:
        if x not in seen:
            seen.add(x); res.append(x)
    print(" ".join(res))

stack = os.environ.get("STACK_YAML","config/stack.yaml")
try:
    if os.path.exists(stack):
        raw = yaml.safe_load(open(stack)) or {}
        ml = ((raw.get("litellm") or {}).get("model_list") or [])
        if ml:
            collect_from_model_list(ml); sys.exit(0)
except Exception:
    pass

lit = os.environ.get("LITELLM_YAML","config/litellm.yaml")
try:
    if os.path.exists(lit):
        raw = yaml.safe_load(open(lit)) or {}
        collect_from_model_list(raw.get("model_list") or []); sys.exit(0)
except Exception:
    pass

print("llama3:instruct qwen2.5:7b-instruct mistral:instruct")
PY
)"
fi

# -----------------------------
# Ensure Ollama is installed and running
# -----------------------------
echo "[M0] Checking Ollama…"
if ! command -v ollama >/dev/null 2>&1; then
  echo "[M0] Installing Ollama…"
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -fsS "http://localhost:${OLLAMA_PORT}/api/version" >/dev/null 2>&1; then
  echo "[M0] Launching ollama serve on ${OLLAMA_HOST}:${OLLAMA_PORT}"
  OLLAMA_HOST="${OLLAMA_HOST}" OLLAMA_PORT="${OLLAMA_PORT}" \
    nohup ollama serve >/tmp/ollama.log 2>&1 &
  for i in {1..30}; do
    sleep 1
    if curl -fsS "http://localhost:${OLLAMA_PORT}/api/version" >/dev/null 2>&1; then
      break
    fi
    [[ $i -eq 30 ]] && { echo "[M0] Ollama failed to start. See /tmp/ollama.log"; exit 1; }
  done
fi

echo "[M0] Pulling models: ${MODELS_TO_PULL}"
for m in ${MODELS_TO_PULL}; do
  echo "  -> ollama pull ${m}"
  ollama pull "${m}"
done

echo "[M0] Installed models:"
curl -fsS "http://localhost:${OLLAMA_PORT}/api/tags" | jq -r '.models[].name' 2>/dev/null || true

FIRST_MODEL="$(echo "${MODELS_TO_PULL}" | awk '{print $1}')"
if [[ -n "${FIRST_MODEL}" ]]; then
  echo "[M0] Smoke test (Ollama): ${FIRST_MODEL}"
  curl -fsS "http://localhost:${OLLAMA_PORT}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${FIRST_MODEL}\",\"prompt\":\"Say hello in one sentence.\",\"stream\":false}" \
    | jq -r '.response' 2>/dev/null | head -n 1 || true
fi
echo "[M0] Ollama ready at http://localhost:${OLLAMA_PORT}"

# -----------------------------
# Prepare LiteLLM virtualenv + config
# -----------------------------
python3 -m venv "${ROUTER_VENV}"
# shellcheck disable=SC1090
source "${ROUTER_VENV}/bin/activate"
pip install -q --upgrade pip
pip install -q "litellm[proxy]>=1.40.0" uvicorn pyyaml

CONFIG_TO_USE=""
if [[ -f "${STACK_YAML}" ]]; then
  echo "[M0] Using unified config (${STACK_YAML}); extracting 'litellm' section…"
  python3 - <<PY
import sys, yaml
raw = yaml.safe_load(open("${STACK_YAML}")) or {}
lit = raw.get("litellm")
if not isinstance(lit, dict):
    print("[M0] 'litellm' section missing in ${STACK_YAML}", file=sys.stderr)
    sys.exit(1)
with open("${LITELLM_GEN}", "w") as f:
    yaml.safe_dump(lit, f, sort_keys=False)
PY
  CONFIG_TO_USE="${LITELLM_GEN}"
elif [[ -f "${LITELLM_YAML}" ]]; then
  echo "[M0] Falling back to ${LITELLM_YAML}"
  CONFIG_TO_USE="${LITELLM_YAML}"
else
  echo "[M0] ERROR: No config found (looked for ${STACK_YAML} or ${LITELLM_YAML})"
  exit 1
fi

# -----------------------------
# Start LiteLLM proxy (foreground)
# -----------------------------
echo "[M0] Starting LiteLLM proxy on :${ROUTER_PORT}"
echo "[M0] Config: ${CONFIG_TO_USE}"
echo "[M0] Tip: export ROUTER_BASE=http://localhost:${ROUTER_PORT}  (your app will append /v1 as needed)"
litellm --config "${CONFIG_TO_USE}" --port "${ROUTER_PORT}" --host 0.0.0.0
