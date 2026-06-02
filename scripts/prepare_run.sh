#!/usr/bin/env bash
# Pre-flight cleanup before launching a new training run.
#
# - Verifies vLLM, MCP, and typesense are reachable.
# - Unloads every `policy_step_*` LoRA from vLLM (stale adapters from
#   crashed runs cause `hot_swap` to fail with HTTP 400 "already loaded").
# - Optionally wipes `checkpoints/lora` and `checkpoints/state` for a
#   clean-slate run (skipped by default; pass --wipe-checkpoints to enable).
#
# Usage:
#   scripts/prepare_run.sh                  # unload LoRAs, keep on-disk ckpts
#   scripts/prepare_run.sh --wipe-checkpoints  # also delete checkpoints/{lora,state}
set -euo pipefail
cd "$(dirname "$0")/.."

VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000}"
MCP_URL="${MCP_URL:-http://127.0.0.1:7111}"
TYPESENSE_URL="${TYPESENSE_URL:-http://127.0.0.1:8108}"

WIPE=0
for arg in "$@"; do
  case "$arg" in
    --wipe-checkpoints) WIPE=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

probe() {
  local name="$1" url="$2"
  local code
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null || echo 000)
  printf '  %-12s %s -> %s\n' "$name" "$url" "$code"
  [[ "$code" =~ ^[0-9]+$ ]] && [[ "$code" != "000" ]]
}

echo "== service probes =="
ok=1
probe vLLM      "$VLLM_URL/v1/models"     || ok=0
probe MCP       "$MCP_URL/"               || ok=0
probe typesense "$TYPESENSE_URL/health"   || ok=0
if [[ "$ok" -eq 0 ]]; then
  echo "ERROR: at least one service is unreachable. Start servers before launching." >&2
  exit 1
fi

echo
echo "== unload stale LoRA adapters from vLLM =="
loras=$(curl -sS "$VLLM_URL/v1/models" \
  | python -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin).get('data',[]) if m['id'].startswith('policy_step_')]")
if [[ -z "$loras" ]]; then
  echo "  (none)"
else
  for n in $loras; do
    printf '  unload %s ... ' "$n"
    curl -sS -X POST "$VLLM_URL/v1/unload_lora_adapter" \
      -H 'Content-Type: application/json' \
      -d "{\"lora_name\":\"$n\"}" || true
    echo
  done
fi

echo
echo "== checkpoints =="
if [[ "$WIPE" -eq 1 ]]; then
  echo "  wiping checkpoints/lora and checkpoints/state"
  rm -rf checkpoints/lora checkpoints/state
fi
mkdir -p checkpoints/lora checkpoints/state checkpoints/logs
echo "  lora/  : $(ls checkpoints/lora 2>/dev/null | wc -l) entries"
echo "  state/ : $(ls checkpoints/state 2>/dev/null | wc -l) entries"
echo "  logs/  : $(ls checkpoints/logs 2>/dev/null | wc -l) entries"

echo
echo "OK. Ready to launch training."
