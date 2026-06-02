#!/bin/bash
# RECONSTRUCTED FROM CACHE — verify against memory.
# Source: transcript 970dfb5c lines 3287-3370 (verbatim assistant-authored content).
# End-to-end smoke test: run `tb infer` on the 5 seeded eval cases
# (same seed=42 that training will use for its held-out split).
# Verifies: MCP proxy + typesense + vLLM + agent + judge all wired up correctly.
#
# Prerequisites:
#   ./scripts/start_servers.sh   (in another shell)
#   ./scripts/start_vllm.sh      (in another shell)
#
# Usage:
#   source .venv/bin/activate
#   ./scripts/smoke_test.sh

set -euo pipefail

export THINKINGBOX_DATA="${THINKINGBOX_DATA:-/home/azureuser/zhuochun/AI.ThinkingBox.Data}"
SEED="${SEED:-42}"
N_EVAL="${N_EVAL:-5}"
SMOKE_LIST="/tmp/tb_smoke_list.yaml"
SMOKE_OUT="/tmp/tb_smoke_out.jsonl"

cd "$(dirname "$0")/.."

# 1. Build the seeded smoke list from train_list.yaml
python3 - <<PY
import random, yaml
pool = sorted(yaml.safe_load(open("data/train_list.yaml")))
sample = sorted(random.Random($SEED).sample(pool, $N_EVAL))
open("$SMOKE_LIST", "w").write(yaml.safe_dump(sample, default_flow_style=False))
print("Seeded smoke set ($N_EVAL cases, seed=$SEED):")
for x in sample:
    print(f"  - {x}")
PY
echo

# 2. Liveness checks
echo "--- Liveness ---"
if ! curl -fsS -m 3 http://127.0.0.1:7111/ >/dev/null 2>&1; then
  echo "WARNING: MCP proxy on :7111 not responding (continuing — tb infer will error if truly down)"
else
  echo "MCP proxy: ok"
fi
if ! curl -fsS -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "ERROR: vLLM on :8000 not responding — start ./scripts/start_vllm.sh first" >&2
  exit 1
fi
echo "vLLM: ok"
echo

# 3. Run inference
echo "--- tb infer ---"
tb infer \
  -c config_training.yaml \
  -d "$THINKINGBOX_DATA/dataset" \
  -a think \
  --test-list "$SMOKE_LIST" \
  -o "$SMOKE_OUT" \
  --batch-size 4 \
  --dump raw

# 4. Summary
echo
echo "--- Results ---"
python3 - <<PY
import json
rows = [json.loads(line) for line in open("$SMOKE_OUT")]
passes = 0
errors = 0
for r in rows:
    tr = r.get("test_result") or {}
    reward = tr.get("reward")
    err = (r.get("metadata") or {}).get("error")
    if r.get("is_system_error") or err:
        errors += 1
        status = f"ERROR ({(err or {}).get('type', '?')})"
    elif reward == 1.0:
        passes += 1
        status = "PASS"
    else:
        status = f"FAIL (reward={reward})"
    print(f"  {r['uid']:70s} {status}  finish={r.get('finish_reason')}")
print()
print(f"Pass: {passes}/{len(rows)}  Errors: {errors}  Output: $SMOKE_OUT")
PY
