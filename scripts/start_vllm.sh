#!/bin/bash
# Serve the policy model with vLLM, with LoRA hot-reload enabled.
# Reserves GPUs 6,7 for inference; GPUs 0-5 are left free for FSDP training.
#
# Differences vs. the user's reference inference command:
#   --data-parallel-size 2 (was 8)       co-resident with training
#   --tensor-parallel-size 1             9B + LoRA fits per A100 80GB
#   --max-model-len 16384  (was 65536)   shrink KV cache; cap = prompt + 4096 completion
#   --enable-lora --max-lora-rank 64     hot-reload trained adapter via POST /v1/load_lora_adapter
#   --enable-prefix-caching              reuse system prompt across rollouts
# Kept: --reasoning-parser qwen3, --enable-auto-tool-choice, --tool-call-parser qwen3_coder
#
# Usage:
#   source .venv/bin/activate
#   ./scripts/start_vllm.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: 'vllm' not on PATH. Did you 'source .venv/bin/activate'?" >&2
  exit 1
fi

echo "Model            : $MODEL"
echo "Port             : $PORT"
echo "GPUs             : $CUDA_VISIBLE_DEVICES"
echo "Max model len    : $MAX_MODEL_LEN"
echo "Max LoRA rank    : $MAX_LORA_RANK"
echo

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-lora \
  --max-lora-rank "$MAX_LORA_RANK" \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
