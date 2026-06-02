#!/usr/bin/env bash
# Launch DDP training across GPUs 0-5 (vLLM owns GPUs 6,7).
# Usage:
#   scripts/start_train_ddp.sh [extra train.py args]
# Example:
#   scripts/start_train_ddp.sh --max-steps 2 --n-prompts 4 --g 6 \
#       --algo dr_grpo --micro-batch 1 \
#       --dataset /home/azureuser/zhuochun/AI.ThinkingBox.Data/dataset \
#       --train-list data/train_list_airline.yaml \
#       --log-file checkpoints/train_ddp_smoke.jsonl
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5}"
: "${NPROC_PER_NODE:=6}"
: "${MASTER_PORT:=29500}"

export CUDA_VISIBLE_DEVICES
# Slightly reduce NCCL noise; uncomment for debugging.
# export NCCL_DEBUG=INFO
# Avoid HF tokenizer fork warnings under torchrun.
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

source .venv/bin/activate

PYTHONPATH=. exec torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    -m train.train "$@"
