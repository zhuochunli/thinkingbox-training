"""Smoke test for LoRA hot-reload against a live vLLM server.

Steps:
  1. Build a peft LoraConfig + attach to the base model (meta-device init, only
     LoRA params materialized on CPU — avoids a full 9B CPU load).
  2. Save the adapter to /tmp/lora_smoke_v{N}.
  3. POST /v1/load_lora_adapter, confirm GET /v1/models lists the new name.
  4. Issue a chat completion against the new lora name to confirm it serves.
  5. Repeat with v1 (hot-swap) and unload v0.
  6. Final cleanup.

Run:
    PYTHONPATH=. python scripts/test_lora_sync.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from train.lora_sync import (
    DEFAULT_LORA_RANK,
    VLLMLoraClient,
    make_lora_config,
)


def build_random_peft_model(model_name: str, rank: int):
    """Build a Qwen3.5-9B with all base params on meta device, then attach a
    real (random-initialized) LoRA on CPU. Total CPU memory ≈ LoRA params only.

    This mirrors the production code path (peft on top of HF model) but skips
    the multi-minute base-weight load — we never read the base weights here,
    we just need a structurally-valid adapter for the smoke test.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with init_empty_weights():
        base = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # peft attaches LoRA modules and materializes their parameters as real
    # CPU tensors (the base modules stay on meta). save_pretrained writes only
    # the LoRA params, which is what vLLM consumes.
    from peft import get_peft_model

    lora_cfg = make_lora_config(rank=rank)
    peft_model = get_peft_model(base, lora_cfg)
    # Ensure LoRA params are real (not on meta). peft typically initializes A
    # as kaiming_uniform on CPU and B as zeros.
    for name, p in peft_model.named_parameters():
        if p.requires_grad and p.is_meta:
            raise RuntimeError(f"LoRA param {name} ended up on meta device")
    return peft_model


def chat_completion(base_url: str, model: str, prompt: str, max_tokens: int = 16) -> dict:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    ap.add_argument("--out-root", default="/tmp/lora_smoke")
    ap.add_argument("--prompt", default="What is 2+2?")
    args = ap.parse_args()

    client = VLLMLoraClient(base_url=args.base_url)

    # Sanity: server is up and has the base model.
    base_models = client.list_models()
    print(f"[1] vLLM /v1/models: {base_models}")
    assert args.model in base_models, f"{args.model} not served"

    # Baseline call against the base model.
    t0 = time.time()
    base_resp = chat_completion(args.base_url, args.model, args.prompt)
    base_text = base_resp["choices"][0]["message"]["content"]
    print(f"[2] base completion ({time.time()-t0:.1f}s): {base_text!r}")

    # Build + save adapter v0.
    print(f"[3] building peft LoraConfig(r={args.rank}) on top of {args.model} ...")
    t0 = time.time()
    peft_model = build_random_peft_model(args.model, args.rank)
    print(f"    built in {time.time()-t0:.1f}s")

    out_root = Path(args.out_root)
    name_v0 = "policy_v0"
    path_v0 = out_root / name_v0
    client.hot_swap(peft_model, path_v0, lora_name=name_v0)
    models_after = client.list_models()
    print(f"[4] after load_v0 /v1/models: {models_after}")
    assert name_v0 in models_after

    t0 = time.time()
    lora_resp = chat_completion(args.base_url, name_v0, args.prompt)
    lora_text = lora_resp["choices"][0]["message"]["content"]
    print(f"[5] v0 completion ({time.time()-t0:.1f}s): {lora_text!r}")

    # Hot-swap: re-randomize LoRA params and save as v1.
    print(f"[6] re-randomizing LoRA params for v1 hot-swap ...")
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if p.requires_grad:
                p.normal_(mean=0.0, std=0.02)
    name_v1 = "policy_v1"
    path_v1 = out_root / name_v1
    client.hot_swap(peft_model, path_v1, lora_name=name_v1, prev_lora_name=name_v0)
    models_after2 = client.list_models()
    print(f"[7] after hot_swap to v1 /v1/models: {models_after2}")
    assert name_v1 in models_after2, "v1 not loaded"
    assert name_v0 not in models_after2, "v0 should have been unloaded"

    t0 = time.time()
    lora_resp2 = chat_completion(args.base_url, name_v1, args.prompt)
    lora_text2 = lora_resp2["choices"][0]["message"]["content"]
    print(f"[8] v1 completion ({time.time()-t0:.1f}s): {lora_text2!r}")

    # Final cleanup.
    client.unload_adapter(name_v1)
    final_models = client.list_models()
    print(f"[9] after cleanup /v1/models: {final_models}")
    assert name_v1 not in final_models

    print("\n=== LoRA hot-reload smoke test PASSED ===")
    print(f"    base : {base_text!r}")
    print(f"    v0   : {lora_text!r}")
    print(f"    v1   : {lora_text2!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
