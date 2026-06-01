"""LoRA hot-reload bridge between FSDP training and the vLLM rollout server.

After each training step we:
  1. Save the current LoRA adapter (peft `save_pretrained`) to a versioned path.
  2. POST `/v1/load_lora_adapter` so vLLM swaps it into the engine.
  3. Subsequent chat-completion requests target `model=lora_name` to use the new policy.

This module intentionally does NOT manage the underlying torch model — callers
own the `PeftModel`/FSDP wrapper. We only handle the disk + HTTP bridge.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import requests

logger = logging.getLogger(__name__)

# Qwen3-style attention + MLP modules — matches the architecture used at
# inference. Update if the base model changes.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.0


def make_lora_config(
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
):
    """Build a peft LoraConfig matching the vLLM server's `--max-lora-rank`."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM,
    )


def save_adapter(peft_model, out_dir: str | os.PathLike) -> str:
    """Save a peft model's LoRA adapter to `out_dir`.

    Returns the absolute path (which is what vLLM expects in `lora_path`).
    Overwrites any existing files in `out_dir`.
    """
    out = Path(out_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out), safe_serialization=True)
    # Sanity: peft writes adapter_config.json + adapter_model.safetensors
    expected = {"adapter_config.json", "adapter_model.safetensors"}
    actual = {p.name for p in out.iterdir()}
    missing = expected - actual
    if missing:
        raise RuntimeError(
            f"peft save_pretrained produced unexpected output. Missing: {missing}. "
            f"Got: {sorted(actual)}"
        )
    return str(out)


@dataclass
class VLLMLoraClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 60.0

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/v1/models", timeout=self.timeout)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    def load_adapter(self, lora_name: str, lora_path: str) -> None:
        """POST /v1/load_lora_adapter; raises on non-2xx."""
        url = f"{self.base_url}/v1/load_lora_adapter"
        payload = {"lora_name": lora_name, "lora_path": str(Path(lora_path).resolve())}
        logger.info("vLLM load_lora_adapter: %s <- %s", lora_name, payload["lora_path"])
        r = requests.post(url, json=payload, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(
                f"load_lora_adapter failed [{r.status_code}]: {r.text}"
            )

    def unload_adapter(self, lora_name: str) -> None:
        url = f"{self.base_url}/v1/unload_lora_adapter"
        logger.info("vLLM unload_lora_adapter: %s", lora_name)
        r = requests.post(url, json={"lora_name": lora_name}, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError(
                f"unload_lora_adapter failed [{r.status_code}]: {r.text}"
            )

    def hot_swap(
        self,
        peft_model,
        out_dir: str | os.PathLike,
        lora_name: str,
        prev_lora_name: Optional[str] = None,
    ) -> str:
        """Save peft_model to out_dir, load into vLLM, then unload previous version.

        Returns the loaded lora_name.
        """
        path = save_adapter(peft_model, out_dir)
        self.load_adapter(lora_name, path)
        if prev_lora_name and prev_lora_name != lora_name:
            try:
                self.unload_adapter(prev_lora_name)
            except Exception as e:  # pragma: no cover — best effort
                logger.warning("unload of previous adapter %r failed: %s", prev_lora_name, e)
        return lora_name
