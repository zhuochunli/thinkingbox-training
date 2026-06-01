"""Save / load / prune training state for resumable RL runs.

LoRA adapter weights are already saved per-step under `<lora_save_dir>/policy_step_{N:05d}/`
by `lora_sync.save_adapter`. This module manages the *companion* state file that
holds the bits PEFT doesn't: optimizer, RNG, step counter, etc.

Layout::

    checkpoints/
      lora/policy_step_{N:05d}/           # LoRA adapter (peft save_pretrained)
      state/state_step_{N:05d}.pt         # this module

State files are written by rank-0 only; every rank loads the same file on resume
so optimizer + RNG stay consistent across DDP ranks.
"""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_STATE_RE = re.compile(r"^state_step_(\d{5})\.pt$")


def state_path(state_dir: str | Path, step: int) -> Path:
    return Path(state_dir) / f"state_step_{step:05d}.pt"


def lora_path_for(lora_save_dir: str | Path, lora_name_template: str, step: int) -> Path:
    return Path(lora_save_dir) / lora_name_template.format(step=step)


def save_train_state(
    step: int,
    optimizer: torch.optim.Optimizer,
    lora_name: str,
    world_size: int,
    out_path: str | Path,
) -> None:
    """Atomically save the training state to `out_path`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    payload: dict[str, Any] = {
        "version": 1,
        "step": step,
        "lora_name": lora_name,
        "world_size": world_size,
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": cuda_rng,
        "python_rng_state": random.getstate(),
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.rename(out_path)
    logger.info("saved train state: %s", out_path)


def load_train_state(
    in_path: str | Path,
    optimizer: torch.optim.Optimizer,
    peft_model,
    lora_save_dir: str | Path,
    lora_name_template: str,
    world_size: int,
    map_location: str | torch.device,
) -> dict[str, Any]:
    """Load state file + restore LoRA adapter weights into `peft_model`.

    Returns the loaded payload (so the caller can read `step`, `lora_name`, etc).
    """
    in_path = Path(in_path)
    logger.info("loading train state: %s", in_path)
    # Always load the raw payload on CPU — RNG state tensors must stay as
    # CPU ByteTensors for `torch.set_rng_state`. Optimizer tensors are moved
    # to the target device below.
    payload = torch.load(in_path, map_location="cpu", weights_only=False)

    if payload.get("version") != 1:
        raise ValueError(f"unsupported state version: {payload.get('version')}")
    if payload["world_size"] != world_size:
        logger.warning(
            "resume world_size mismatch: saved=%d current=%d "
            "(optimizer state will still load; reproducibility may diverge)",
            payload["world_size"], world_size,
        )

    # ----- LoRA weights: load from the companion adapter dir into peft_model -----
    step = payload["step"]
    adapter_dir = lora_path_for(lora_save_dir, lora_name_template, step)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"resume needs LoRA adapter at {adapter_dir} (companion to {in_path})"
        )
    # Use PEFT's helpers so the on-disk key naming (no adapter-name suffix) is
    # remapped to the live model's key naming (with "default" suffix).
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    sd = load_peft_weights(str(adapter_dir), device=str(map_location))
    result = set_peft_model_state_dict(peft_model, sd, adapter_name="default")
    # PEFT returns a NamedTuple(missing_keys, unexpected_keys) like nn.Module.
    unexpected = getattr(result, "unexpected_keys", []) or []
    missing = getattr(result, "missing_keys", []) or []
    if unexpected:
        raise RuntimeError(
            f"unexpected keys when loading LoRA from {adapter_dir}: {unexpected[:5]}... "
            f"({len(unexpected)} total)"
        )
    lora_missing = [k for k in missing if ("lora_A" in k or "lora_B" in k)]
    if lora_missing:
        raise RuntimeError(
            f"LoRA keys missing after load: {lora_missing[:5]}... "
            f"({len(lora_missing)} total)"
        )
    logger.info("restored LoRA adapter weights from %s", adapter_dir)

    # ----- Optimizer -----
    optimizer.load_state_dict(payload["optimizer"])
    # Move optimizer state tensors to the current device (PyTorch loads to wherever
    # they were saved, which may not match the current rank's device).
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(map_location)
    logger.info("restored optimizer state (%d param groups)", len(optimizer.param_groups))

    # ----- RNG -----
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all"):
        try:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        except Exception as e:
            logger.warning("could not restore CUDA RNG state (likely device-count mismatch): %s", e)
    random.setstate(payload["python_rng_state"])

    return payload


def find_latest_state(state_dir: str | Path) -> Path | None:
    """Return the highest-numbered state file under `state_dir`, or None."""
    state_dir = Path(state_dir)
    if not state_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for p in state_dir.iterdir():
        m = _STATE_RE.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def resolve_resume_path(resume_arg: str, state_dir: str | Path) -> Path:
    """Resolve `--resume PATH|latest` into a concrete file path."""
    if resume_arg == "latest":
        p = find_latest_state(state_dir)
        if p is None:
            raise FileNotFoundError(f"--resume latest: no state files under {state_dir}")
        return p
    return Path(resume_arg)


def prune_old_state(
    state_dir: str | Path,
    lora_save_dir: str | Path,
    lora_name_template: str,
    keep: int,
) -> None:
    """Keep only the latest `keep` state files (and their companion LoRA dirs).

    `keep <= 0` disables pruning. We never prune step 0 (initial adapter, useful
    as a baseline reference).
    """
    if keep <= 0:
        return
    state_dir = Path(state_dir)
    if not state_dir.exists():
        return
    pairs: list[tuple[int, Path]] = []
    for p in state_dir.iterdir():
        m = _STATE_RE.match(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort()
    # Drop step 0 from the candidate set to never prune it; keep the most recent
    # `keep` of the rest.
    keepable = [(s, p) for s, p in pairs if s != 0]
    to_remove = keepable[:-keep] if len(keepable) > keep else []
    for step, sp in to_remove:
        try:
            sp.unlink()
        except OSError as e:
            logger.warning("could not unlink %s: %s", sp, e)
        adapter_dir = lora_path_for(lora_save_dir, lora_name_template, step)
        if adapter_dir.exists():
            import shutil
            try:
                shutil.rmtree(adapter_dir)
            except OSError as e:
                logger.warning("could not rmtree %s: %s", adapter_dir, e)
        logger.info("pruned checkpoint step=%d", step)
