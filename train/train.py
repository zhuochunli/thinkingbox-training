"""Single-GPU RL training loop for thinkingbox.

Pipeline per step:
  1. Sample a batch of prompts from the train split.
  2. Rollout G samples per prompt against vLLM (current LoRA name).
  3. Score with the configured reward function.
  4. Tokenize each rollout (assistant-mask via train.tokenize_chat).
  5. Forward through the policy (no grad) → π_old logprobs.
     Forward with adapter disabled → π_ref logprobs.
  6. PPO inner epochs: forward → compute_rl_loss → backward → step.
  7. Save adapter, hot-swap into vLLM, rebuild the rollout runner.
  8. Log metrics. (Eval pass added in Step G.)

Designed for a single GPU (e.g. cuda:0) with the 9B base in bfloat16 + LoRA
trainable params + gradient checkpointing. FSDP wiring is a follow-up.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from thinkingbox.cli.common import load_yaml
from thinkingbox.common.config_types import ConfigFile
from thinkingbox.common.http_client import initialize_dns_cache

from train.data_pipeline import hydrate, load_test_list, split_train_eval
from train.lora_sync import VLLMLoraClient, make_lora_config
from train.rewards import score_rollouts
from train.rl_loss import (
    RLLossConfig,
    compute_advantages,
    compute_rl_loss,
    gather_token_logprobs,
)
from train.rollout import Rollout, RolloutRunner
from train.tokenize_chat import (
    TokenizedRollout,
    render_with_assistant_mask,
)
# === STEP G integration (reconstructed) ===
from train import patches
from train.checkpoint import (
    load_train_state,
    lora_path_for,
    prune_old_state,
    resolve_resume_path,
    save_train_state,
    state_path,
)
from train.eval_loop import log_eval, run_eval
from train.wandb_logger import WandbLogger
# === END STEP G ===

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers (no-op when launched without torchrun)
# ---------------------------------------------------------------------------
def setup_ddp() -> tuple[int, int, int]:
    """Init NCCL process group from torchrun env. Returns (rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def ddp_barrier(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def peft_of(policy):
    """Return the underlying PeftModel whether or not `policy` is a DDP wrapper."""
    return policy.module if isinstance(policy, torch.nn.parallel.DistributedDataParallel) else policy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen3.5-9B"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True

    # Data
    dataset_dir: str = ""               # required
    train_list: str = "data/train_list.yaml"
    agent: str = "think"

    # Rollout
    n_prompts_per_step: int = 2
    n_samples_per_prompt: int = 4
    rollout_concurrency: int = 4

    # Training
    max_steps: int = 2
    lr: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ppo_epochs: int = 1
    micro_batch_size: int = 1           # process this many rollouts at a time
    max_seq_len: int = 12288            # truncate left if a rollout exceeds this

    # RL loss
    rl_loss: RLLossConfig = field(default_factory=RLLossConfig)
    reward_fn: str = "binary_test_result"

    # vLLM hot-reload
    tb_config: str = "config_training.yaml"
    vllm_base_url: str = "http://127.0.0.1:8000"
    lora_save_dir: str = "checkpoints/lora"
    lora_name_template: str = "policy_step_{step:05d}"

    # I/O
    log_file: str = "checkpoints/train_log.jsonl"
    seed: int = 0

    # === STEP G integration (reconstructed) ===
    state_save_dir: str = "checkpoints/state"
    # === END STEP G ===


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor          # [B, T]
    attention_mask: torch.Tensor     # [B, T]
    assistant_mask: torch.Tensor     # [B, T]
    rewards: torch.Tensor            # [B]
    group_ids: torch.Tensor          # [B]
    seq_lens: torch.Tensor           # [B] real token counts


def collate(
    tokenized: list[TokenizedRollout],
    rewards: list[float],
    group_ids: list[int],
    pad_id: int,
    max_seq_len: int,
    device: torch.device,
) -> Batch:
    truncated_ids: list[list[int]] = []
    truncated_masks: list[list[int]] = []
    for tr in tokenized:
        ids = tr.input_ids
        mask = tr.assistant_mask
        if len(ids) > max_seq_len:
            # left-truncate (keep the tail with most of the assistant content)
            ids = ids[-max_seq_len:]
            mask = mask[-max_seq_len:]
        truncated_ids.append(ids)
        truncated_masks.append(mask)

    lens = [len(x) for x in truncated_ids]
    T = max(lens)
    B = len(truncated_ids)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    amask = torch.zeros((B, T), dtype=torch.long)
    for i, (ids, m) in enumerate(zip(truncated_ids, truncated_masks)):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L] = 1
        amask[i, :L] = torch.tensor(m, dtype=torch.long)
    return Batch(
        input_ids=input_ids.to(device),
        attention_mask=attn.to(device),
        assistant_mask=amask.to(device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        group_ids=torch.tensor(group_ids, dtype=torch.long, device=device),
        seq_lens=torch.tensor(lens, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Per-batch logprob computation (memory-friendly via micro-batches)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_logprobs(
    model,
    batch: Batch,
    micro_batch_size: int,
) -> torch.Tensor:
    """Returns [B, T-1] gather log π(target | prefix) under `model`'s current state.

    Pass the underlying PeftModel (not the DDP wrapper) since this is a no-grad
    forward and the DDP forward hook would only add overhead.
    """
    model = peft_of(model)
    B, T = batch.input_ids.shape
    out = torch.empty((B, T - 1), dtype=torch.float32, device=batch.input_ids.device)
    for s in range(0, B, micro_batch_size):
        e = min(s + micro_batch_size, B)
        ids = batch.input_ids[s:e]
        attn = batch.attention_mask[s:e]
        logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
        shift_logits = logits[:, :-1, :].float()
        shift_targets = ids[:, 1:]
        out[s:e] = gather_token_logprobs(shift_logits, shift_targets)
    return out


def policy_forward_with_grad(
    model,
    batch: Batch,
    micro_batch_size: int,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    cfg: RLLossConfig,
) -> dict:
    """Forward + loss + backward over the batch in micro-batches.

    Accumulates gradients across micro-batches with proper weighting so that
    the effective loss is identical to a single-shot forward over the full
    batch (assuming `loss_agg='token_mean'` — the default for DAPO/Dr. GRPO).
    """
    B, T = batch.input_ids.shape
    # Pre-compute per-micro-batch token weights so token_mean sums correctly
    total_tokens = batch.assistant_mask[:, 1:].sum().clamp_min(1).item()

    # Pre-compute advantages on the FULL batch (group stats need all members)
    advantages_full = compute_advantages(batch.rewards, batch.group_ids, cfg)

    diag_sum = {"loss": 0.0, "pg_loss": 0.0, "kl_loss": 0.0, "entropy_loss": 0.0,
                "mean_advantage": 0.0, "mean_ratio": 0.0, "clip_frac": 0.0,
                "num_tokens": 0.0}
    diag_w = 0.0

    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    micro_starts = list(range(0, B, micro_batch_size))
    last_idx = len(micro_starts) - 1

    for i, s in enumerate(micro_starts):
        e = min(s + micro_batch_size, B)
        ids = batch.input_ids[s:e]
        attn = batch.attention_mask[s:e]
        amask = batch.assistant_mask[s:e]

        logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits.float()

        out = compute_rl_loss(
            logits=logits,
            input_ids=ids,
            assistant_mask=amask,
            old_logprobs=old_logprobs[s:e],
            ref_logprobs=ref_logprobs[s:e] if ref_logprobs is not None else None,
            rewards=batch.rewards[s:e],
            group_ids=batch.group_ids[s:e],
            advantages=advantages_full[s:e],  # injected so micro-batching is correct
            cfg=cfg,
        )
        # Re-weight so the effective batch-level loss == token_mean over full batch
        sub_tokens = amask[:, 1:].sum().clamp_min(1).item()
        w = sub_tokens / total_tokens
        # Skip DDP grad sync on all but the last micro-batch (one all-reduce per step).
        sync_ctx = model.no_sync() if (is_ddp and i < last_idx) else nullcontext()
        with sync_ctx:
            (out["loss"] * w).backward()
        for k, v in out.items():
            if torch.is_tensor(v):
                diag_sum[k] += float(v.detach()) * w
        diag_w += w

    return {k: v / max(diag_w, 1e-12) for k, v in diag_sum.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_runner(tb_cfg: ConfigFile, lora_name: str, concurrency: int) -> RolloutRunner:
    # Mutate the deployment so vLLM dispatches to our LoRA
    new_cfg = tb_cfg.model_copy(deep=True)
    new_cfg.orchestrator.agent_model.deployment = lora_name
    return RolloutRunner(config=new_cfg, concurrency=concurrency, dump_raw=False)


def run_one_step(
    step: int,
    cfg: TrainConfig,
    tb_cfg: ConfigFile,
    train_cases: list,
    policy,
    tokenizer,
    optimizer,
    lora_client: VLLMLoraClient,
    current_lora_name: str,
    log_fh,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[str, dict | None]:
    device = torch.device(cfg.device)
    t_step = time.monotonic()

    # ----- Sample prompts (deterministic across ranks: separate RNG seeded by step) -----
    by_name = {tc.uid: tc for tc in train_cases}
    rng = random.Random(cfg.seed * 1_000_003 + step)
    pick = rng.sample(sorted(by_name), cfg.n_prompts_per_step)
    # Shard prompts across ranks. Each rank rolls out FULL G samples for its
    # subset, so per-prompt groups stay complete on one rank → local
    # compute_advantages == global group-norm advantage.
    if cfg.n_prompts_per_step % world_size != 0:
        raise ValueError(
            f"--n-prompts ({cfg.n_prompts_per_step}) must be divisible by "
            f"world_size ({world_size}) for prompt-level DDP sharding."
        )
    local_pick = pick[rank::world_size]
    batch_tcs = [by_name[n] for n in local_pick]

    # ----- Rollout (this rank's slice only) -----
    runner = build_runner(tb_cfg, current_lora_name, cfg.rollout_concurrency)
    t0 = time.monotonic()
    rollouts_by_uid = asyncio.run(
        runner.rollout_many(batch_tcs, n_samples=cfg.n_samples_per_prompt)
    )
    t_rollout = time.monotonic() - t0
    flat: list[Rollout] = []
    group_ids: list[int] = []
    for gid, uid in enumerate(sorted(rollouts_by_uid)):
        for r in rollouts_by_uid[uid]:
            flat.append(r)
            group_ids.append(gid)
    n_sys_errors = sum(r.is_system_error for r in flat)
    if not flat:
        logger.warning("step %d: no rollouts; skipping", step)
        return current_lora_name, None

    # ----- Reward -----
    rewards = score_rollouts(flat, name=cfg.reward_fn)

    # ----- Tokenize (skip rollouts whose chat-template render fails, e.g. system errors) -----
    tokenized: list = []
    render_keep: list[int] = []
    for i, r in enumerate(flat):
        if r.is_system_error:
            continue
        try:
            tokenized.append(render_with_assistant_mask(r.messages, tokenizer))
            render_keep.append(i)
        except Exception as e:
            roles = [getattr(m, "role", "?") for m in r.messages]
            logger.warning("step %d: render failed for rollout %d (%s: %s); roles=%s",
                           step, i, type(e).__name__, str(e)[:160], roles)
    flat = [flat[i] for i in render_keep]
    rewards = [rewards[i] for i in render_keep]
    group_ids = [group_ids[i] for i in render_keep]

    # Skip degenerate rollouts (no assistant tokens to train on)
    keep = [i for i, tr in enumerate(tokenized) if any(tr.assistant_mask)]
    if not keep:
        logger.warning("step %d: no usable rollouts (sys_errors=%d); skipping",
                       step, n_sys_errors)
        return current_lora_name, None
    if len(keep) < len(tokenized):
        logger.info("step %d: dropping %d empty-mask rollouts", step, len(tokenized) - len(keep))
    tokenized = [tokenized[i] for i in keep]
    flat = [flat[i] for i in keep]
    rewards = [rewards[i] for i in keep]
    group_ids = [group_ids[i] for i in keep]

    # ----- Collate -----
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch = collate(
        tokenized, rewards, group_ids,
        pad_id=pad_id, max_seq_len=cfg.max_seq_len, device=device,
    )

    # ----- π_old and π_ref (no grad) -----
    t0 = time.monotonic()
    policy.eval()
    old_logp = compute_logprobs(policy, batch, cfg.micro_batch_size)
    with peft_of(policy).disable_adapter():
        ref_logp = compute_logprobs(policy, batch, cfg.micro_batch_size)
    t_logp = time.monotonic() - t0

    # ----- PPO inner epochs -----
    policy.train()
    t0 = time.monotonic()
    for inner in range(cfg.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        diag = policy_forward_with_grad(
            policy, batch, cfg.micro_batch_size, old_logp, ref_logp, cfg.rl_loss,
        )
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], cfg.grad_clip
            )
        optimizer.step()
    t_update = time.monotonic() - t0

    # ----- Hot-swap LoRA into vLLM (rank 0 only; other ranks barrier below) -----
    next_name = cfg.lora_name_template.format(step=step + 1)
    next_path = Path(cfg.lora_save_dir) / next_name
    if is_rank0(rank):
        lora_client.hot_swap(
            peft_of(policy), next_path,
            lora_name=next_name, prev_lora_name=current_lora_name,
        )
    ddp_barrier(world_size)

    # ----- Log -----
    reward_arr = torch.tensor(rewards, dtype=torch.float32)
    log_row = {
        "step": step,
        "lora_name": current_lora_name,
        "next_lora_name": next_name,
        "n_rollouts": len(flat),
        "n_sys_errors": n_sys_errors,
        "reward_mean": float(reward_arr.mean()),
        "reward_std": float(reward_arr.std(unbiased=False)) if len(rewards) > 1 else 0.0,
        "reward_min": float(reward_arr.min()),
        "reward_max": float(reward_arr.max()),
        "loss": diag["loss"],
        "pg_loss": diag["pg_loss"],
        "kl_loss": diag["kl_loss"],
        "mean_advantage": diag["mean_advantage"],
        "mean_ratio": diag["mean_ratio"],
        "clip_frac": diag["clip_frac"],
        "max_seq_len": int(batch.seq_lens.max()),
        "mean_seq_len": float(batch.seq_lens.float().mean()),
        "t_rollout_s": round(t_rollout, 2),
        "t_logp_s": round(t_logp, 2),
        "t_update_s": round(t_update, 2),
        "t_step_s": round(time.monotonic() - t_step, 2),
    }
    msg = (
        f"step={step:04d}  loss={log_row['loss']:+.4f}  pg={log_row['pg_loss']:+.4f}  "
        f"kl={log_row['kl_loss']:.4f}  reward μ={log_row['reward_mean']:.3f}±{log_row['reward_std']:.3f}  "
        f"clip={log_row['clip_frac']:.2f}  rollout={log_row['t_rollout_s']}s  "
        f"update={log_row['t_update_s']}s  total={log_row['t_step_s']}s"
    )
    if is_rank0(rank):
        logger.info(msg)
        if log_fh is not None:
            log_fh.write(json.dumps(log_row) + "\n")
            log_fh.flush()

    return next_name, log_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb-config", default="config_training.yaml")
    ap.add_argument(
        "--dataset",
        default=os.environ.get(
            "THINKINGBOX_DATA",
            "/home/azureuser/zhuochun_microsoft/AI.ThinkingBox.Data",
        )
        + "/dataset",
    )
    ap.add_argument("--train-list", default="data/train_list.yaml")
    ap.add_argument("--agent", default="think")
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--algo", choices=["grpo", "dr_grpo", "dapo"], default="dr_grpo")
    ap.add_argument("--lora-save-dir", default="checkpoints/lora")
    ap.add_argument("--log-file", default="checkpoints/train_log.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ddp-find-unused-parameters", action="store_true",
                    help="Pass find_unused_parameters=True to DDP (slower; use only if needed).")
    # === STEP G integration (reconstructed) ===
    ap.add_argument("--max-seq-len", type=int, default=16384,
                    help="Truncate left if a rollout exceeds this token count.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Run eval every N training steps (0 disables periodic eval).")
    ap.add_argument("--eval-at-start", action="store_true",
                    help="Run an eval pass before the first training step.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="Save train state every N steps (0 disables).")
    ap.add_argument("--keep-checkpoints", type=int, default=4,
                    help="Keep only the latest N state checkpoints (step 0 is always kept).")
    ap.add_argument("--state-save-dir", default="checkpoints/state")
    ap.add_argument("--resume", default=None,
                    help="Path to a state_step_*.pt file, or 'latest'.")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb scalar logging (rank-0 only).")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-project", default="microsoft-train")
    ap.add_argument("--wandb-entity", default="zhuochun-university-of-pittsburgh")
    ap.add_argument("--weave", action="store_true",
                    help="Enable weave tracing (uses --wandb-project / --wandb-entity).")
    # === END STEP G ===
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ----- RL loss config from --algo -----
    if args.algo == "grpo":
        rl = RLLossConfig(advantage="group_norm", clip_low=0.2, clip_high=0.2,
                          loss_agg="seq_mean_token_sum", kl_coef=0.04, kl_estimator="k3")
    elif args.algo == "dr_grpo":
        rl = RLLossConfig(advantage="group_mean", clip_low=0.2, clip_high=0.2,
                          loss_agg="token_mean", kl_coef=0.04, kl_estimator="k3")
    elif args.algo == "dapo":
        rl = RLLossConfig(advantage="group_mean", clip_low=0.2, clip_high=0.28,
                          loss_agg="token_mean", kl_coef=0.0)
    else:
        raise ValueError(args.algo)

    cfg = TrainConfig(
        dataset_dir=args.dataset,
        train_list=args.train_list,
        agent=args.agent,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        n_prompts_per_step=args.n_prompts,
        n_samples_per_prompt=args.g,
        rollout_concurrency=args.concurrency,
        micro_batch_size=args.micro_batch,
        max_steps=args.max_steps,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        device=args.device,
        rl_loss=rl,
        tb_config=args.tb_config,
        lora_save_dir=args.lora_save_dir,
        log_file=args.log_file,
        seed=args.seed,
        # === STEP G integration (reconstructed) ===
        max_seq_len=args.max_seq_len,
        state_save_dir=args.state_save_dir,
        # === END STEP G ===
    )
    # ----- DDP setup (no-op if not launched via torchrun) -----
    rank, world_size, local_rank = setup_ddp()
    if torch.cuda.is_available():
        cfg.device = f"cuda:{local_rank}"
    else:
        cfg.device = "cpu"

    set_seed(cfg.seed)
    initialize_dns_cache()

    # === STEP G integration (reconstructed) ===
    # Patch AgentSession before any rollout runner is built so the merged-system
    # prefix is in place for both training and eval rollouts.
    patches.apply()
    # === END STEP G ===

    # ----- Load thinkingbox config + data -----
    tb_cfg: ConfigFile = load_yaml(cfg.tb_config, ConfigFile)
    names = load_test_list(cfg.train_list)
    train_names, eval_names = split_train_eval(names)
    if is_rank0(rank):
        logger.info("rank=%d/%d  device=%s  train=%d  eval=%d  algo=%s",
                    rank, world_size, cfg.device, len(train_names), len(eval_names), args.algo)
    train_cases = hydrate(train_names, cfg.dataset_dir, cfg.agent)
    if is_rank0(rank):
        logger.info("hydrated %d train cases", len(train_cases))
    eval_cases = hydrate(eval_names, cfg.dataset_dir, cfg.agent)
    if is_rank0(rank):
        logger.info("hydrated %d eval cases", len(eval_cases))

    # ----- Load model + LoRA -----
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    if is_rank0(rank):
        logger.info("loading %s in %s on %s ...", cfg.model_name, cfg.dtype, cfg.device)
    t0 = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, trust_remote_code=True,
    )
    base.to(cfg.device)
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        # Required when only LoRA params are trainable: keeps embedding output's
        # requires_grad=True so autograd can backprop through the checkpointed graph.
        base.enable_input_require_grads()
    peft_model = get_peft_model(base, make_lora_config(cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout))
    if is_rank0(rank):
        logger.info("model loaded in %.1fs; trainable params: %s",
                    time.monotonic() - t0,
                    sum(p.numel() for p in peft_model.parameters() if p.requires_grad))

    # Wrap with DDP. With LoRA-only training every trainable param is touched
    # by every micro-batch, so find_unused_parameters=False is correct and faster.
    if world_size > 1:
        policy = torch.nn.parallel.DistributedDataParallel(
            peft_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    else:
        policy = peft_model

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ----- Save & load initial adapter (B=0 → behaves as base model) -----
    lora_client = VLLMLoraClient(base_url=cfg.vllm_base_url)
    init_name = cfg.lora_name_template.format(step=0)
    init_path = Path(cfg.lora_save_dir) / init_name

    # === STEP G integration (reconstructed) ===
    # Resolve --resume into either a fresh start (step 0) or a restored state.
    start_step = 0
    if args.resume:
        resume_file = resolve_resume_path(args.resume, cfg.state_save_dir)
        payload = load_train_state(
            resume_file, optimizer, peft_of(policy),
            cfg.lora_save_dir, cfg.lora_name_template,
            world_size=world_size,
            map_location=cfg.device,
        )
        start_step = payload["step"] + 1
        current_lora_name = payload["lora_name"]
        resume_adapter_dir = lora_path_for(
            cfg.lora_save_dir, cfg.lora_name_template, payload["step"]
        )
        if is_rank0(rank):
            lora_client.load_adapter(current_lora_name, str(resume_adapter_dir))
            logger.info(
                "resumed from %s; next step=%d  lora=%s",
                resume_file, start_step, current_lora_name,
            )
        ddp_barrier(world_size)
    else:
        if is_rank0(rank):
            lora_client.hot_swap(peft_of(policy), init_path, lora_name=init_name)
            logger.info("loaded initial adapter into vLLM: %s", init_name)
        current_lora_name = init_name
        ddp_barrier(world_size)
    # === END STEP G ===

    # === STEP G integration (reconstructed) ===
    # wandb / weave init: rank-0 only. Other ranks get a no-op WandbLogger.
    wandb_log = WandbLogger.maybe_init(
        enabled=args.wandb and is_rank0(rank),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        run_id=None,  # TODO[recover]: persist+restore wandb run_id via checkpoint payload
        weave_enabled=args.weave and is_rank0(rank),
        config={**dataclasses.asdict(cfg), "algo": args.algo,
                "world_size": world_size,
                "resumed": args.resume is not None},
    )
    # Persist a step-0 anchor checkpoint so --resume latest always works (rank 0).
    if not args.resume and args.save_every > 0 and is_rank0(rank):
        save_train_state(
            step=0, optimizer=optimizer, lora_name=init_name,
            world_size=world_size,
            out_path=state_path(cfg.state_save_dir, 0),
        )
    ddp_barrier(world_size)
    # === END STEP G ===

    # ----- Train -----
    if is_rank0(rank):
        Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
    ddp_barrier(world_size)

    log_fh_cm = (
        open(cfg.log_file, "a", encoding="utf-8") if is_rank0(rank) else nullcontext()
    )
    try:
        with log_fh_cm as log_fh:
            # === STEP G integration (reconstructed) ===
            if args.eval_at_start and start_step == 0:
                if is_rank0(rank):
                    eval_row = run_eval(
                        start_step, tb_cfg, eval_cases, current_lora_name, cfg.reward_fn,
                    )
                    log_eval(eval_row, log_fh)
                    wandb_log.log_eval(start_step, eval_row)
                ddp_barrier(world_size)
            # === END STEP G ===
            for step in range(start_step, cfg.max_steps):
                current_lora_name, train_row = run_one_step(
                    step, cfg, tb_cfg, train_cases, policy, tokenizer, optimizer,
                    lora_client, current_lora_name, log_fh,
                    rank=rank, world_size=world_size,
                )
                # === STEP G integration (reconstructed) ===
                if train_row is not None and is_rank0(rank):
                    wandb_log.log_train(step, train_row)
                # Periodic eval (rank 0 only; other ranks barrier).
                if args.eval_every > 0 and ((step + 1) % args.eval_every == 0):
                    if is_rank0(rank):
                        eval_row = run_eval(
                            step + 1, tb_cfg, eval_cases, current_lora_name, cfg.reward_fn,
                        )
                        log_eval(eval_row, log_fh)
                        wandb_log.log_eval(step + 1, eval_row)
                    ddp_barrier(world_size)
                # Periodic state checkpoint + prune (rank 0 only).
                if args.save_every > 0 and ((step + 1) % args.save_every == 0):
                    if is_rank0(rank):
                        save_train_state(
                            step=step + 1, optimizer=optimizer,
                            lora_name=current_lora_name, world_size=world_size,
                            out_path=state_path(cfg.state_save_dir, step + 1),
                        )
                        prune_old_state(
                            cfg.state_save_dir, cfg.lora_save_dir,
                            cfg.lora_name_template, keep=args.keep_checkpoints,
                        )
                    ddp_barrier(world_size)
                # === END STEP G ===

        if is_rank0(rank):
            logger.info("training complete; final adapter: %s", current_lora_name)
    finally:
        # === STEP G integration (reconstructed) ===
        wandb_log.finish()
        # === END STEP G ===
        cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
