"""Generic on-policy RL loss for LLM fine-tuning.

Supports GRPO, DAPO, leave-one-out, and reduces to vanilla PG when configured
naively. The whole loss is config-driven so we can switch algorithms without
forking the training loop.

Inputs (per batch):
  * logits:        FloatTensor [B, T, V] from the current policy
  * old_logprobs:  FloatTensor [B, T-1] log π_old(a_t | s_<t)  (token-level)
                   Use the policy's own pre-update logprobs for true PPO-style
                   ratio. For pure REINFORCE pass logits.detach()'s logprobs.
  * ref_logprobs:  FloatTensor [B, T-1] log π_ref(a_t | s_<t)  (frozen base);
                   only required if kl_coef > 0.
  * input_ids:     LongTensor  [B, T]
  * assistant_mask:BoolTensor  [B, T]   1 over policy-generated tokens
  * group_ids:     LongTensor  [B]      which prompt-group each sample belongs to
  * rewards:       FloatTensor [B]      scalar reward per rollout
  * cfg:           RLLossConfig

Returns dict with:
  loss, pg_loss, kl_loss, entropy_loss, mean_advantage, clip_frac, mean_ratio
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RLLossConfig:
    # Advantage estimator
    advantage: str = "group_norm"          # group_norm | group_mean | leave_one_out
    advantage_eps: float = 1e-6

    # PPO clip (asymmetric supported, à la DAPO clip-higher)
    clip_low: float = 0.2
    clip_high: float = 0.2

    # Loss aggregation across tokens
    loss_agg: str = "token_mean"           # token_mean (DAPO) | seq_mean_token_sum (GRPO-orig)

    # KL anchor to reference policy
    kl_coef: float = 0.0
    kl_estimator: str = "k3"               # k1=logr, k2=0.5*logr^2, k3=exp(r)-r-1 (Schulman)

    # Optional entropy regularization (subtracts coef * H)
    entropy_coef: float = 0.0

    # Numerical clamp for ratios to avoid overflow when policy drifts hard
    max_ratio: float = 10.0


# ---------------------------------------------------------------------------
# Advantage estimators
# ---------------------------------------------------------------------------
def compute_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, cfg: RLLossConfig) -> torch.Tensor:
    """Per-rollout scalar advantage based on group statistics.

    rewards   : [B]
    group_ids : [B] integer group id
    returns   : [B] advantage (will be broadcast over tokens later)
    """
    adv = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        m = group_ids == g
        r = rewards[m]
        if cfg.advantage == "group_norm":
            mu, sigma = r.mean(), r.std(unbiased=False)
            adv[m] = (r - mu) / (sigma + cfg.advantage_eps)
        elif cfg.advantage == "group_mean":
            adv[m] = r - r.mean()
        elif cfg.advantage == "leave_one_out":
            n = r.numel()
            if n <= 1:
                adv[m] = 0.0
            else:
                loo_mean = (r.sum() - r) / (n - 1)
                adv[m] = r - loo_mean
        else:
            raise ValueError(f"unknown advantage estimator: {cfg.advantage!r}")
    return adv


# ---------------------------------------------------------------------------
# Per-token logprobs / entropy
# ---------------------------------------------------------------------------
def gather_token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """logits: [B, T, V]; target_ids: [B, T] → logprobs of target token at each pos.

    Caller is responsible for the next-token shift (i.e. pass logits[:, :-1]
    and ids[:, 1:]).
    """
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Categorical entropy per position. logits: [B, T, V] → [B, T]."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------
def compute_rl_loss(
    *,
    logits: torch.Tensor,            # [B, T, V]   — current policy, requires grad
    input_ids: torch.Tensor,         # [B, T]
    assistant_mask: torch.Tensor,    # [B, T] bool/int
    old_logprobs: torch.Tensor,      # [B, T-1]    π_old (detached)
    ref_logprobs: Optional[torch.Tensor],  # [B, T-1] π_ref (detached) or None
    rewards: torch.Tensor,           # [B]
    group_ids: torch.Tensor,         # [B] int
    cfg: RLLossConfig,
) -> dict:
    B, T, V = logits.shape
    assert input_ids.shape == (B, T)
    assert assistant_mask.shape == (B, T)
    assert old_logprobs.shape == (B, T - 1)

    # Next-token shift: predict token t+1 from logits at position t
    shift_logits = logits[:, :-1, :]                    # [B, T-1, V]
    shift_targets = input_ids[:, 1:]                    # [B, T-1]
    # A position t (in the shifted space) trains on token t+1, so the mask
    # is the assistant_mask of the *target* token.
    mask = assistant_mask[:, 1:].to(shift_logits.dtype)  # [B, T-1]

    new_logprobs = gather_token_logprobs(shift_logits, shift_targets)  # [B, T-1]

    # Importance ratio
    logr = new_logprobs - old_logprobs
    ratio = torch.exp(logr.clamp(max=float(torch.log(torch.tensor(cfg.max_ratio)))))

    # Advantages: per-rollout scalar, broadcast to tokens
    advantages = compute_advantages(rewards, group_ids, cfg)  # [B]
    adv_t = advantages.unsqueeze(1).expand_as(ratio)          # [B, T-1]

    # PPO clipped objective (per token)
    unclipped = ratio * adv_t
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_low, 1.0 + cfg.clip_high) * adv_t
    pg_per_tok = -torch.minimum(unclipped, clipped)
    clipped_mask = (unclipped != clipped).to(mask.dtype) * mask
    clip_frac = (clipped_mask.sum() / mask.sum().clamp_min(1.0))

    pg_loss = _reduce(pg_per_tok, mask, cfg.loss_agg)

    # KL anchor
    if cfg.kl_coef > 0.0 and ref_logprobs is not None:
        kl_per_tok = _kl_estimator(new_logprobs, ref_logprobs, cfg.kl_estimator)
        kl_loss = _reduce(kl_per_tok, mask, cfg.loss_agg)
    else:
        kl_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

    # Entropy bonus (subtracted from loss → encourages exploration)
    if cfg.entropy_coef > 0.0:
        ent_per_tok = token_entropy(shift_logits)
        entropy = _reduce(ent_per_tok, mask, cfg.loss_agg)
        entropy_loss = -cfg.entropy_coef * entropy
    else:
        entropy_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

    loss = pg_loss + cfg.kl_coef * kl_loss + entropy_loss

    return {
        "loss": loss,
        "pg_loss": pg_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "entropy_loss": entropy_loss.detach(),
        "mean_advantage": advantages.mean().detach(),
        "mean_ratio": (ratio * mask).sum().detach() / mask.sum().clamp_min(1.0).detach(),
        "clip_frac": clip_frac.detach(),
        "num_tokens": mask.sum().detach(),
    }


def _reduce(per_tok: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Aggregate a per-token quantity into a scalar."""
    if mode == "token_mean":
        # DAPO: average over all assistant tokens in the batch (long answers
        # get proportionally more weight, fixes GRPO length bias).
        return (per_tok * mask).sum() / mask.sum().clamp_min(1.0)
    elif mode == "seq_mean_token_sum":
        # GRPO-original: sum-over-tokens within each sequence, mean across the
        # batch. Tends to underweight long sequences.
        per_seq = (per_tok * mask).sum(dim=-1)
        return per_seq.mean()
    elif mode == "seq_mean_token_mean":
        per_seq = (per_tok * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        return per_seq.mean()
    else:
        raise ValueError(f"unknown loss_agg: {mode!r}")


def _kl_estimator(new_logp: torch.Tensor, ref_logp: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-token KL estimator. All are unbiased estimators of KL(π_new || π_ref).
    See http://joschu.net/blog/kl-approx.html
    """
    logr = ref_logp - new_logp  # log(π_ref / π_new); KL = E_new[-logr]
    if kind == "k1":
        return -logr
    if kind == "k2":
        return 0.5 * logr.pow(2)
    if kind == "k3":
        # Always-positive, low-variance Schulman estimator
        return torch.exp(logr) - logr - 1.0
    raise ValueError(f"unknown kl_estimator: {kind!r}")
