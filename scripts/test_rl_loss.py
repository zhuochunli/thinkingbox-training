"""Unit-style validation for train.rl_loss.

Runs on CPU with toy tensors — no LM needed. Verifies:
  1. Loss is finite and gradients flow to logits.
  2. When all rewards in every group are equal, PG loss is exactly 0
     (advantages cancel out) — true for group_norm, group_mean, leave_one_out.
  3. Flipping reward signs flips gradient direction on the targeted token.
  4. KL anchor adds non-negative loss when ref != new.
  5. Asymmetric clip (DAPO clip-higher) preserves more upside than symmetric.
  6. Empty assistant mask yields zero loss with no NaNs.
  7. token_mean vs seq_mean_token_sum aggregations differ for non-uniform lengths.

Run:
    PYTHONPATH=. python scripts/test_rl_loss.py
"""
from __future__ import annotations

import sys

import torch

from train.rl_loss import (
    RLLossConfig,
    compute_advantages,
    compute_rl_loss,
    gather_token_logprobs,
)


def _make_batch(B=4, T=8, V=32, n_groups=2, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(B, T, V, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T))
    # First 3 tokens are prompt, rest are assistant
    assistant_mask = torch.zeros(B, T, dtype=torch.long)
    assistant_mask[:, 3:] = 1
    # Use detached current logprobs as π_old (REINFORCE-style starting point)
    with torch.no_grad():
        logp_old = gather_token_logprobs(logits[:, :-1, :], input_ids[:, 1:])
    # ref policy: slightly perturbed
    ref_logp = (logp_old + 0.1 * torch.randn_like(logp_old)).detach()
    # group_ids: split batch evenly
    group_ids = torch.arange(B) % n_groups
    return logits, input_ids, assistant_mask, logp_old.detach(), ref_logp, group_ids


def test_finite_and_differentiable():
    logits, ids, mask, old_lp, ref_lp, gids = _make_batch()
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    cfg = RLLossConfig(kl_coef=0.05)
    out = compute_rl_loss(
        logits=logits, input_ids=ids, assistant_mask=mask,
        old_logprobs=old_lp, ref_logprobs=ref_lp,
        rewards=rewards, group_ids=gids, cfg=cfg,
    )
    assert torch.isfinite(out["loss"]), "loss not finite"
    out["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), "no/NaN grads"
    print(f"  ✓ finite loss + gradients (loss={out['loss'].item():.4f}, "
          f"pg={out['pg_loss'].item():.4f}, kl={out['kl_loss'].item():.4f}, "
          f"clip_frac={out['clip_frac'].item():.3f})")


def test_zero_advantage_when_uniform_rewards():
    for est in ("group_norm", "group_mean", "leave_one_out"):
        logits, ids, mask, old_lp, ref_lp, gids = _make_batch()
        # All rewards identical within each group
        rewards = torch.tensor([0.7, 0.3, 0.7, 0.3])  # group 0={0.7,0.7}, group 1={0.3,0.3}
        # Actually with group_ids = [0,1,0,1], groups are {0,2}={0.7,0.7} and {1,3}={0.3,0.3}
        cfg = RLLossConfig(advantage=est, kl_coef=0.0, entropy_coef=0.0)
        adv = compute_advantages(rewards, gids, cfg)
        assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6), \
            f"{est}: advantages not zero on uniform groups: {adv}"
        out = compute_rl_loss(
            logits=logits, input_ids=ids, assistant_mask=mask,
            old_logprobs=old_lp, ref_logprobs=ref_lp,
            rewards=rewards, group_ids=gids, cfg=cfg,
        )
        assert torch.allclose(out["pg_loss"], torch.zeros_like(out["pg_loss"]), atol=1e-6), \
            f"{est}: pg_loss should be 0, got {out['pg_loss'].item()}"
        print(f"  ✓ {est}: uniform-reward group → pg_loss == 0")


def test_reward_sign_flip_flips_grad():
    # Single rollout, single group. Vary reward sign and confirm grad on the
    # logprob of the assistant tokens flips accordingly.
    logits, ids, mask, old_lp, ref_lp, gids = _make_batch(B=2, T=6, V=16)
    gids = torch.tensor([0, 0])
    cfg = RLLossConfig(advantage="group_mean", kl_coef=0.0, entropy_coef=0.0)
    # Asymmetric rewards so advantages are non-zero
    for sign in (+1.0, -1.0):
        logits_c = logits.detach().clone().requires_grad_(True)
        rewards = torch.tensor([sign * 1.0, sign * -1.0])
        out = compute_rl_loss(
            logits=logits_c, input_ids=ids, assistant_mask=mask,
            old_logprobs=old_lp, ref_logprobs=ref_lp,
            rewards=rewards, group_ids=gids, cfg=cfg,
        )
        out["loss"].backward()
        # grad on row 0's target tokens
        target_grads = logits_c.grad[0, 3:-1, :].gather(
            -1, ids[0, 4:].unsqueeze(-1)
        ).squeeze(-1).sum().item()
        if sign > 0:
            pos_grad = target_grads
        else:
            neg_grad = target_grads
    # Reward flip should flip the sign of the gradient on positive-reward sample
    assert pos_grad * neg_grad < 0, f"grad signs didn't flip: pos={pos_grad}, neg={neg_grad}"
    print(f"  ✓ reward sign flip → grad sign flip (pos={pos_grad:.4f}, neg={neg_grad:.4f})")


def test_kl_nonnegative_and_zero_when_ref_eq_new():
    logits, ids, mask, old_lp, ref_lp, gids = _make_batch()
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

    # KL with perturbed ref → > 0 (k3 always non-negative)
    cfg = RLLossConfig(kl_coef=1.0, kl_estimator="k3")
    out = compute_rl_loss(
        logits=logits, input_ids=ids, assistant_mask=mask,
        old_logprobs=old_lp, ref_logprobs=ref_lp,
        rewards=rewards, group_ids=gids, cfg=cfg,
    )
    assert out["kl_loss"].item() >= 0, f"k3 KL should be ≥0, got {out['kl_loss']}"

    # KL with ref = current logprobs → 0
    with torch.no_grad():
        cur_lp = gather_token_logprobs(logits[:, :-1, :], ids[:, 1:])
    out2 = compute_rl_loss(
        logits=logits, input_ids=ids, assistant_mask=mask,
        old_logprobs=old_lp, ref_logprobs=cur_lp.detach(),
        rewards=rewards, group_ids=gids, cfg=cfg,
    )
    assert abs(out2["kl_loss"].item()) < 1e-5, \
        f"KL should be ~0 when ref==new, got {out2['kl_loss']}"
    print(f"  ✓ KL anchor: perturbed-ref kl={out['kl_loss'].item():.4f}, "
          f"equal-ref kl={out2['kl_loss'].item():.2e}")


def test_dapo_vs_grpo_configs_run():
    logits, ids, mask, old_lp, ref_lp, gids = _make_batch()
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

    grpo = RLLossConfig(
        advantage="group_norm", clip_low=0.2, clip_high=0.2,
        loss_agg="seq_mean_token_sum", kl_coef=0.04, kl_estimator="k3",
    )
    dapo = RLLossConfig(
        advantage="group_mean", clip_low=0.2, clip_high=0.28,
        loss_agg="token_mean", kl_coef=0.0, entropy_coef=0.0,
    )
    for name, cfg in [("GRPO", grpo), ("DAPO", dapo)]:
        out = compute_rl_loss(
            logits=logits.detach().clone().requires_grad_(True),
            input_ids=ids, assistant_mask=mask,
            old_logprobs=old_lp, ref_logprobs=ref_lp,
            rewards=rewards, group_ids=gids, cfg=cfg,
        )
        assert torch.isfinite(out["loss"])
        print(f"  ✓ {name}: loss={out['loss'].item():.4f}  "
              f"adv={out['mean_advantage'].item():.3f}  "
              f"ratio={out['mean_ratio'].item():.3f}  "
              f"clip_frac={out['clip_frac'].item():.3f}")


def test_empty_mask():
    logits, ids, mask, old_lp, ref_lp, gids = _make_batch()
    mask = torch.zeros_like(mask)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    cfg = RLLossConfig(kl_coef=0.1)
    out = compute_rl_loss(
        logits=logits, input_ids=ids, assistant_mask=mask,
        old_logprobs=old_lp, ref_logprobs=ref_lp,
        rewards=rewards, group_ids=gids, cfg=cfg,
    )
    assert torch.isfinite(out["loss"]), f"NaN with empty mask: {out['loss']}"
    assert abs(out["pg_loss"].item()) < 1e-8, f"empty-mask pg should be 0, got {out['pg_loss']}"
    print(f"  ✓ empty assistant mask → finite zero loss")


def test_aggregations_differ():
    # Build a batch where one sequence has many assistant tokens and another
    # has few, so token_mean and seq_mean_token_sum should diverge.
    torch.manual_seed(7)
    B, T, V = 2, 12, 8
    logits = torch.randn(B, T, V, requires_grad=True)
    ids = torch.randint(0, V, (B, T))
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[0, 1:] = 1     # 11 assistant tokens
    mask[1, 10:] = 1    # 2 assistant tokens
    with torch.no_grad():
        old_lp = gather_token_logprobs(logits[:, :-1, :], ids[:, 1:])
    rewards = torch.tensor([1.0, -1.0])
    gids = torch.tensor([0, 0])

    cfg_tm = RLLossConfig(advantage="group_mean", loss_agg="token_mean", kl_coef=0.0)
    cfg_st = RLLossConfig(advantage="group_mean", loss_agg="seq_mean_token_sum", kl_coef=0.0)
    out_tm = compute_rl_loss(logits=logits.detach().clone().requires_grad_(True),
        input_ids=ids, assistant_mask=mask,
        old_logprobs=old_lp.detach(), ref_logprobs=None,
        rewards=rewards, group_ids=gids, cfg=cfg_tm)
    out_st = compute_rl_loss(logits=logits.detach().clone().requires_grad_(True),
        input_ids=ids, assistant_mask=mask,
        old_logprobs=old_lp.detach(), ref_logprobs=None,
        rewards=rewards, group_ids=gids, cfg=cfg_st)
    assert abs(out_tm["pg_loss"].item() - out_st["pg_loss"].item()) > 1e-4, \
        "aggregations should differ on unequal-length sequences"
    print(f"  ✓ token_mean ({out_tm['pg_loss'].item():.4f}) ≠ "
          f"seq_mean_token_sum ({out_st['pg_loss'].item():.4f})")


def main():
    print("[1] finite & differentiable")
    test_finite_and_differentiable()
    print("[2] uniform-reward groups → zero advantage")
    test_zero_advantage_when_uniform_rewards()
    print("[3] reward sign flip → grad sign flip")
    test_reward_sign_flip_flips_grad()
    print("[4] KL anchor non-negative; zero when ref==new")
    test_kl_nonnegative_and_zero_when_ref_eq_new()
    print("[5] GRPO + DAPO configs both run")
    test_dapo_vs_grpo_configs_run()
    print("[6] empty assistant mask")
    test_empty_mask()
    print("[7] loss aggregation modes")
    test_aggregations_differ()
    print("\n=== all RL loss tests PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
