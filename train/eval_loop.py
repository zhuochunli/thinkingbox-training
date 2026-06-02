"""Lightweight periodic eval for thinkingbox-training.

One sample per case (no grouping), runs the configured rollout + reward path,
returns aggregate metrics. Designed to be cheap: rank-0 only, parallelism set
to len(eval_cases) so the whole eval is one HTTP burst against vLLM.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from thinkingbox.common.config_types import ConfigFile, HydratedTestCase

from train.rewards import score_rollouts
from train.rollout import RolloutRunner

logger = logging.getLogger(__name__)


def _build_runner(tb_cfg: ConfigFile, lora_name: str, concurrency: int) -> RolloutRunner:
    new_cfg = tb_cfg.model_copy(deep=True)
    new_cfg.orchestrator.agent_model.deployment = lora_name
    return RolloutRunner(config=new_cfg, concurrency=concurrency, dump_raw=False)


def run_eval(
    step: int,
    tb_cfg: ConfigFile,
    eval_cases: list[HydratedTestCase],
    lora_name: str,
    reward_fn: str = "binary_test_result",
) -> dict[str, Any]:
    """Roll out 1 sample per eval case against vLLM, return aggregate metrics.

    Cheap on purpose: no tokenization, no policy forward, no grad. The whole
    eval pass = one parallel rollout burst + reward scoring.
    """
    if not eval_cases:
        return {"phase": "eval", "step": step, "n_cases": 0, "skipped": "no_cases"}

    runner = _build_runner(tb_cfg, lora_name, concurrency=len(eval_cases))
    t0 = time.monotonic()
    rollouts_by_uid = asyncio.run(runner.rollout_many(eval_cases, n_samples=1))
    t_eval = time.monotonic() - t0

    flat = []
    uids = []
    for uid in sorted(rollouts_by_uid):
        for r in rollouts_by_uid[uid]:
            flat.append(r)
            uids.append(uid)
    if not flat:
        return {"phase": "eval", "step": step, "n_cases": 0, "skipped": "no_rollouts",
                "t_eval_s": round(t_eval, 2)}

    n_sys_errors = sum(r.is_system_error for r in flat)
    rewards = score_rollouts(flat, name=reward_fn)
    n = len(rewards)
    mean = sum(rewards) / n if n else 0.0
    pass_rate = sum(1 for r in rewards if r > 0) / n if n else 0.0
    per_case = [{"uid": u, "reward": float(r)} for u, r in zip(uids, rewards)]

    return {
        "phase": "eval",
        "step": step,
        "lora_name": lora_name,
        "n_cases": n,
        "n_sys_errors": n_sys_errors,
        "reward_mean": float(mean),
        "pass_rate": float(pass_rate),
        "t_eval_s": round(t_eval, 2),
        "per_case": per_case,
    }


def log_eval(row: dict, log_fh) -> None:
    """Pretty-print eval row to logger + append JSONL."""
    if row.get("skipped"):
        logger.info("eval step=%04d skipped (%s)", row["step"], row["skipped"])
    else:
        logger.info(
            "eval step=%04d  n=%d  reward μ=%.3f  pass=%.2f  sys_err=%d  t=%.1fs  lora=%s",
            row["step"], row["n_cases"], row["reward_mean"], row["pass_rate"],
            row["n_sys_errors"], row["t_eval_s"], row["lora_name"],
        )
    if log_fh is not None:
        log_fh.write(json.dumps(row) + "\n")
        log_fh.flush()
