"""Smoke-test the rollout runner: G rollouts per prompt × N prompts → JSONL.

Usage:
    python scripts/demo_rollout.py \
        --config config_training.yaml \
        --dataset $THINKINGBOX_DATA/dataset \
        --train-list data/train_list.yaml \
        --n-prompts 2 --g 4 --concurrency 4 \
        --out /tmp/tb_demo_rollouts.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import statistics
import time
from pathlib import Path

from thinkingbox.cli.common import load_yaml
from thinkingbox.common.config_types import ConfigFile
from thinkingbox.common.http_client import initialize_dns_cache

from train.data_pipeline import hydrate, load_test_list, split_train_eval
from train.rollout import RolloutRunner


async def amain(args):
    config = load_yaml(args.config, ConfigFile)
    initialize_dns_cache()

    all_names = load_test_list(args.train_list)
    train_names, eval_names = split_train_eval(all_names)
    logging.info(
        "train_list=%d  train=%d  eval=%d", len(all_names), len(train_names), len(eval_names)
    )
    logging.info("eval_names = %s", eval_names)

    pick_names = (eval_names if args.use_eval else train_names)[: args.n_prompts]
    logging.info("rolling out %d prompts × g=%d : %s", len(pick_names), args.g, pick_names)

    tcs = hydrate(pick_names, args.dataset, args.agent)
    logging.info("hydrated %d cases", len(tcs))

    runner = RolloutRunner(config=config, concurrency=args.concurrency, dump_raw=True)

    t0 = time.monotonic()
    by_uid = await runner.rollout_many(tcs, n_samples=args.g)
    elapsed = time.monotonic() - t0

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for uid in sorted(by_uid):
            for r in by_uid[uid]:
                f.write(json.dumps(dataclasses.asdict(r), default=str))
                f.write("\n")
                n_total += 1

    print(f"\n=== Rollout summary ({elapsed:.1f}s, {n_total} rollouts) ===")
    for uid in sorted(by_uid):
        rs = by_uid[uid]
        rewards = [r.reward for r in rs]
        errs = sum(r.is_system_error for r in rs)
        finishes = sorted({r.finish_reason for r in rs})
        mean = statistics.mean(rewards)
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        print(
            f"  {uid:<70s}  n={len(rs)}  reward μ={mean:.2f} σ={std:.2f}  "
            f"errors={errs}  finish={','.join(finishes)}"
        )
    print(f"\nwrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_training.yaml")
    p.add_argument(
        "--dataset",
        default=os.environ.get(
            "THINKINGBOX_DATA",
            "/home/azureuser/zhuochun/AI.ThinkingBox.Data",
        )
        + "/dataset",
    )
    p.add_argument("--agent", default="think")
    p.add_argument("--train-list", default="data/train_list.yaml")
    p.add_argument(
        "--use-eval",
        action="store_true",
        help="pick prompts from the eval split instead of train",
    )
    p.add_argument("--n-prompts", type=int, default=2)
    p.add_argument("--g", type=int, default=4, help="rollouts per prompt")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default="/tmp/tb_demo_rollouts.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(amain(parse_args()))
