# RECONSTRUCTED FROM CACHE — verify against memory.
# Source: transcript 970dfb5c lines 4624-4729 (verbatim assistant-authored content).
"""Async rollout runner: produces `G` trajectories per prompt for GRPO.

Reuses `thinkingbox.cli.infer.TBWorker` so the rollout matches what `tb infer`
does end-to-end (agent loop + test/judge → `TestResult.reward`).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic_core import to_jsonable_python

from thinkingbox.cli.infer import TBWorker
from thinkingbox.common.chat_types import DecodeResult
from thinkingbox.common.config_types import ConfigFile, HydratedTestCase

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    uid: str
    sample_idx: int
    reward: float
    is_correct: bool
    is_system_error: bool
    finish_reason: str
    messages: list[dict]
    raw_messages: list[dict] | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_decode_result(cls, result: DecodeResult, sample_idx: int) -> "Rollout":
        tr = result.test_result
        reward = float(tr.reward) if tr is not None else 0.0
        is_correct = bool(tr.result) if tr is not None else False
        return cls(
            uid=result.uid,
            sample_idx=sample_idx,
            reward=reward,
            is_correct=is_correct,
            is_system_error=bool(result.is_system_error),
            finish_reason=str(result.finish_reason),
            messages=to_jsonable_python(result.messages),
            raw_messages=result.raw_messages,
            metadata=to_jsonable_python(result.metadata),
        )


class RolloutRunner:
    """Run G rollouts per test case, bounded by a global concurrency semaphore."""

    def __init__(
        self,
        config: ConfigFile,
        concurrency: int = 8,
        dump_raw: bool = True,
    ):
        self.config = config
        self.concurrency = concurrency
        self.dump_raw = dump_raw
        self._sem = asyncio.Semaphore(concurrency)
        # a single TBWorker is stateless across `.work()` calls (each opens its
        # own MCPProxy session) so we can share it
        self._worker = TBWorker(
            config=config,
            skip_test=False,
            skip_agent=False,
            dump_tools=False,
            dump_testcontext=False,
            dump_userllm=False,
            dump_raw=dump_raw,
            debug_test=False,
            linger_sessions=False,
        )

    async def _one(self, tc: HydratedTestCase, sample_idx: int) -> Rollout:
        async with self._sem:
            # deep-copy so per-sample metadata (e.g. repetition idx) doesn't leak
            tc_copy = tc.model_copy(deep=True)
            tc_copy.metadata["rollout_sample_idx"] = sample_idx
            work_result = await self._worker.work(tc_copy)
            return Rollout.from_decode_result(work_result.result, sample_idx)

    async def rollout(self, tc: HydratedTestCase, n_samples: int) -> list[Rollout]:
        """Produce `n_samples` rollouts for a single test case (concurrent)."""
        tasks = [self._one(tc, i) for i in range(n_samples)]
        return await asyncio.gather(*tasks)

    async def rollout_many(
        self,
        tcs: list[HydratedTestCase],
        n_samples: int,
    ) -> dict[str, list[Rollout]]:
        """Run rollouts for every tc; returns {uid: [Rollout, ...]}."""
        tasks = [self._one(tc, i) for tc in tcs for i in range(n_samples)]
        flat: list[Rollout] = await asyncio.gather(*tasks)
        out: dict[str, list[Rollout]] = {}
        for r in flat:
            out.setdefault(r.uid, []).append(r)
        for uid in out:
            out[uid].sort(key=lambda r: r.sample_idx)
        return out
