"""Load test-case lists, do seeded train/eval split, hydrate cases."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import yaml

from thinkingbox.common.config_types import HydratedTestCase
from thinkingbox.common.hydrator import iter_cases_by_names

# default eval split — small, fixed, used both during training validation and
# in the smoke_test script's seeding helper
DEFAULT_EVAL_SIZE = 5
DEFAULT_SEED = 42


def load_test_list(path: str | Path) -> list[str]:
    """Load a YAML list of 'filename:testname' strings."""
    path = Path(path).expanduser()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a YAML list, got {type(data).__name__}")
    return [str(x) for x in data]


def split_train_eval(
    names: Iterable[str],
    eval_size: int = DEFAULT_EVAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], list[str]]:
    """Deterministically split a name list into (train, eval).

    Eval set is `random.Random(seed).sample(sorted(names), eval_size)`.
    Train set is `names \\ eval` preserving the original order.
    """
    names = list(names)
    pool = sorted(set(names))
    if eval_size > len(pool):
        raise ValueError(f"eval_size={eval_size} > pool size {len(pool)}")
    eval_names = random.Random(seed).sample(pool, eval_size)
    eval_set = set(eval_names)
    train_names = [n for n in names if n not in eval_set]
    return train_names, eval_names


def hydrate(
    names: Iterable[str],
    dataset_dir: str | Path,
    agent: str,
    strict: bool = True,
) -> list[HydratedTestCase]:
    """Resolve names → HydratedTestCase objects via the thinkingbox hydrator."""
    return list(
        iter_cases_by_names(
            list(names),
            base_dir=str(Path(dataset_dir).expanduser()),
            agent=agent,
            strict=strict,
        )
    )
