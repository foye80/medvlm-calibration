from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from typing import Sequence

DEFAULT_SEED = 42
SMOKE_ITEMS = 8
SMOKE_STEPS = 1


@dataclass(frozen=True)
class RuntimeConfig:
    smoke: bool
    seed: int = DEFAULT_SEED
    log_level: str = "INFO"
    smoke_items: int = SMOKE_ITEMS
    smoke_steps: int = SMOKE_STEPS


def add_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--smoke", action="store_true", help="Run a tiny no-network smoke pass.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(smoke=bool(args.smoke), seed=int(args.seed), log_level=str(args.log_level))


def setup_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def limit_for_smoke(items: Sequence[object], smoke: bool, max_items: int = SMOKE_ITEMS) -> Sequence[object]:
    if not smoke:
        return items
    return items[:max_items]


def log_smoke_stub(module_name: str, config: RuntimeConfig, action: str) -> None:
    logger = logging.getLogger(module_name)
    set_seed(config.seed)
    if config.smoke:
        logger.info(
            "smoke mode: %s limited to %s items or %s step",
            action,
            config.smoke_items,
            config.smoke_steps,
        )
    else:
        logger.info("starting %s", action)
