from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence

from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, setup_logging

logger = logging.getLogger(__name__)


def max_probability(probabilities: Sequence[float]) -> float:
    if not probabilities:
        raise ValueError("probabilities must be non-empty")
    return float(max(probabilities))


def entropy(probabilities: Sequence[float]) -> float:
    if not probabilities:
        raise ValueError("probabilities must be non-empty")
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probabilities))


def parse_verbalized_confidence(text: str) -> float | None:
    match = re.search(r"(?<!\d)(100|[1-9]?\d)(?!\d)", text)
    if match is None:
        return None
    value = int(match.group(1))
    if value < 0 or value > 100:
        return None
    return value / 100.0


def self_consistency_confidence(answers: Sequence[str]) -> float:
    if not answers:
        raise ValueError("answers must be non-empty")
    counts = Counter(answer.strip().lower() for answer in answers)
    return max(counts.values()) / len(answers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute confidence signals.")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "confidence signal extraction")
    if config.smoke:
        probs = [0.8, 0.2]
        logger.info("max_prob=%.3f entropy=%.3f", max_probability(probs), entropy(probs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
