from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, setup_logging

logger = logging.getLogger(__name__)


def accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have equal length")
    if not labels:
        raise ValueError("labels must be non-empty")
    return sum(int(p == y) for p, y in zip(predictions, labels, strict=True)) / len(labels)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 15,
) -> float:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    if not confidences:
        raise ValueError("confidences must be non-empty")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    n = len(confidences)
    ece = 0.0
    for bin_idx in range(n_bins):
        lower = bin_idx / n_bins
        upper = (bin_idx + 1) / n_bins
        mask = [
            (conf >= lower and (conf < upper or bin_idx == n_bins - 1))
            for conf in confidences
        ]
        count = sum(mask)
        if count == 0:
            continue
        bin_acc = sum(int(ok) for ok, use in zip(correct, mask, strict=True) if use) / count
        bin_conf = sum(conf for conf, use in zip(confidences, mask, strict=True) if use) / count
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute calibration and selective prediction metrics.")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "metric computation")
    if config.smoke:
        logger.info("accuracy=%.3f", accuracy([0, 1, 1], [0, 0, 1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
