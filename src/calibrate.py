from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.optimize import minimize_scalar

from src.models.score import softmax
from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, setup_logging

logger = logging.getLogger(__name__)

# Bounds for the 1-D temperature search. T<1 sharpens, T>1 softens.
_T_MIN = 0.05
_T_MAX = 20.0


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    objective: float | None = None
    n_items: int = 0


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    return softmax(logits, temperature=temperature)


def negative_log_likelihood(
    option_scores: Sequence[Sequence[float]],
    labels: Sequence[int],
    temperature: float,
) -> float:
    """Mean NLL of the gold option under softmax(scores / temperature)."""
    if len(option_scores) != len(labels):
        raise ValueError("option_scores and labels must have equal length")
    if not labels:
        raise ValueError("labels must be non-empty")
    total = 0.0
    for scores, gold in zip(option_scores, labels, strict=True):
        probs = softmax(scores, temperature=temperature)
        total += -math.log(max(probs[gold], 1e-12))
    return total / len(labels)


def fit_temperature(
    option_scores: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    smoke: bool = False,
) -> TemperatureFit:
    """Fit a single scalar temperature by minimizing NLL of the gold option.

    The scores are per-item option log-likelihoods (length-normalized).
    Must be fit on a held-out calibration split, never on the test labels.
    """
    if smoke:
        return TemperatureFit(temperature=1.0, objective=None, n_items=len(labels))
    if len(option_scores) != len(labels):
        raise ValueError("option_scores and labels must have equal length")
    if not labels:
        raise ValueError("labels must be non-empty")

    result = minimize_scalar(
        lambda t: negative_log_likelihood(option_scores, labels, t),
        bounds=(_T_MIN, _T_MAX),
        method="bounded",
    )
    return TemperatureFit(
        temperature=float(result.x),
        objective=float(result.fun),
        n_items=len(labels),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit post-hoc calibration models.")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "post-hoc calibration")
    if config.smoke:
        logger.info("temperature=%s", fit_temperature([[1.0, 0.0]], [0], smoke=True).temperature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
