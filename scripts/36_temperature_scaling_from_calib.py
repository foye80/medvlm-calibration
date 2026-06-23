#!/usr/bin/env python3
"""Fit temperature on calib split and evaluate on clean test split."""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase5_aggregate import _ece, _parse_json_col  # noqa: E402
from src.calibrate import fit_temperature, negative_log_likelihood  # noqa: E402
from src.models.score import softmax  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _scores_and_labels(df: pd.DataFrame) -> tuple[list[list[float]], list[int]]:
    scores = _parse_json_col(df["normalized_scores"])
    labels = df["gold_idx"].astype(int).tolist()
    keep = [(s, y) for s, y in zip(scores, labels, strict=True) if s and 0 <= y < len(s)]
    return [s for s, _ in keep], [y for _, y in keep]


def _metrics_from_scores(
    scores: list[list[float]], labels: list[int], temperature: float
) -> tuple[float, float, float]:
    confs: list[float] = []
    correct: list[float] = []
    for item_scores, gold in zip(scores, labels, strict=True):
        probs = softmax(item_scores, temperature=temperature)
        pred = int(np.argmax(probs))
        confs.append(float(max(probs)))
        correct.append(float(pred == gold))
    accuracy = float(np.mean(correct))
    ece = _ece(np.array(confs), np.array(correct))
    nll = negative_log_likelihood(scores, labels, temperature)
    return accuracy, ece, nll


def _parse_test_name(path: str) -> tuple[str, str, str] | None:
    name = os.path.basename(path)
    match = re.match(
        r"phase4_(.+?)_(zero_shot|vqa_rad|slake_en|pathvqa)_on_"
        r"(vqa_rad|slake_en|pathvqa)_test\.csv$",
        name,
    )
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/temperature_scaling_calib.csv")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "phase4_*_on_*_test.csv")
    test_files = sorted(
        path for path in glob.glob(pattern)
        if "_s1" not in path and "_s2" not in path and "_s3" not in path
    )
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for test_path in test_files:
        parsed = _parse_test_name(test_path)
        if parsed is None:
            continue
        model, condition, eval_dataset = parsed
        calib_path = os.path.join(
            args.results_dir,
            f"phase5_calib_{model}_{condition}_on_{eval_dataset}_calib.csv",
        )
        if not os.path.exists(calib_path):
            missing.append(calib_path)
            continue

        calib_scores, calib_labels = _scores_and_labels(pd.read_csv(calib_path))
        test_scores, test_labels = _scores_and_labels(pd.read_csv(test_path))
        if not calib_labels or not test_labels:
            logger.warning("Skipping empty fit/eval cell calib=%s test=%s", calib_path, test_path)
            continue

        fit = fit_temperature(calib_scores, calib_labels)
        acc_before, ece_before, nll_before = _metrics_from_scores(test_scores, test_labels, 1.0)
        acc_after, ece_after, nll_after = _metrics_from_scores(
            test_scores, test_labels, fit.temperature
        )
        rows.append(
            {
                "model": model,
                "condition": condition,
                "eval_dataset": eval_dataset,
                "n_calib": len(calib_labels),
                "n_test": len(test_labels),
                "temperature": fit.temperature,
                "calib_nll": fit.objective,
                "accuracy_before": acc_before,
                "accuracy_after": acc_after,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "nll_before": nll_before,
                "nll_after": nll_after,
            }
        )
        logger.info(
            "%s/%s on %s: T=%.4f ECE %.4f -> %.4f NLL %.4f -> %.4f",
            model,
            condition,
            eval_dataset,
            fit.temperature,
            ece_before,
            ece_after,
            nll_before,
            nll_after,
        )

    if missing:
        logger.warning("Missing %d calib files", len(missing))
        for path in missing[:20]:
            logger.warning("missing calib file: %s", path)
        if args.require_all:
            raise SystemExit(f"missing {len(missing)} calib files")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.6f")
    logger.info("Wrote %d rows to %s", len(out), args.out)
    if len(out):
        print("\n=== Calib-fit temperature scaling summary ===")
        print(f"completed cells          : {len(out)}")
        print(f"mean T                   : {out.temperature.mean():.3f}")
        print(f"mean ECE before -> after : {out.ece_before.mean():.4f} -> {out.ece_after.mean():.4f}")
        print(f"mean NLL before -> after : {out.nll_before.mean():.4f} -> {out.nll_after.mean():.4f}")
        print(f"ECE improved cells       : {(out.ece_after < out.ece_before).mean() * 100:.1f}%")
        print(f"NLL improved cells       : {(out.nll_after < out.nll_before).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
