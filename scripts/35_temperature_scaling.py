#!/usr/bin/env python3
"""RQ3 — post-hoc temperature scaling.

Fits a single scalar temperature T (minimizing NLL) and reports calibration
before vs after, per (model, condition, eval_dataset).

Protocol note
-------------
The locked study protocol (PROJECT_SPEC §6.3) fits T on the dedicated `calib`
split. Inference on the `calib` split has not been run yet, so this script
provides a leakage-free demonstration: each per-item TEST file is split in
half by a fixed seed; T is fit on the fit-half and ALL metrics are reported
on the held-out eval-half. No item is used for both fitting and evaluation.
When calib-split inference becomes available, point --fit-from at it instead.

Usage:
    python scripts/35_temperature_scaling.py [--results-dir results]
        [--out results/temperature_scaling.csv]
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.calibrate import fit_temperature, negative_log_likelihood  # noqa: E402
from src.models.score import softmax  # noqa: E402
from scripts.phase5_aggregate import _ece, _parse_json_col  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42


def _scores_and_labels(df: pd.DataFrame):
    scores = _parse_json_col(df["normalized_scores"])
    labels = df["gold_idx"].astype(int).tolist()
    # Keep only items whose gold index is in range.
    keep = [(s, y) for s, y in zip(scores, labels) if s and 0 <= y < len(s)]
    scores = [s for s, _ in keep]
    labels = [y for _, y in keep]
    return scores, labels


def _ece_from_scores(scores, labels, temperature):
    confs, correct = [], []
    for s, y in zip(scores, labels):
        p = softmax(s, temperature=temperature)
        pred = int(np.argmax(p))
        confs.append(max(p))
        correct.append(float(pred == y))
    return _ece(np.array(confs), np.array(correct))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/temperature_scaling.csv")
    args = ap.parse_args()

    pattern = os.path.join(args.results_dir, "phase4_*_on_*_test.csv")
    files = sorted(f for f in glob.glob(pattern) if "_s1" not in f and "_s2" not in f
                   and "_s3" not in f and "corrupt" not in f)
    log.info("Found %d clean test files", len(files))

    rng = np.random.default_rng(SEED)
    rows = []
    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.match(
            r"phase4_(.+?)_(zero_shot|vqa_rad|slake_en|pathvqa)_on_"
            r"(vqa_rad|slake_en|pathvqa)_test\.csv",
            fname,
        )
        if m is None:
            continue
        model, cond, eval_ds = m.group(1), m.group(2), m.group(3)
        df = pd.read_csv(fpath)
        scores, labels = _scores_and_labels(df)
        if len(labels) < 20:
            continue

        idx = np.arange(len(labels))
        rng.shuffle(idx)
        half = len(idx) // 2
        fit_idx, eval_idx = idx[:half], idx[half:]
        fit_s = [scores[i] for i in fit_idx]
        fit_y = [labels[i] for i in fit_idx]
        eval_s = [scores[i] for i in eval_idx]
        eval_y = [labels[i] for i in eval_idx]

        ft = fit_temperature(fit_s, fit_y)
        ece_before = _ece_from_scores(eval_s, eval_y, 1.0)
        ece_after = _ece_from_scores(eval_s, eval_y, ft.temperature)
        nll_before = negative_log_likelihood(eval_s, eval_y, 1.0)
        nll_after = negative_log_likelihood(eval_s, eval_y, ft.temperature)

        rows.append(dict(
            model=model, condition=cond, eval_dataset=eval_ds,
            n_eval=len(eval_y), temperature=round(ft.temperature, 4),
            ece_before=round(ece_before, 4), ece_after=round(ece_after, 4),
            nll_before=round(nll_before, 4), nll_after=round(nll_after, 4),
        ))
        log.info("%s/%s→%s  T=%.3f  ECE %.3f→%.3f  NLL %.3f→%.3f",
                 model, cond, eval_ds, ft.temperature,
                 ece_before, ece_after, nll_before, nll_after)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    log.info("Wrote %d rows to %s", len(out), args.out)
    if len(out):
        print("\n=== Temperature scaling summary (mean over cells) ===")
        print(f"mean T              : {out.temperature.mean():.3f}")
        print(f"mean ECE before→after: {out.ece_before.mean():.4f} → {out.ece_after.mean():.4f}")
        print(f"mean NLL before→after: {out.nll_before.mean():.4f} → {out.nll_after.mean():.4f}")
        improved = (out.ece_after < out.ece_before).mean()
        print(f"cells with ECE improved: {improved*100:.0f}%")


if __name__ == "__main__":
    main()
