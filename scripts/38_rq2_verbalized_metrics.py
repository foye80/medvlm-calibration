#!/usr/bin/env python3
"""RQ2 — verbalized-confidence metrics.

For each verbalized cell, joins by uid to the matching phase4 clean prediction
CSV (for per-item correctness), then reports parse rate, mean verbalized
confidence, and error-detection AUROC (verbalized confidence vs correctness).
Writes results/rq2_verbalized_metrics.csv.

Mapping: verbalized condition 'zero_shot' -> prediction 'zero_shot';
'ft_<ds>' -> prediction '<ds>' (FT clean cells are named by train dataset).
"""
from __future__ import annotations

import glob
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.phase5_aggregate import _auroc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS = "results"
MODELS = ["qwen25vl", "internvl", "llavaov", "smolvlm", "medgemma", "huatuo"]


def _parse_name(path: str):
    base = os.path.basename(path)
    m = re.match(r"verbalized_(.+?)_(zero_shot|ft_[a-z_]+)_on_([a-z_]+)_test\.csv", base)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _pred_condition(cond: str) -> str:
    return "zero_shot" if cond == "zero_shot" else cond[len("ft_"):]


def main() -> None:
    rows = []
    for path in sorted(glob.glob(f"{RESULTS}/verbalized_*_test.csv")):
        parsed = _parse_name(path)
        if parsed is None:
            log.warning("cannot parse %s", path)
            continue
        model, cond, ds = parsed
        vdf = pd.read_csv(path)
        n = len(vdf)
        parse_rate = float(vdf["parsed"].mean()) if n else 0.0

        pred_path = f"{RESULTS}/phase4_{model}_{_pred_condition(cond)}_on_{ds}_test.csv"
        auroc = np.nan
        mean_conf = np.nan
        n_joined = 0
        if os.path.exists(pred_path):
            pdf = pd.read_csv(pred_path)[["uid", "correct"]]
            merged = vdf.merge(pdf, on="uid", how="inner")
            merged = merged[merged["parsed"] == 1].dropna(subset=["verbalized_confidence"])
            n_joined = len(merged)
            if n_joined:
                conf = merged["verbalized_confidence"].to_numpy(dtype=float)
                correct = merged["correct"].to_numpy(dtype=float)
                mean_conf = float(conf.mean())
                if 0 < correct.sum() < len(correct):  # need both classes
                    auroc = float(_auroc(conf, correct))
        else:
            log.warning("no prediction file for join: %s", pred_path)

        rows.append({
            "model": model, "condition": cond, "dataset": ds,
            "n": n, "parse_rate": round(parse_rate, 4),
            "n_joined": n_joined, "mean_verbalized_conf": mean_conf,
            "auroc_verbalized": auroc,
        })
        log.info("%s %s on %s: parse_rate=%.3f mean_conf=%s auroc=%s",
                 model, cond, ds, parse_rate,
                 "NA" if np.isnan(mean_conf) else f"{mean_conf:.3f}",
                 "NA" if np.isnan(auroc) else f"{auroc:.3f}")

    df = pd.DataFrame(rows)
    out = f"{RESULTS}/rq2_verbalized_metrics.csv"
    df.to_csv(out, index=False)
    log.info("wrote %d rows -> %s", len(df), out)

    print("\n=== Verbalized confidence per model (mean over conditions) ===")
    summ = df.groupby("model").agg(
        parse_rate=("parse_rate", "mean"),
        mean_conf=("mean_verbalized_conf", "mean"),
        auroc=("auroc_verbalized", "mean"),
    )
    print(summ.to_string(float_format="%.3f"))
    print("\n(compare auroc_verbalized to optsoftmax/entropy ~0.71 from master_metrics)")


if __name__ == "__main__":
    main()
