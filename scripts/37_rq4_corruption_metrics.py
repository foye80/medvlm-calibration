#!/usr/bin/env python3
"""RQ4b — image-corruption degradation metrics + Fig 3.

For each (model, FT-on-ID adapter, corruption, severity) reads the phase4
corruption prediction CSV and computes point-estimate accuracy / ECE / AURC /
selective-accuracy@70%. Severity 0 is the clean ID baseline. Writes
results/rq4_corruption_metrics.csv and figures/fig3_corruption_degradation.png.

Reuses the metric functions from scripts/phase5_aggregate.py for consistency.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.phase5_aggregate import (  # noqa: E402
    _aurc,
    _ece,
    _parse_json_col,
    _selective_accuracy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS = ["qwen25vl", "internvl", "llavaov", "smolvlm", "medgemma", "huatuo"]
DATASETS = ["vqa_rad", "slake_en", "pathvqa"]
CORRUPTIONS = [
    "gaussian_noise", "gaussian_blur", "motion_blur",
    "brightness_shift", "contrast_shift", "jpeg_compression", "downscale_upscale",
]
SEVERITIES = [1, 2, 3]
RESULTS = "results"


def _cell_metrics(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    confs = df["confidence"].to_numpy(dtype=float)
    correct = df["correct"].to_numpy(dtype=float)
    aurc, _ = _aurc(confs, correct)
    return {
        "n": int(len(df)),
        "accuracy": float(correct.mean()),
        "ece": float(_ece(confs, correct)),
        "aurc": float(aurc),
        "sel_acc_70": float(_selective_accuracy(confs, correct, 0.70)),
    }


def collect() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            # severity 0 = clean ID baseline
            clean = _cell_metrics(f"{RESULTS}/phase4_{model}_{ds}_on_{ds}_test.csv")
            if clean is not None:
                rows.append({"model": model, "train_dataset": ds, "corruption": "clean",
                             "severity": 0, **clean})
            for corr in CORRUPTIONS:
                for sev in SEVERITIES:
                    m = _cell_metrics(
                        f"{RESULTS}/phase4_{model}_{ds}_on_{ds}_test_{corr}_s{sev}.csv"
                    )
                    if m is None:
                        log.warning("missing %s %s %s s%s", model, ds, corr, sev)
                        continue
                    rows.append({"model": model, "train_dataset": ds, "corruption": corr,
                                 "severity": sev, **m})
    return pd.DataFrame(rows)


def make_fig3(df: pd.DataFrame, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Average over corruptions + train datasets → mean metric per (model, severity).
    # Clean (sev 0) averaged over datasets only.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("accuracy", "Accuracy", False), ("ece", "ECE", True),
               ("sel_acc_70", "Selective Acc @70% cov", False)]
    cmap = plt.get_cmap("tab10")
    for ax, (col, title, lower_better) in zip(axes, metrics):
        for mi, model in enumerate(MODELS):
            sub = df[df["model"] == model]
            ys = []
            for sev in [0, 1, 2, 3]:
                v = sub[sub["severity"] == sev][col]
                ys.append(v.mean() if len(v) else np.nan)
            ax.plot([0, 1, 2, 3], ys, marker="o", label=model, color=cmap(mi))
        ax.set_xlabel("Corruption severity (0 = clean)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs severity" + (" (↓ better)" if lower_better else " (↑ better)"))
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("RQ4b — Calibration & selective prediction degrade under image corruption "
                 "(mean over 7 corruptions × 3 ID datasets)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("saved figure → %s", out_path)


def main() -> None:
    df = collect()
    out_csv = f"{RESULTS}/rq4_corruption_metrics.csv"
    df.to_csv(out_csv, index=False)
    log.info("wrote %d rows → %s", len(df), out_csv)

    # Print a compact degradation summary: mean over models+corruptions per severity.
    print("\n=== RQ4b degradation (mean over all models × corruptions × ID datasets) ===")
    summ = df.groupby("severity")[["accuracy", "ece", "aurc", "sel_acc_70"]].mean()
    print(summ.to_string(float_format="%.4f"))

    make_fig3(df, "figures/fig3_corruption_degradation.png")


if __name__ == "__main__":
    main()
