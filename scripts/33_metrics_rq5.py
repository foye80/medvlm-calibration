#!/usr/bin/env python3
"""RQ5 — per-modality calibration metrics and Figure 4.

Reads rq5_*_omnimedvqa_test.csv files from results/, joins with
data/omnimedvqa_items.csv for modality labels, computes per-modality
ECE/AURC/accuracy, writes results/rq5_modality_metrics.csv, and
generates figures/fig4_modality_calibration.png.

Usage:
    python scripts/33_metrics_rq5.py [--results-dir results] [--items-csv data/omnimedvqa_items.csv]
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 1000
MIN_ITEMS_PER_MODALITY = 20  # skip modality-model cell if too few items

MODEL_DISPLAY = {
    "qwen25vl": "Qwen2.5-VL",
    "internvl": "InternVL2.5",
    "llavaov": "LLaVA-OV",
    "smolvlm": "SmolVLM",
    "medgemma": "MedGemma",
    "huatuo": "HuatuoGPT",
}
MODEL_ORDER = ["qwen25vl", "internvl", "llavaov", "smolvlm", "medgemma", "huatuo"]

MODALITY_ABBREV = {
    "MR (Mag-netic Resonance Imaging)": "MRI",
    "CT(Computed Tomography)": "CT",
    "ultrasound": "Ultrasound",
    "X-Ray": "X-Ray",
    "Dermoscopy": "Dermoscopy",
    "Microscopy Images": "Microscopy",
    "Fundus Photography": "Fundus",
    "OCT (Optical Coherence Tomography": "OCT",
}


# ──────────────────────────── metric functions ───────────────────────────────

def _ece(confs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    n = len(confs)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (confs >= lo) & (confs < hi) if b < n_bins - 1 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - confs[mask].mean())
    return float(ece)


def _brier(probs: np.ndarray, gold_idx: np.ndarray) -> float:
    n, k = probs.shape
    targets = np.zeros_like(probs)
    targets[np.arange(n), gold_idx] = 1.0
    return float(np.mean(np.sum((probs - targets) ** 2, axis=1)))


def _nll(probs: np.ndarray, gold_idx: np.ndarray) -> float:
    p_gold = probs[np.arange(len(gold_idx)), gold_idx]
    return float(-np.mean(np.log(np.clip(p_gold, 1e-12, 1.0))))


def _aurc(confs: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    n = len(confs)
    order = np.argsort(-confs)
    correct_sorted = correct[order]
    cum_correct = np.cumsum(correct_sorted)
    ks = np.arange(1, n + 1)
    risks = 1.0 - cum_correct / ks
    aurc = float(risks.mean())
    m = int(correct.sum())
    oracle_risks = np.zeros(n)
    for k in range(m + 1, n + 1):
        oracle_risks[k - 1] = (k - m) / k
    e_aurc = aurc - float(oracle_risks.mean())
    return aurc, e_aurc


def _bootstrap_ci(fn, *arrays, n: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    point = fn(*arrays)
    samples = []
    size = len(arrays[0])
    for _ in range(n):
        idx = rng.integers(0, size, size=size)
        resampled = [a[idx] for a in arrays]
        try:
            samples.append(fn(*resampled))
        except Exception:
            continue
    if not samples:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(samples, 100 * alpha / 2)), float(np.percentile(samples, 100 * (1 - alpha / 2)))


def _parse_json_col(series: pd.Series) -> list[list[float]]:
    out = []
    for v in series:
        if isinstance(v, list):
            out.append(v)
        else:
            try:
                out.append(json.loads(str(v)))
            except Exception:
                try:
                    out.append(ast.literal_eval(str(v)))
                except Exception:
                    out.append([])
    return out


# ──────────────────────────── per-cell computation ───────────────────────────

def compute_modality_cell(df: pd.DataFrame) -> dict:
    n = len(df)
    correct = df["correct"].astype(int).values.astype(float)
    confs = df["confidence"].astype(float).values

    probs_raw = _parse_json_col(df["probabilities"])
    max_cols = max((len(p) for p in probs_raw), default=2)
    probs = np.zeros((n, max_cols))
    for i, p in enumerate(probs_raw):
        if p:
            row_sum = sum(p)
            if row_sum > 0:
                probs[i, : len(p)] = [x / row_sum for x in p]
            else:
                probs[i, : len(p)] = p
    gold_idx = df["gold_idx"].astype(int).values

    ece_val, ece_lo, ece_hi = _bootstrap_ci(_ece, confs, correct)
    aurc_val, e_aurc_val = _aurc(confs, correct)
    aurc_b, aurc_lo, aurc_hi = _bootstrap_ci(lambda c, r: _aurc(c, r)[0], confs, correct)

    return {
        "n": n,
        "accuracy": float(correct.mean()),
        "ece": ece_val,
        "ece_ci_lo": ece_lo,
        "ece_ci_hi": ece_hi,
        "brier": _brier(probs, gold_idx),
        "nll": _nll(probs, gold_idx),
        "aurc": aurc_val,
        "aurc_ci_lo": aurc_lo,
        "aurc_ci_hi": aurc_hi,
        "e_aurc": e_aurc_val,
    }


# ──────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--items-csv", default="data/omnimedvqa_items.csv")
    parser.add_argument("--out-csv", default="results/rq5_modality_metrics.csv")
    parser.add_argument("--out-fig", default="figures/fig4_modality_calibration.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    items_csv = Path(args.items_csv)

    if not items_csv.exists():
        raise FileNotFoundError(f"items CSV not found: {items_csv}. Run 31_prepare_omnimedvqa_full.py first.")

    # Load modality map uid → modality
    uid_modality: dict[str, str] = {}
    with open(items_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            uid_modality[row["uid"]] = row["modality"]
    log.info("loaded %d uid→modality mappings", len(uid_modality))

    # Discover result CSVs
    pattern = str(results_dir / "rq5_*_omnimedvqa_test.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"no files matching {pattern}. Run 32_infer_rq5_omnimedvqa.sh first.")
    log.info("found %d result CSVs", len(csv_files))

    # Parse model/condition from filename: rq5_{model}_{condition}_omnimedvqa_test.csv
    file_re = re.compile(r"rq5_(.+?)_(zero_shot|ft_\w+)_omnimedvqa_test\.csv$")

    rows_out: list[dict] = []

    for csv_file in csv_files:
        m = file_re.search(Path(csv_file).name)
        if not m:
            log.warning("cannot parse filename %s — skip", csv_file)
            continue
        model, condition = m.group(1), m.group(2)
        log.info("processing model=%s condition=%s", model, condition)

        df = pd.read_csv(csv_file)
        # Attach modality
        df["modality"] = df["uid"].map(uid_modality)
        missing = df["modality"].isna().sum()
        if missing > 0:
            log.warning("  %d rows have no modality match — dropping", missing)
            df = df.dropna(subset=["modality"])

        for modality, grp in df.groupby("modality"):
            if len(grp) < MIN_ITEMS_PER_MODALITY:
                log.info("  skip modality=%s n=%d (< %d)", modality, len(grp), MIN_ITEMS_PER_MODALITY)
                continue
            cell = compute_modality_cell(grp)
            rows_out.append({
                "model": model,
                "condition": condition,
                "modality": modality,
                **cell,
            })
            log.info("  modality=%-45s n=%4d acc=%.3f ece=%.4f aurc=%.4f",
                     modality, cell["n"], cell["accuracy"], cell["ece"], cell["aurc"])

    if not rows_out:
        log.error("no results computed — nothing to write")
        return

    out_df = pd.DataFrame(rows_out)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    log.info("wrote %d rows → %s", len(out_df), args.out_csv)

    # ── Figure 4 ─────────────────────────────────────────────────────────────
    _make_fig4(out_df, Path(args.out_fig))


def _make_fig4(df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        log.warning("matplotlib not available — skipping figure")
        return

    conditions = sorted(df["condition"].unique())
    n_cond = len(conditions)

    # Abbreviate modalities for display
    df = df.copy()
    df["modality_abbrev"] = df["modality"].map(lambda m: MODALITY_ABBREV.get(m, m))

    modality_order = sorted(df["modality_abbrev"].unique())
    model_labels = [MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER]

    fig, axes = plt.subplots(
        1, n_cond,
        figsize=(6 * n_cond, 5),
        squeeze=False,
    )

    vmin, vmax = 0.0, df["ece"].quantile(0.95)
    cmap = "YlOrRd"

    for col, cond in enumerate(conditions):
        ax = axes[0][col]
        sub = df[df["condition"] == cond]

        # Build matrix: rows=models, cols=modalities
        matrix = np.full((len(MODEL_ORDER), len(modality_order)), np.nan)
        for ri, model in enumerate(MODEL_ORDER):
            for ci, mod in enumerate(modality_order):
                cell = sub[(sub["model"] == model) & (sub["modality_abbrev"] == mod)]
                if len(cell) == 1:
                    matrix[ri, ci] = cell["ece"].values[0]

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        # Annotate cells
        for ri in range(len(MODEL_ORDER)):
            for ci in range(len(modality_order)):
                val = matrix[ri, ci]
                if not np.isnan(val):
                    text_color = "white" if val > (vmin + vmax) * 0.6 else "black"
                    ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                            fontsize=7.5, color=text_color)

        ax.set_xticks(range(len(modality_order)))
        ax.set_xticklabels(modality_order, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels(model_labels, fontsize=9)
        ax.set_title(f"ECE ({cond.replace('_', ' ')})", fontsize=11, fontweight="bold")

        plt.colorbar(im, ax=ax, label="ECE", fraction=0.046, pad=0.04)

    fig.suptitle(
        "Figure 4 — Per-modality Calibration Error (OmniMedVQA)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("saved figure → %s", out_path)

    # ── also save AURC heatmap ────────────────────────────────────────────────
    aurc_path = out_path.parent / out_path.name.replace("fig4_", "fig4b_aurc_")
    fig2, axes2 = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), squeeze=False)
    vmin_a = df["aurc"].min()
    vmax_a = df["aurc"].quantile(0.95)
    for col, cond in enumerate(conditions):
        ax = axes2[0][col]
        sub = df[df["condition"] == cond]
        matrix = np.full((len(MODEL_ORDER), len(modality_order)), np.nan)
        for ri, model in enumerate(MODEL_ORDER):
            for ci, mod in enumerate(modality_order):
                cell = sub[(sub["model"] == model) & (sub["modality_abbrev"] == mod)]
                if len(cell) == 1:
                    matrix[ri, ci] = cell["aurc"].values[0]
        im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=vmin_a, vmax=vmax_a)
        for ri in range(len(MODEL_ORDER)):
            for ci in range(len(modality_order)):
                val = matrix[ri, ci]
                if not np.isnan(val):
                    ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                            fontsize=7.5, color="black")
        ax.set_xticks(range(len(modality_order)))
        ax.set_xticklabels(modality_order, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels(model_labels, fontsize=9)
        ax.set_title(f"AURC ({cond.replace('_', ' ')})", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="AURC", fraction=0.046, pad=0.04)
    fig2.suptitle("Figure 4b — Per-modality AURC (OmniMedVQA)", fontsize=12, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2.savefig(aurc_path, dpi=150, bbox_inches="tight")
    log.info("saved AURC figure → %s", aurc_path)


if __name__ == "__main__":
    main()
