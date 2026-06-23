#!/usr/bin/env python3
"""Model reliability fingerprint figure (macaron-pastel, IJMI/Elsevier house style).

Panel A: 4 small bar sub-panels (Accuracy, ECE, AURC, error-detection AUROC),
         one bar per model, fine-tuned in-distribution macro-averages, 95% CI,
         value printed inside the bar, paired-bootstrap significance brackets.
Panel B: radar over 5 within-metric-normalised axes (relative ranking).

Numbers are recomputed from the stored per-item predictions with the SAME metric
definitions used to build results/master_metrics.csv, so the figure matches the tables.
Outputs vector PDF + SVG (+ PNG and a greyscale check PNG).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGS = REPO / "figures"

# ───────────────────────── metric defs (verbatim from phase5_aggregate) ──────
def _ece(confs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    n = len(confs)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        if b == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        c = int(mask.sum())
        if c == 0:
            continue
        ece += (c / n) * abs(correct[mask].mean() - confs[mask].mean())
    return float(ece)


def _aurc(confs: np.ndarray, correct: np.ndarray) -> float:
    n = len(confs)
    order = np.argsort(-confs)
    cum = np.cumsum(correct[order])
    ks = np.arange(1, n + 1)
    return float((1.0 - cum / ks).mean())


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-scores)
    ls = labels[order]
    n_pos, n_neg = ls.sum(), len(ls) - ls.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(ls) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ls) / n_neg])
    return float(np.trapezoid(tpr, fpr))


# ───────────────────────── models, palette, markers ──────────────────────────
GENERAL = ["qwen25vl", "internvl", "llavaov", "smolvlm"]
MEDICAL = ["medgemma", "huatuo"]
MODELS = GENERAL + MEDICAL
LABEL = {
    "qwen25vl": "Qwen2.5-VL", "internvl": "InternVL2.5", "llavaov": "LLaVA-OV",
    "smolvlm": "SmolVLM", "medgemma": "MedGemma", "huatuo": "HuatuoGPT-V",
}
FILL = {
    "qwen25vl": "#E6B0AB", "internvl": "#F4DE9E", "llavaov": "#AECFE0",
    "smolvlm": "#BBD3A4", "medgemma": "#CBBBDD", "huatuo": "#E8C2A0",
}
STROKE = {
    "qwen25vl": "#C0726B", "internvl": "#D9B85C", "llavaov": "#7FB0C8",
    "smolvlm": "#8DB173", "medgemma": "#9E86BE", "huatuo": "#C99A6B",
}
MARKER = {
    "qwen25vl": "o", "internvl": "s", "llavaov": "^",
    "smolvlm": "D", "medgemma": "*", "huatuo": "P",
}
DATASETS = ["vqa_rad", "slake_en", "pathvqa"]
INK = "#2b2b2b"
N_BOOT = 1000
SEED = 42

# ───────────────────────── load per-item in-distribution cells ───────────────
def load_cell(model: str, ds: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"phase4_{model}_{ds}_on_{ds}_test.csv")
    df = df[["uid", "correct", "confidence"]].copy()
    df["correct"] = df["correct"].astype(float)
    df["confidence"] = df["confidence"].astype(float)
    return df


CELLS = {m: {ds: load_cell(m, ds) for ds in DATASETS} for m in MODELS}


def cell_metric(df: pd.DataFrame, metric: str) -> float:
    c = df["correct"].values
    p = df["confidence"].values
    if metric == "accuracy":
        return float(c.mean())
    if metric == "ece":
        return _ece(p, c)
    if metric == "aurc":
        return _aurc(p, c)
    if metric == "auroc":
        return _auroc(-p, 1 - c)
    raise ValueError(metric)


def macro(model: str, metric: str) -> float:
    return float(np.mean([cell_metric(CELLS[model][ds], metric) for ds in DATASETS]))


def macro_ci(model: str, metric: str) -> tuple[float, float, float]:
    """Macro-average over 3 datasets; CI by resampling items within each dataset."""
    rng = np.random.default_rng(SEED)
    arrs = {ds: (CELLS[model][ds]["correct"].values,
                 CELLS[model][ds]["confidence"].values) for ds in DATASETS}
    point = macro(model, metric)
    samples = []
    for _ in range(N_BOOT):
        vals = []
        for ds in DATASETS:
            c, p = arrs[ds]
            idx = rng.integers(0, len(c), len(c))
            cb, pb = c[idx], p[idx]
            if metric == "accuracy":
                vals.append(cb.mean())
            elif metric == "ece":
                vals.append(_ece(pb, cb))
            elif metric == "aurc":
                vals.append(_aurc(pb, cb))
            elif metric == "auroc":
                vals.append(_auroc(-pb, 1 - cb))
        samples.append(np.mean(vals))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_sig(m1: str, m2: str, metric: str) -> str:
    """Paired bootstrap of macro metric difference on shared items. Returns '', '*', '**'."""
    rng = np.random.default_rng(SEED)
    shared = {}
    for ds in DATASETS:
        a = CELLS[m1][ds].set_index("uid")
        b = CELLS[m2][ds].set_index("uid")
        common = a.index.intersection(b.index)
        shared[ds] = (a.loc[common, "correct"].values, a.loc[common, "confidence"].values,
                      b.loc[common, "correct"].values, b.loc[common, "confidence"].values)

    def md(resample):
        d = []
        for ds in DATASETS:
            ca, pa, cb, pb = shared[ds]
            if resample is not None:
                idx = resample[ds]
                ca, pa, cb, pb = ca[idx], pa[idx], cb[idx], pb[idx]
            f = {"accuracy": lambda c, p: c.mean(), "ece": _ece, "aurc": _aurc,
                 "auroc": lambda c, p: _auroc(-p, 1 - c)}[metric]
            if metric in ("ece", "aurc"):
                d.append(f(pa, ca) - f(pb, cb))
            else:
                d.append(f(ca, pa) - f(cb, pb))
        return np.mean(d)

    diffs = []
    for _ in range(N_BOOT):
        resample = {ds: rng.integers(0, len(shared[ds][0]), len(shared[ds][0])) for ds in DATASETS}
        diffs.append(md(resample))
    lo95, hi95 = np.percentile(diffs, [2.5, 97.5])
    lo99, hi99 = np.percentile(diffs, [0.5, 99.5])
    if lo99 > 0 or hi99 < 0:
        return "**"
    if lo95 > 0 or hi95 < 0:
        return "*"
    return ""


# ───────────────────────── compute table ─────────────────────────────────────
METRICS_A = [
    ("accuracy", "Accuracy", "higher is better"),
    ("ece", "ECE", "lower is better"),
    ("aurc", "AURC", "lower is better"),
    ("auroc", "Error-detection AUROC", "higher is better"),
]
VAL = {met: {m: macro_ci(m, met) for m in MODELS} for met, _, _ in METRICS_A}

# cross-dataset ECE (robustness): macro over the 6 off-diagonal cells per model
master = pd.read_csv(RESULTS / "master_metrics.csv")
cross_ece = {}
for m in MODELS:
    sub = master[(master.model == m) & (master.condition != "zero_shot") &
                 (master.condition != master.eval_dataset)]
    cross_ece[m] = float(sub["ece"].mean())
indist_ece = {m: VAL["ece"][m][0] for m in MODELS}
ece_increase = {m: cross_ece[m] - indist_ece[m] for m in MODELS}

print("model      acc    ece    aurc   auroc  crossECE  dECE")
for m in MODELS:
    print(f"{m:10s} {VAL['accuracy'][m][0]:.3f}  {VAL['ece'][m][0]:.3f}  "
          f"{VAL['aurc'][m][0]:.3f}  {VAL['auroc'][m][0]:.3f}  {cross_ece[m]:.3f}     {ece_increase[m]:.3f}")

# radar "goodness" then within-axis min-max to [0,1]
RADAR_AXES = ["Accuracy", "Calibration\n(1−ECE)", "Selective pred.\n(1−AURC)",
              "Error detection\n(AUROC)", "Robustness\n(shift)"]
goodness = {
    "Accuracy": {m: VAL["accuracy"][m][0] for m in MODELS},
    RADAR_AXES[1]: {m: -VAL["ece"][m][0] for m in MODELS},
    RADAR_AXES[2]: {m: -VAL["aurc"][m][0] for m in MODELS},
    RADAR_AXES[3]: {m: VAL["auroc"][m][0] for m in MODELS},
    RADAR_AXES[4]: {m: -ece_increase[m] for m in MODELS},
}
radar_norm = {}
for ax_name, gd in goodness.items():
    vals = np.array([gd[m] for m in MODELS])
    lo, hi = vals.min(), vals.max()
    radar_norm[ax_name] = {m: float((gd[m] - lo) / (hi - lo)) if hi > lo else 0.5 for m in MODELS}

# ───────────────────────── plotting helpers ──────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.facecolor": "white",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
})

XPOS = [0, 1, 2, 3, 4.7, 5.7]          # general block, gap, medical block
XMAP = dict(zip(MODELS, XPOS))


def draw_break(ax, y0):
    """Small axis-break mark near the bottom of a truncated y-axis."""
    dx, dy = 0.012, 0.012
    for x in (0.0,):
        ax.plot([x - dx, x + dx], [y0 - dy, y0 + dy], transform=ax.get_yaxis_transform(),
                clip_on=False, color="#444444", lw=0.8, zorder=20)
        ax.plot([x - dx, x + dx], [y0 + dy - 0.006, y0 + 3 * dy - 0.006],
                transform=ax.get_yaxis_transform(), clip_on=False, color="#444444", lw=0.8, zorder=20)


def bar_panel(ax, metric, title, sub, ylim, ybase, brackets):
    truncated = ybase > 0
    for m in MODELS:
        pt, lo, hi = VAL[metric][m]
        x = XMAP[m]
        ax.bar(x, pt - ybase, bottom=ybase, width=0.82, color=FILL[m],
               edgecolor=STROKE[m], linewidth=1.1, zorder=3)
        ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="none", ecolor="black",
                    elinewidth=0.9, capsize=2.4, capthick=0.9, zorder=5)
        # value printed above the bar (above the upper CI cap), rotated vertical
        ax.text(x, hi + (ylim[1] - ybase) * 0.025, f"{pt:.2f}", ha="center", va="bottom",
                rotation=90, fontsize=6.2, color="#222222", zorder=6)
    ax.set_xticks(list(XMAP.values()))
    ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=40, ha="right", fontsize=6.2)
    ax.set_ylim(ybase, ylim[1])
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=18)
    ax.text(0.5, 1.045, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.0, color="#777777")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.tick_params(axis="x", length=0)
    if truncated:
        draw_break(ax, ybase)
    # significance brackets
    top = ylim[1]
    for j, (a, b, code) in enumerate(brackets):
        xa, xb = XMAP[a], XMAP[b]
        ha = VAL[metric][a][2]
        hb = VAL[metric][b][2]
        ybr = max(ha, hb) + (top - ybase) * (0.18 + 0.15 * j)
        ax.plot([xa, xa, xb, xb], [ybr - (top - ybase) * 0.02, ybr, ybr, ybr - (top - ybase) * 0.02],
                color="#333333", lw=0.8, zorder=8)
        ax.text((xa + xb) / 2, ybr, code, ha="center", va="bottom", fontsize=8, color="#333333")


def radar(ax):
    K = len(RADAR_AXES)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    ang_closed = np.concatenate([angles, angles[:1]])
    # rings
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(np.linspace(0, 2 * np.pi, 200), [r] * 200, color="#dddddd", lw=0.6, zorder=1)
        ax.text(np.pi / 2, r, f"{r:.1f}", fontsize=5.6, color="#999999",
                ha="center", va="bottom", zorder=2)
    for a in angles:
        ax.plot([a, a], [0, 1.0], color="#dddddd", lw=0.6, zorder=1)
    for m in MODELS:
        vals = np.array([radar_norm[ax_name][m] for ax_name in RADAR_AXES])
        vc = np.concatenate([vals, vals[:1]])
        ax.plot(ang_closed, vc, color=STROKE[m], lw=1.3, zorder=4,
                marker=MARKER[m], markersize=5.5, markerfacecolor=FILL[m],
                markeredgecolor=STROKE[m], markeredgewidth=0.8, clip_on=False)
        ax.fill(ang_closed, vc, color=FILL[m], alpha=0.13, zorder=3)
    ax.set_xticks(angles)
    ax.set_xticklabels(RADAR_AXES, fontsize=6.8)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.08)
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)


# ───────────────────────── significance contrasts ───────────────────────────
# Key, pre-registered contrasts only (paired bootstrap on shared items).
worst_ece = max(MODELS, key=lambda m: VAL["ece"][m][0])     # most miscalibrated
best_ece = min(MODELS, key=lambda m: VAL["ece"][m][0])      # best calibrated
best_general_ece = min(GENERAL, key=lambda m: VAL["ece"][m][0])
ece_brackets = []
for a, b in [(best_ece, worst_ece), (best_general_ece, "medgemma")]:
    code = paired_sig(a, b, "ece")
    if code:
        ece_brackets.append((a, b, code))
print("ECE brackets:", ece_brackets,
      "| best", best_ece, "worst", worst_ece, "bestGen", best_general_ece)

# ───────────────────────── figure ────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 7.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.55], hspace=0.62, wspace=0.42,
                      left=0.07, right=0.97, top=0.91, bottom=0.06)

panel_cfg = {
    "accuracy": dict(ylim=(0.5, 1.0), ybase=0.5, brackets=[]),
    "ece": dict(ylim=(0.0, 0.40), ybase=0.0, brackets=ece_brackets),
    "aurc": dict(ylim=(0.0, 0.40), ybase=0.0, brackets=[]),
    "auroc": dict(ylim=(0.5, 1.0), ybase=0.5, brackets=[]),
}
for i, (met, title, sub) in enumerate(METRICS_A):
    ax = fig.add_subplot(gs[0, i])
    c = panel_cfg[met]
    bar_panel(ax, met, title, sub, c["ylim"], c["ybase"], c["brackets"])

axr = fig.add_subplot(gs[1, 1:3], projection="polar")
radar(axr)

# legend (model colour + marker), placed to the right of the radar
handles = [plt.Line2D([0], [0], marker=MARKER[m], color=STROKE[m], markerfacecolor=FILL[m],
                      markeredgecolor=STROKE[m], markersize=7, lw=1.3,
                      label=LABEL[m] + (" (med)" if m in MEDICAL else ""))
           for m in MODELS]
fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.99, 0.30),
           fontsize=6.8, frameon=False, handlelength=1.6, labelspacing=0.7,
           title="Model", title_fontsize=7.2)

# Panel labels
fig.text(0.012, 0.95, "A", fontsize=14, fontweight="bold")
fig.text(0.012, 0.555, "B", fontsize=14, fontweight="bold")

FIGS.mkdir(exist_ok=True)
stem = FIGS / "fig_reliability_fingerprint"
fig.savefig(stem.with_suffix(".pdf"))
fig.savefig(stem.with_suffix(".svg"))
fig.savefig(stem.with_suffix(".png"), dpi=300)
print("wrote", stem.with_suffix(".pdf"))

# greyscale check
from PIL import Image
img = Image.open(stem.with_suffix(".png")).convert("L")
img.save(FIGS / "fig_reliability_fingerprint_grey.png")
print("wrote greyscale check")
