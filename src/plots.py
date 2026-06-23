from __future__ import annotations

import argparse
import ast
import json
import logging
import os
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

TRAIN_DATASETS = ["vqa_rad", "slake_en", "pathvqa"]
MODEL_ORDER = ["qwen25vl", "internvl", "llavaov", "smolvlm", "medgemma", "huatuo"]
MODEL_DISPLAY = {
    "qwen25vl": "Qwen2.5-VL",
    "internvl": "InternVL2.5",
    "llavaov": "LLaVA-OV",
    "smolvlm": "SmolVLM",
    "medgemma": "MedGemma",
    "huatuo": "HuatuoGPT",
}
MODEL_COLORS = {
    "qwen25vl": "#0072B2",
    "internvl": "#56B4E9",
    "llavaov": "#D55E00",
    "smolvlm": "#009E73",
    "medgemma": "#CC79A7",
    "huatuo": "#E69F00",
}
PAPER_BLUE = MODEL_COLORS["qwen25vl"]
PAPER_TEAL = MODEL_COLORS["smolvlm"]
PAPER_ORANGE = MODEL_COLORS["huatuo"]
PAPER_RUST = MODEL_COLORS["llavaov"]
PAPER_GRAY = "#6f6f6f"
PAPER_LIGHT_GRAY = "#d8d8d8"
PASTEL_BLUE = "#9ecae1"
PASTEL_BLUE_LIGHT = "#d6ebf7"
PASTEL_BLUE_DARK = "#4f8eb8"
PASTEL_PINK = "#f2a6bd"
PASTEL_PINK_LIGHT = "#fde0ea"
PASTEL_PINK_DARK = "#c96c8d"
REF_BLUE = "#0F80FF"
REF_BLUE_FILL = "#D9ECFF"
REF_RED = "#FC6666"
REF_RED_FILL = "#FFE1E1"
REF12_BLUE = "#7B95C6"
REF12_CYAN = "#49C2D9"
REF12_LIGHT_CYAN = "#A1D8E8"
REF12_GREEN = "#67A583"
REF12_LIGHT_GREEN = "#A2C986"
REF12_PALE_GREEN = "#D0E2C0"
REF12_YELLOW = "#FDED95"
REF12_PEACH = "#FCC1A6"
REF12_ORANGE = "#F59D7E"
LANCET_BLUE = "#00468B"
LANCET_RED = "#ED0000"
LANCET_GREEN = "#42B540"
LANCET_TEAL = "#0099B4"
LANCET_PURPLE = "#925E9F"
LANCET_GRAY = "#ADB6B6"
LANCET_BLACK = "#1B1919"
DATASET_DISPLAY = {
    "vqa_rad": "VQA-RAD",
    "slake_en": "SLAKE-en",
    "pathvqa": "PathVQA",
}
MODALITY_ABBREV = {
    "MR (Mag-netic Resonance Imaging)": "MRI",
    "CT(Computed Tomography)": "CT",
    "ultrasound": "Ultrasound",
    "X-Ray": "X-ray",
    "Dermoscopy": "Dermoscopy",
    "Microscopy Images": "Microscopy",
    "Fundus Photography": "Fundus",
    "OCT (Optical Coherence Tomography": "OCT",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _per_item_path(results_dir: str, model: str, condition: str, eval_dataset: str) -> str:
    return os.path.join(
        results_dir, f"phase4_{model}_{condition}_on_{eval_dataset}_test.csv"
    )


def _parse_list(value: object) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    text = str(value)
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return []
    if not isinstance(parsed, list):
        return []
    return [float(v) for v in parsed]


def _softmax(scores: list[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(scores, dtype=float) / max(float(temperature), 1e-8)
    arr = arr - np.nanmax(arr)
    exp = np.exp(arr)
    return exp / exp.sum()


def _confs_correct_from_scores(df: pd.DataFrame, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    confs: list[float] = []
    correct: list[float] = []
    for scores_raw, gold_raw in zip(df["normalized_scores"], df["gold_idx"], strict=True):
        scores = _parse_list(scores_raw)
        gold = int(gold_raw)
        if not scores or gold < 0 or gold >= len(scores):
            continue
        probs = _softmax(scores, temperature=temperature)
        pred = int(np.argmax(probs))
        confs.append(float(probs[pred]))
        correct.append(float(pred == gold))
    return np.asarray(confs), np.asarray(correct)


def _reliability_bins(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    n = len(confs)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (confs >= lo) & (confs < hi) if b < n_bins - 1 else (confs >= lo) & (confs <= hi)
        if int(mask.sum()) == 0:
            continue
        rows.append(
            {
                "bin": b,
                "mean_conf": float(confs[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "n": int(mask.sum()),
                "weight": float(mask.sum() / n),
            }
        )
    return pd.DataFrame(rows)


def _ece(confs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    ece = 0.0
    n = len(confs)
    if n == 0:
        return float("nan")
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (confs >= lo) & (confs < hi) if b < n_bins - 1 else (confs >= lo) & (confs <= hi)
        if int(mask.sum()) == 0:
            continue
        ece += float(mask.sum() / n) * abs(float(correct[mask].mean()) - float(confs[mask].mean()))
    return float(ece)


def _risk_coverage(confs: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-confs)
    correct_sorted = correct[order]
    ks = np.arange(1, len(confs) + 1)
    coverage = ks / len(confs)
    risk = 1.0 - np.cumsum(correct_sorted) / ks
    return coverage, risk


def _model_list(master: pd.DataFrame) -> list[str]:
    present = set(master["model"].unique())
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def _fig_rq1_ece(master: pd.DataFrame, out_path: str) -> str | None:
    """Paired ECE view using the soft reference palette."""
    models = _model_list(master)
    rows = []
    for model in models:
        zero = master[(master.model == model) & (master.condition == "zero_shot")]
        ft = master[(master.model == model) & (master.condition == master.eval_dataset)]
        if zero.empty or ft.empty:
            continue
        rows.append(
            {
                "model": model,
                "zero_shot": float(zero.groupby("eval_dataset").ece.mean().mean()),
                "fine_tuned": float(ft.ece.mean()),
            }
        )
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(rows))
    for i, row in enumerate(rows):
        improved = row["fine_tuned"] <= row["zero_shot"]
        line_color = REF12_LIGHT_GREEN if improved else REF12_ORANGE
        ax.plot(
            [i, i],
            [row["zero_shot"], row["fine_tuned"]],
            color=line_color,
            lw=2.0,
            alpha=0.86,
            zorder=2,
        )
        ax.scatter(
            i,
            row["zero_shot"],
            s=62,
            marker="o",
            facecolor=REF12_BLUE,
            edgecolor=REF12_BLUE,
            lw=0.5,
            alpha=0.92,
            zorder=3,
        )
        ax.scatter(
            i,
            row["fine_tuned"],
            s=64,
            marker="s",
            facecolor=REF12_PEACH,
            edgecolor=REF12_PEACH,
            lw=0.5,
            alpha=0.94,
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(r["model"], r["model"]) for r in rows], rotation=25, ha="right")
    ax.set_ylabel("Expected calibration error")
    ax.set_ylim(0, max(0.50, max(max(r["zero_shot"], r["fine_tuned"]) for r in rows) + 0.04))
    ax.set_title("Calibration before and after LoRA fine-tuning")
    ax.scatter([], [], s=62, facecolor=REF12_BLUE, edgecolor=REF12_BLUE, lw=0.5, label="Zero-shot")
    ax.scatter([], [], s=64, marker="s", facecolor=REF12_PEACH, edgecolor=REF12_PEACH, lw=0.5, label="Fine-tuned ID")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99,
        0.94,
        "Lower is better",
        ha="right",
        va="top",
        transform=ax.transAxes,
        color="#555555",
    )
    fig.tight_layout()
    return _save(fig, out_path)


def _draw_reliability(
    ax: plt.Axes,
    confs: np.ndarray,
    correct: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str = "o",
) -> None:
    bins = _reliability_bins(confs, correct, n_bins=10)
    if bins.empty:
        return
    sizes = 34 + 220 * bins["weight"].to_numpy()
    ax.plot(bins["mean_conf"], bins["accuracy"], color=color, lw=1.8, alpha=0.9)
    ax.scatter(
        bins["mean_conf"],
        bins["accuracy"],
        s=sizes,
        marker=marker,
        facecolor=color,
        edgecolor=color,
        lw=0.25,
        alpha=0.9,
        label=f"{label} (ECE {_ece(confs, correct):.3f})",
        clip_on=False,
        zorder=3,
    )


def _fig_reliability_diagram(results_dir: str, out_path: str) -> str | None:
    """Main-text reliability diagram for a known overconfident cell."""
    model = "qwen25vl"
    dataset = "vqa_rad"
    zero_path = _per_item_path(results_dir, model, "zero_shot", dataset)
    ft_path = _per_item_path(results_dir, model, dataset, dataset)
    temp_path = os.path.join(results_dir, "temperature_scaling_calib.csv")
    if not (os.path.exists(zero_path) and os.path.exists(ft_path) and os.path.exists(temp_path)):
        logger.warning("missing inputs for reliability diagram")
        return None

    zero = pd.read_csv(zero_path)
    ft = pd.read_csv(ft_path)
    temps = pd.read_csv(temp_path)
    cell = temps[
        (temps["model"] == model)
        & (temps["condition"] == dataset)
        & (temps["eval_dataset"] == dataset)
    ]
    if cell.empty:
        logger.warning("missing temperature for %s/%s", model, dataset)
        return None
    temperature = float(cell.iloc[0]["temperature"])

    z_conf = zero["confidence"].astype(float).to_numpy()
    z_corr = zero["correct"].astype(int).to_numpy(dtype=float)
    ft_conf = ft["confidence"].astype(float).to_numpy()
    ft_corr = ft["correct"].astype(int).to_numpy(dtype=float)
    ts_conf, ts_corr = _confs_correct_from_scores(ft, temperature=temperature)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharex=True, sharey=True)
    for ax in axes:
        ax.plot([0, 1], [0, 1], color=REF12_PALE_GREEN, lw=1.0, ls="--", label="Perfect calibration")
        ax.set_xlim(-0.035, 1.035)
        ax.set_ylim(-0.035, 1.035)
        ax.set_xlabel("Mean confidence")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Empirical accuracy")

    _draw_reliability(
        axes[0],
        z_conf,
        z_corr,
        label="Zero-shot",
        color=REF12_BLUE,
        marker="o",
    )
    _draw_reliability(
        axes[0],
        ft_conf,
        ft_corr,
        label="Fine-tuned",
        color=REF12_ORANGE,
        marker="s",
    )
    axes[0].set_title("A. Fine-tuning can worsen calibration")

    _draw_reliability(
        axes[1],
        ft_conf,
        ft_corr,
        label="Before scaling",
        color=REF12_ORANGE,
        marker="s",
    )
    _draw_reliability(
        axes[1],
        ts_conf,
        ts_corr,
        label="After scaling",
        color=REF12_GREEN,
        marker="^",
    )
    axes[1].set_title("B. Temperature scaling pulls confidence back")
    axes[1].text(
        0.53,
        0.08,
        f"T={temperature:.2f}",
        ha="left",
        va="bottom",
        transform=axes[1].transAxes,
        fontsize=7,
        color="#666666",
    )

    axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    axes[1].legend(frameon=False, loc="upper left", fontsize=7)
    fig.suptitle("Reliability diagram: Qwen2.5-VL on VQA-RAD", y=0.995)
    fig.tight_layout()
    return _save(fig, out_path)


def _fig_risk_coverage_dataset(results_dir: str, eval_dataset: str, out_path: str) -> str | None:
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    drawn = False
    for model in MODEL_ORDER:
        path = _per_item_path(results_dir, model, eval_dataset, eval_dataset)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        coverage, risk = _risk_coverage(
            df["confidence"].to_numpy(float),
            df["correct"].astype(int).to_numpy(float),
        )
        ax.plot(
            coverage,
            risk,
            lw=1.8,
            label=MODEL_DISPLAY.get(model, model),
            color=MODEL_COLORS.get(model, "#666666"),
        )
        drawn = True
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{DATASET_DISPLAY.get(eval_dataset, eval_dataset)}")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    if not drawn:
        plt.close(fig)
        return None
    return _save(fig, out_path)


def _fig_risk_coverage_all(results_dir: str, out_path: str) -> str | None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    drawn = False
    for ax, dataset in zip(axes, TRAIN_DATASETS, strict=True):
        for model in MODEL_ORDER:
            path = _per_item_path(results_dir, model, dataset, dataset)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            coverage, risk = _risk_coverage(
                df["confidence"].to_numpy(float),
                df["correct"].astype(int).to_numpy(float),
            )
            ax.plot(
                coverage,
                risk,
                lw=1.6,
                label=MODEL_DISPLAY.get(model, model),
                color=MODEL_COLORS.get(model, "#666666"),
            )
            drawn = True
        ax.set_title(DATASET_DISPLAY.get(dataset, dataset))
        ax.set_xlabel("Coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Selective risk")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Risk-coverage curves for in-distribution fine-tuned cells", y=1.03)
    fig.tight_layout()
    if not drawn:
        plt.close(fig)
        return None
    return _save(fig, out_path)


def _fig_corruption_supp(corruption: pd.DataFrame, out_path: str) -> str | None:
    if corruption.empty:
        return None
    metrics = [
        ("accuracy", "Accuracy", "Higher is better"),
        ("ece", "Expected calibration error", "Lower is better"),
        ("aurc", "AURC", "Lower is better"),
        ("sel_acc_70", "Selective accuracy at 70% coverage", "Higher is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 6.4), sharex=True)
    for ax, (col, title, note) in zip(axes.ravel(), metrics, strict=True):
        for model in MODEL_ORDER:
            sub = corruption[corruption["model"] == model]
            if sub.empty:
                continue
            ys = [sub[sub["severity"] == sev][col].mean() for sev in [0, 1, 2, 3]]
            ax.plot(
                [0, 1, 2, 3],
                ys,
                marker="o",
                lw=1.7,
                ms=4,
                color=MODEL_COLORS.get(model, "#666666"),
                label=MODEL_DISPLAY.get(model, model),
            )
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1, 2, 3])
        ax.text(0.02, 0.94, note, transform=ax.transAxes, ha="left", va="top", color="#555555", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Corruption severity (0 = clean)")
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Full same-domain corruption curves", y=1.02)
    fig.tight_layout()
    return _save(fig, out_path)


def _modality_summary(rq5: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = rq5[rq5["condition"] == condition].copy()
    sub["modality_short"] = sub["modality"].map(lambda m: MODALITY_ABBREV.get(m, m))
    return (
        sub.groupby("modality_short", as_index=False)
        .agg(accuracy=("accuracy", "mean"), ece=("ece", "mean"), aurc=("aurc", "mean"))
    )


def _fig_reliability_breakdown(
    master: pd.DataFrame,
    corruption: pd.DataFrame,
    rq5: pd.DataFrame,
    out_path: str,
) -> str | None:
    if master.empty or corruption.empty or rq5.empty:
        return None

    id_df = master[(master["condition"] != "zero_shot") & (master["condition"] == master["eval_dataset"])]
    cross_df = master[(master["condition"] != "zero_shot") & (master["condition"] != master["eval_dataset"])]
    sev3 = corruption[corruption["severity"] == 3]
    modal_zero = _modality_summary(rq5, "zero_shot")
    modal_ft = _modality_summary(rq5, "ft_vqa_rad")

    worst_row = modal_zero.sort_values("ece", ascending=False).iloc[0]
    worst_modality = float(worst_row["ece"])
    worst_modality_name = str(worst_row["modality_short"])
    stress = pd.DataFrame(
        [
            {"setting": "FT ID", "ece": float(id_df["ece"].mean())},
            {"setting": "Corruption\nseverity 3", "ece": float(sev3["ece"].mean())},
            {"setting": "Cross-\ndataset", "ece": float(cross_df["ece"].mean())},
            {"setting": f"Worst modality\n({worst_modality_name})", "ece": worst_modality},
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.1))
    ax = axes[0, 0]
    bars = ax.bar(
        np.arange(len(stress)),
        stress["ece"],
        color=[REF12_BLUE, REF12_CYAN, REF12_PEACH, REF12_ORANGE],
        edgecolor=[REF12_BLUE, REF12_CYAN, REF12_PEACH, REF12_ORANGE],
        linewidth=0.8,
        alpha=0.9,
        width=0.72,
    )
    ax.set_xticks(np.arange(len(stress)))
    ax.set_xticklabels(stress["setting"])
    ax.set_ylabel("Mean ECE")
    ax.set_ylim(0, 0.80)
    ax.set_title("A. Where calibration breaks")
    for bar, value in zip(bars, stress["ece"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    metrics = [("accuracy", "Accuracy"), ("ece", "ECE"), ("aurc", "AURC")]
    x = np.arange(len(metrics))
    width = 0.35
    id_vals = [float(id_df[col].mean()) for col, _ in metrics]
    cross_vals = [float(cross_df[col].mean()) for col, _ in metrics]
    ax.bar(
        x - width / 2,
        id_vals,
        width,
        color=REF12_BLUE,
        edgecolor=REF12_BLUE,
        linewidth=0.7,
        alpha=0.9,
        label="Fine-tuned ID",
    )
    ax.bar(
        x + width / 2,
        cross_vals,
        width,
        color=REF12_ORANGE,
        edgecolor=REF12_ORANGE,
        linewidth=0.7,
        alpha=0.9,
        label="Cross-dataset",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("B. Source shift changes all reliability metrics")
    ax.legend(frameon=False, loc="upper right")
    for xpos, value in zip(x - width / 2, id_vals, strict=True):
        ax.text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    for xpos, value in zip(x + width / 2, cross_vals, strict=True):
        ax.text(xpos, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    ax = axes[1, 0]
    corr_summary = corruption.groupby("severity", as_index=False).agg(
        accuracy=("accuracy", "mean"),
        ece=("ece", "mean"),
        aurc=("aurc", "mean"),
    )
    for col, label, color, marker, linestyle in [
        ("accuracy", "Accuracy", REF12_BLUE, "o", "-"),
        ("ece", "ECE", REF12_ORANGE, "s", "-"),
        ("aurc", "AURC", REF12_GREEN, "^", "--"),
    ]:
        ax.plot(
            corr_summary["severity"],
            corr_summary[col],
            marker=marker,
            lw=2.0,
            ls=linestyle,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            alpha=0.95,
            label=label,
        )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Corruption severity (0 = clean)")
    ax.set_ylabel("Mean metric value")
    ax.set_title("C. Pixel corruptions degrade mildly")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    merged = modal_zero[["modality_short", "ece"]].rename(columns={"ece": "zero_shot"}).merge(
        modal_ft[["modality_short", "ece"]].rename(columns={"ece": "ft_vqa_rad"}),
        on="modality_short",
        how="left",
    )
    merged = merged.sort_values("zero_shot", ascending=False)
    x = np.arange(len(merged))
    ax.bar(
        x - 0.19,
        merged["zero_shot"],
        0.38,
        color=REF12_BLUE,
        edgecolor=REF12_BLUE,
        linewidth=0.7,
        alpha=0.9,
        label="Zero-shot",
    )
    ax.bar(
        x + 0.19,
        merged["ft_vqa_rad"],
        0.38,
        color=REF12_LIGHT_GREEN,
        edgecolor=REF12_LIGHT_GREEN,
        linewidth=0.7,
        alpha=0.9,
        label="VQA-RAD adapted",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(merged["modality_short"], rotation=32, ha="right")
    ax.set_ylim(0, 0.80)
    ax.set_ylabel("Mean ECE")
    ax.set_title("D. Modality shift exposes large failures")
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Reliability breaks under source and modality shift, not mild pixel corruption", y=1.01)
    fig.tight_layout()
    return _save(fig, out_path)


def make_figures(
    *,
    results_dir: str = "results",
    figures_dir: str = "figures",
    smoke: bool = False,
) -> list[str]:
    if smoke:
        return []
    _set_style()
    os.makedirs(figures_dir, exist_ok=True)
    produced: list[str] = []

    master_path = os.path.join(results_dir, "master_metrics.csv")
    master = pd.read_csv(master_path) if os.path.exists(master_path) else pd.DataFrame()
    if not master.empty:
        for path in [
            _fig_rq1_ece(master, os.path.join(figures_dir, "fig1_rq1_ece_before_after.pdf")),
            _fig_reliability_diagram(results_dir, os.path.join(figures_dir, "fig2_reliability_diagram.pdf")),
        ]:
            if path:
                produced.append(path)
    else:
        logger.warning("master_metrics.csv not found; skipping main calibration figures")

    for dataset in TRAIN_DATASETS:
        path = _fig_risk_coverage_dataset(
            results_dir,
            dataset,
            os.path.join(figures_dir, f"figS_risk_coverage_{dataset}.pdf"),
        )
        if path:
            produced.append(path)
    path = _fig_risk_coverage_all(results_dir, os.path.join(figures_dir, "figS_risk_coverage_all_datasets.pdf"))
    if path:
        produced.append(path)

    corruption_path = os.path.join(results_dir, "rq4_corruption_metrics.csv")
    rq5_path = os.path.join(results_dir, "rq5_modality_metrics.csv")
    corruption = pd.read_csv(corruption_path) if os.path.exists(corruption_path) else pd.DataFrame()
    rq5 = pd.read_csv(rq5_path) if os.path.exists(rq5_path) else pd.DataFrame()
    path = _fig_corruption_supp(corruption, os.path.join(figures_dir, "figS_corruption_degradation_full.pdf"))
    if path:
        produced.append(path)
    if not master.empty and not corruption.empty and not rq5.empty:
        path = _fig_reliability_breakdown(
            master,
            corruption,
            rq5,
            os.path.join(figures_dir, "fig5_reliability_breakdown.pdf"),
        )
        if path:
            produced.append(path)

    for model in _model_list(master) if not master.empty else []:
        # Keep model-level reliability diagrams as supplemental audit material.
        path = _fig_model_reliability_grid(
            results_dir,
            model,
            os.path.join(figures_dir, f"reliability_{model}.pdf"),
        )
        if path:
            produced.append(path)

    for fig_path in produced:
        logger.info("wrote %s", fig_path)
    return produced


def _fig_model_reliability_grid(results_dir: str, model: str, out_path: str) -> str | None:
    fig, axes = plt.subplots(2, len(TRAIN_DATASETS), figsize=(8.6, 5.3), sharex=True, sharey=True)
    any_drawn = False
    for col, dataset in enumerate(TRAIN_DATASETS):
        for row, (condition, label, color) in enumerate(
            [
                ("zero_shot", "Zero-shot", "#666666"),
                (dataset, "Fine-tuned", MODEL_COLORS.get(model, "#444444")),
            ]
        ):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], "--", color="#777777", lw=0.9)
            path = _per_item_path(results_dir, model, condition, dataset)
            if os.path.exists(path):
                df = pd.read_csv(path)
                confs = df["confidence"].astype(float).to_numpy()
                correct = df["correct"].astype(int).to_numpy(dtype=float)
                _draw_reliability(ax, confs, correct, label=label, color=color)
                any_drawn = True
            ax.set_title(DATASET_DISPLAY.get(dataset, dataset) if row == 0 else "")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel(f"{label}\nAccuracy")
            if row == 1:
                ax.set_xlabel("Confidence")
    fig.suptitle(f"Reliability diagrams: {MODEL_DISPLAY.get(model, model)}", y=1.01)
    fig.tight_layout()
    if not any_drawn:
        plt.close(fig)
        return None
    return _save(fig, out_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render project figures.")
    add_runtime_args(parser)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "figure generation")
    paths = make_figures(
        results_dir=args.results_dir, figures_dir=args.figures_dir, smoke=config.smoke
    )
    logger.info("generated %d figures", len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
