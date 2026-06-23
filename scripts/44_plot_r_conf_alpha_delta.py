#!/usr/bin/env python3
"""Plot the r_conf(alpha) sweep and fine-tuning deltas."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SETTING_LABELS = {
    "zero_shot": "Zero-shot",
    "ft_in_dataset": "In-dataset FT",
    "ft_cross_dataset": "Cross-dataset FT",
    "ft_cross_dataset_eval_mean": "Cross-dataset FT",
}

COLORS = {
    "zero_shot": "#4C78A8",
    "ft_in_dataset": "#E15759",
    "ft_cross_dataset": "#F28E2B",
    "ft_cross_dataset_eval_mean": "#F28E2B",
}

MARKERS = {
    "zero_shot": "o",
    "ft_in_dataset": "s",
    "ft_cross_dataset": "^",
    "ft_cross_dataset_eval_mean": "^",
}

ALPHAS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00]
ALPHA_LABELS = ["0.50", "0.60", "0.70", "0.80", "0.90", "0.95", "0.99", "1.00"]
ALPHA_POS = {alpha: idx for idx, alpha in enumerate(ALPHAS)}


def pct(series: pd.Series) -> pd.Series:
    return series * 100.0


def x_positions(alpha_series: pd.Series) -> list[int]:
    return [ALPHA_POS[round(float(alpha), 2)] for alpha in alpha_series]


def plot_r_conf(ax: plt.Axes, summary: pd.DataFrame) -> None:
    order = ["zero_shot", "ft_in_dataset", "ft_cross_dataset"]
    for setting in order:
        sub = summary[summary["setting"] == setting].sort_values("alpha")
        x = x_positions(sub["alpha"])
        y = pct(sub["mean_r_conf"]).to_numpy()
        lo = pct(sub["mean_r_conf_ci_low"]).to_numpy()
        hi = pct(sub["mean_r_conf_ci_high"]).to_numpy()
        ax.plot(
            x,
            y,
            marker=MARKERS[setting],
            markersize=4.5,
            lw=2.0,
            color=COLORS[setting],
            label=SETTING_LABELS[setting],
        )
        ax.fill_between(x, lo, hi, color=COLORS[setting], alpha=0.15, linewidth=0)

    ax.axvline(ALPHA_POS[0.90], color="#777777", lw=0.9, ls=(0, (3, 3)))
    ax.text(ALPHA_POS[0.90] + 0.08, 7, "primary\nα=0.90", ha="left", va="bottom", fontsize=7.2, color="#555555")
    ax.set_title("A. High-confidence error concentration")
    ax.set_xlabel("Confidence threshold α")
    ax.set_ylabel("CER@α (%)")
    ax.set_xlim(-0.15, len(ALPHAS) - 0.85)
    ax.set_ylim(0, 104)
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels(ALPHA_LABELS)
    ax.tick_params(axis="x", rotation=35)
    ax.yaxis.set_major_formatter(lambda x, _pos: f"{x:.0f}%")
    ax.grid(color="#E6E6E6", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=8.0)


def plot_delta(ax: plt.Axes, delta: pd.DataFrame, cross_eval_mean: pd.DataFrame) -> None:
    delta_in = delta[delta["setting"] == "ft_in_dataset"].copy()
    cross = cross_eval_mean.copy()
    plot_specs = [
        ("ft_in_dataset", delta_in, "In-dataset FT"),
        ("ft_cross_dataset_eval_mean", cross, "Cross-dataset FT"),
    ]

    ax.axhline(0, color="#333333", lw=1.0)
    ax.axhspan(-35, 0, color="#F2B8B5", alpha=0.18, linewidth=0)
    ax.text(0.505, -33.0, "delta < 0: FT higher than zero-shot", fontsize=7.2, color="#8B2C2B")

    for setting, sub, label in plot_specs:
        sub = sub.sort_values("alpha")
        x = x_positions(sub["alpha"])
        y = pct(sub["mean_delta_r_conf"]).to_numpy()
        lo = pct(sub["mean_delta_ci_low"]).to_numpy()
        hi = pct(sub["mean_delta_ci_high"]).to_numpy()
        ax.plot(
            x,
            y,
            marker=MARKERS[setting],
            markersize=4.5,
            lw=2.0,
            color=COLORS[setting],
            label=label,
        )
        ax.fill_between(x, lo, hi, color=COLORS[setting], alpha=0.15, linewidth=0)

        alpha90 = sub[sub["alpha"].round(2) == 0.90]
        if not alpha90.empty:
            row = alpha90.iloc[0]
            ax.scatter([ALPHA_POS[0.90]], [row["mean_delta_r_conf"] * 100], s=48, color=COLORS[setting], edgecolor="white", zorder=4)
            dy = -4.5 if setting == "ft_cross_dataset_eval_mean" else 3.5
            ax.text(
                ALPHA_POS[0.90] + 0.08,
                row["mean_delta_r_conf"] * 100 + dy,
                f"{row['mean_delta_r_conf']*100:.1f} pp",
                color=COLORS[setting],
                fontsize=8.2,
                fontweight="bold",
                ha="left",
                va="center",
            )

    ax.set_title("B. Delta relative to matched zero-shot")
    ax.set_xlabel("Confidence threshold α")
    ax.set_ylabel("Delta CER@α (percentage points)")
    ax.set_xlim(-0.15, len(ALPHAS) - 0.85)
    ax.set_ylim(-35, 8)
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels(ALPHA_LABELS)
    ax.tick_params(axis="x", rotation=35)
    ax.yaxis.set_major_formatter(lambda x, _pos: f"{x:.0f} pp")
    ax.grid(color="#E6E6E6", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=8.0)


def main() -> None:
    summary = pd.read_csv(RESULTS / "r_conf_summary_by_alpha.csv")
    delta = pd.read_csv(RESULTS / "r_conf_delta_summary_by_alpha.csv")
    cross_eval_mean = pd.read_csv(RESULTS / "r_conf_delta_summary_cross_eval_mean_by_alpha.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.6,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 8.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.45, 3.55), constrained_layout=True)
    plot_r_conf(axes[0], summary)
    plot_delta(axes[1], delta, cross_eval_mean)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        out = FIGURES / f"r_conf_alpha_delta.{ext}"
        fig.savefig(out, dpi=320)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
