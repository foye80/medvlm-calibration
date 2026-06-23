#!/usr/bin/env python3
"""Plot delta_r converted to high-confidence error case counts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COLORS = {
    "ft_in_dataset": "#E15759",
    "ft_cross_dataset": "#F28E2B",
}

LABELS = {
    "ft_in_dataset": "In-dataset FT",
    "ft_cross_dataset": "Cross-dataset FT",
}

MARKERS = {
    "ft_in_dataset": "s",
    "ft_cross_dataset": "^",
}

ALPHAS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
ALPHA_LABELS = ["0.50", "0.60", "0.70", "0.80", "0.90", "0.95", "0.99"]
ALPHA_POS = {alpha: idx for idx, alpha in enumerate(ALPHAS)}


def build_requested_conversion() -> pd.DataFrame:
    """Compute the user's requested conversion: mean delta_r * total FT errors."""
    delta = pd.read_csv(RESULTS / "r_conf_delta_summary_by_alpha.csv")
    case_summary = pd.read_csv(RESULTS / "r_conf_case_delta_summary_by_alpha.csv")
    merged = delta.merge(
        case_summary[["setting", "alpha", "ft_n_error_total", "case_delta_high_conf_errors"]],
        on=["setting", "alpha"],
        how="left",
    )
    merged["mean_delta_times_ft_errors"] = merged["mean_delta_r_conf"] * merged["ft_n_error_total"]
    merged["excess_after_ft_from_mean_delta"] = -merged["mean_delta_times_ft_errors"]
    merged = merged.rename(columns={"case_delta_high_conf_errors": "cell_weighted_case_delta"})
    merged["cell_weighted_excess_after_ft"] = -merged["cell_weighted_case_delta"]
    out = RESULTS / "r_conf_mean_delta_case_conversion_by_alpha.csv"
    merged.to_csv(out, index=False)
    print(f"Wrote {out} ({len(merged)} rows)")
    return merged


def x_positions(alpha_series: pd.Series) -> list[int]:
    return [ALPHA_POS[round(float(alpha), 2)] for alpha in alpha_series]


def plot_series(ax: plt.Axes, data: pd.DataFrame, y_col: str, title: str, ylabel: str) -> None:
    ax.axhline(0, color="#333333", lw=1.0)
    ymin = float(data[y_col].min())
    ymax = float(data[y_col].max())
    if ymin < 0:
        ax.axhspan(ymin * 1.12, 0, color="#F2B8B5", alpha=0.16, linewidth=0)
    for setting in ["ft_in_dataset", "ft_cross_dataset"]:
        sub = data[data["setting"] == setting].sort_values("alpha")
        x = x_positions(sub["alpha"])
        ax.plot(
            x,
            sub[y_col],
            marker=MARKERS[setting],
            markersize=4.8,
            lw=2.0,
            color=COLORS[setting],
            label=LABELS[setting],
        )
        alpha90 = sub[sub["alpha"].round(2) == 0.90]
        if not alpha90.empty:
            row = alpha90.iloc[0]
            dy = 0.05 * max(abs(data[y_col].min()), abs(data[y_col].max()), 1.0)
            offset = dy if setting == "ft_in_dataset" else -dy
            ax.text(
                ALPHA_POS[0.90] + 0.08,
                row[y_col] + offset,
                f"{row[y_col]:+.0f}",
                color=COLORS[setting],
                fontsize=8.2,
                fontweight="bold",
                ha="left",
                va="center",
            )
    ax.axvline(ALPHA_POS[0.90], color="#777777", lw=0.9, ls=(0, (3, 3)))
    ax.set_title(title)
    ax.set_xlabel("Confidence threshold alpha")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.15, len(ALPHAS) - 0.85)
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels(ALPHA_LABELS)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(color="#E6E6E6", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    data = build_requested_conversion()

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
    plot_series(
        axes[0],
        data,
        "mean_delta_times_ft_errors",
        "A. Requested conversion: delta_r * total FT errors",
        "delta_r * total FT errors",
    )
    axes[0].text(
        0.04,
        0.08,
        "negative: more high-confidence errors after FT",
        transform=axes[0].transAxes,
        fontsize=7.3,
        color="#8B2C2B",
        va="bottom",
    )
    plot_series(
        axes[1],
        data,
        "cell_weighted_case_delta",
        "B. Cell-weighted sensitivity",
        "sum(delta_r * FT errors)",
    )
    axes[1].text(
        0.04,
        0.08,
        "computed within each model-dataset cell first",
        transform=axes[1].transAxes,
        fontsize=7.3,
        color="#555555",
        va="bottom",
    )
    axes[0].legend(frameon=False, loc="upper right", fontsize=8.0)

    fig.suptitle("High-confidence error burden implied by delta_r", fontsize=11.1, fontweight="bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        out = FIGURES / f"r_conf_case_delta.{ext}"
        fig.savefig(out, dpi=320)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
