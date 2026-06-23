#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from src import plots as paper_plots  # noqa: E402


BLUE = paper_plots.PAPER_BLUE
TEAL = paper_plots.PAPER_TEAL
ORANGE = paper_plots.PAPER_ORANGE
RUST = paper_plots.PAPER_RUST
GRAY = paper_plots.PAPER_GRAY
LIGHT_GRAY = "#eef0f2"
BORDER = "#d8dee3"
INK = "#263238"
MUTED = "#66747a"
TEAL_FILL = "#e8f4f1"
ORANGE_FILL = "#fff4df"
BLUE_FILL = "#e8f0f7"
GRAY_FILL = "#f7f8f9"
LANCET_BLUE = paper_plots.LANCET_BLUE
LANCET_RED = paper_plots.LANCET_RED
LANCET_PURPLE = paper_plots.LANCET_PURPLE
LANCET_GRAY = paper_plots.LANCET_GRAY
LANCET_BLACK = paper_plots.LANCET_BLACK
LANCET_BLUE_FILL = "#E8F1FA"
LANCET_RED_FILL = "#FDEAEA"
LANCET_PURPLE_FILL = "#F1EAF3"

MODEL_TITLES = {
    "qwen25vl": "Qwen2.5-VL-7B",
    "internvl": "InternVL2.5-8B",
    "llavaov": "LLaVA-OneVision-7B",
    "smolvlm": "SmolVLM-2.2B",
    "medgemma": "MedGemma-4B",
    "huatuo": "HuatuoGPT-Vision-7B",
}
DATASET_TITLES = {
    "vqa_rad": "VQA-RAD",
    "slake_en": "SLAKE",
    "pathvqa": "PathVQA",
}


def set_style() -> None:
    paper_plots._set_style()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    *,
    face: str = "white",
    edge: str = BORDER,
    lw: float = 1.0,
    radius: float = 0.015,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#9aa4aa") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def label(ax: plt.Axes, x: float, y: float, text: str, *, color: str = MUTED, size: int = 7) -> None:
    ax.text(x, y, text.upper(), ha="center", va="center", color=color, fontsize=size, weight="bold")


def draw_stage_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    face: str,
    edge: str,
    title_color: str = INK,
) -> None:
    rounded_box(ax, (x, y), (w, h), face=face, edge=edge)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=8, weight="bold", color=title_color)
    ax.text(
        x + w / 2,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=7,
        color=INK,
        linespacing=1.25,
    )


def draw_small_reliability(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, (x, y), (w, h), face="white", edge=BORDER, radius=0.01)
    pad = 0.028
    x0, x1 = x + pad, x + w - pad
    y0, y1 = y + pad, y + h - pad
    ax.plot([x0, x1], [y0, y1], color=LIGHT_GRAY, lw=1.0, ls="--")
    ax.plot(
        [x0, x0 + 0.36 * w, x0 + 0.70 * w, x1],
        [y0, y0 + 0.22 * h, y0 + 0.42 * h, y1 - 0.01],
        color=BLUE,
        lw=1.7,
    )
    ax.text(x + w / 2, y - 0.018, "Calibration (ECE)", ha="center", va="top", fontsize=6.2, color=INK)


def draw_small_selective(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, (x, y), (w, h), face="white", edge=BORDER, radius=0.01)
    pad = 0.028
    xs = [x + pad, x + 0.35 * w, x + 0.68 * w, x + w - pad]
    ys = [y + h - pad, y + 0.57 * h, y + 0.32 * h, y + pad]
    ax.plot(xs, ys, color=TEAL, lw=1.7)
    ax.text(x + w / 2, y - 0.018, "Selective prediction (AURC)", ha="center", va="top", fontsize=6.2, color=INK)


def make_framework(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for x, title in [
        (0.095, "Input"),
        (0.285, "Models"),
        (0.505, "Adaptation"),
        (0.695, "Confidence"),
        (0.895, "Evaluation"),
    ]:
        label(ax, x, 0.91, title)

    rounded_box(ax, (0.04, 0.25), (0.115, 0.44), face=GRAY_FILL, edge=BORDER)
    ax.add_patch(Rectangle((0.061, 0.38), 0.073, 0.19, facecolor="#2f3a40", edgecolor="#2f3a40"))
    ax.add_patch(Circle((0.082, 0.475), 0.028, facecolor="#f4f6f7", edgecolor="none", alpha=0.92))
    ax.add_patch(Circle((0.111, 0.475), 0.028, facecolor="#f4f6f7", edgecolor="none", alpha=0.92))
    ax.add_patch(FancyBboxPatch((0.124, 0.61), 0.027, 0.04, boxstyle="round,pad=0.006", facecolor="white", edgecolor=BORDER))
    ax.text(0.137, 0.63, "?", ha="center", va="center", fontsize=8, weight="bold", color=GRAY)
    ax.text(0.097, 0.315, "VQA-RAD\nSLAKE\nPathVQA", ha="center", va="center", fontsize=7.5, color=INK, linespacing=1.15)
    ax.text(0.097, 0.215, "radiology + pathology", ha="center", va="top", fontsize=6.5, color=MUTED)

    rounded_box(ax, (0.205, 0.23), (0.155, 0.50), face=GRAY_FILL, edge=BORDER)
    ax.text(0.225, 0.67, "General", ha="left", va="center", fontsize=7.2, weight="bold", color=INK)
    y = 0.612
    for model in ["qwen25vl", "internvl", "llavaov", "smolvlm"]:
        color = paper_plots.MODEL_COLORS[model]
        rounded_box(ax, (0.237, y), (0.095, 0.038), face="white", edge=BORDER, radius=0.006)
        ax.add_patch(Rectangle((0.237, y), 0.006, 0.038, facecolor=color, edgecolor=color))
        ax.text(0.285, y + 0.019, paper_plots.MODEL_DISPLAY[model], ha="center", va="center", fontsize=5.8, color=INK)
        y -= 0.052
    ax.text(0.225, y - 0.004, "Medical", ha="left", va="center", fontsize=7.2, weight="bold", color=INK)
    y -= 0.056
    for model in ["medgemma", "huatuo"]:
        color = paper_plots.MODEL_COLORS[model]
        rounded_box(ax, (0.237, y), (0.095, 0.038), face=TEAL_FILL, edge="#b7ded6", radius=0.006)
        ax.add_patch(Rectangle((0.237, y), 0.006, 0.038, facecolor=color, edgecolor=color))
        ax.text(0.285, y + 0.019, paper_plots.MODEL_DISPLAY[model], ha="center", va="center", fontsize=5.8, color=INK)
        y -= 0.052

    draw_stage_card(ax, 0.435, 0.49, 0.135, 0.11, "Zero-shot", "no adapter", face="white", edge=BORDER)
    draw_stage_card(ax, 0.435, 0.32, 0.135, 0.11, "Fine-tuned LoRA", "small trainable\nlanguage-side ranks", face=TEAL_FILL, edge="#b7ded6")
    ax.add_patch(Circle((0.582, 0.375), 0.02, facecolor="white", edgecolor=BORDER))
    ax.text(0.582, 0.375, "T", ha="center", va="center", fontsize=8, weight="bold", color=TEAL)

    rounded_box(ax, (0.64, 0.32), (0.13, 0.32), face=ORANGE_FILL, edge="#efd49a")
    ax.text(0.705, 0.585, "Per-answer probability", ha="center", va="center", fontsize=6.7, color=INK)
    for i, (lab, val, color) in enumerate([("yes", 0.76, ORANGE), ("no", 0.34, LIGHT_GRAY), ("maybe", 0.22, LIGHT_GRAY)]):
        yy = 0.525 - i * 0.075
        ax.add_patch(Rectangle((0.662, yy), 0.078 * val, 0.025, facecolor=color, edgecolor="none"))
        ax.text(0.748, yy + 0.012, lab, ha="right", va="center", fontsize=5.8, color=MUTED)
    ax.text(0.705, 0.355, "confidence = max probability", ha="center", va="center", fontsize=6.3, color=MUTED)

    draw_small_reliability(ax, 0.84, 0.48, 0.105, 0.15)
    draw_small_selective(ax, 0.84, 0.26, 0.105, 0.15)

    for start, end in [
        ((0.155, 0.48), (0.205, 0.48)),
        ((0.36, 0.48), (0.435, 0.545)),
        ((0.36, 0.48), (0.435, 0.375)),
        ((0.57, 0.545), (0.64, 0.52)),
        ((0.57, 0.375), (0.64, 0.45)),
        ((0.77, 0.48), (0.84, 0.555)),
        ((0.77, 0.48), (0.84, 0.335)),
    ]:
        arrow(ax, start, end)

    ax.plot([0.04, 0.96], [0.16, 0.16], color=LIGHT_GRAY, lw=0.9)
    ax.text(
        0.5,
        0.105,
        "Measured under: temperature scaling - image corruption - cross-dataset shift - per-modality shift",
        ha="center",
        va="center",
        fontsize=6.5,
        color=MUTED,
    )
    save(fig, path)


def make_lora_supp(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(ax, (0.035, 0.56), (0.93, 0.35), face="white", edge=BORDER)
    ax.text(0.055, 0.87, "(a) What LoRA is", ha="left", va="center", fontsize=9, weight="bold", color=INK)
    ax.add_patch(Rectangle((0.16, 0.66), 0.085, 0.12, facecolor=GRAY_FILL, edgecolor=BORDER))
    ax.text(0.202, 0.72, "W", ha="center", va="center", fontsize=18, weight="bold", color=GRAY)
    ax.text(0.202, 0.62, "pretrained\nfrozen", ha="center", va="top", fontsize=7, color=MUTED)
    ax.add_patch(Circle((0.202, 0.807), 0.018, facecolor="white", edgecolor=BORDER))
    ax.text(0.202, 0.807, "lock", ha="center", va="center", fontsize=4.8, color=GRAY)
    ax.text(0.31, 0.72, "+", ha="center", va="center", fontsize=22, color=GRAY)
    ax.add_patch(Rectangle((0.37, 0.61), 0.045, 0.20, facecolor=TEAL_FILL, edgecolor="#b7ded6"))
    ax.add_patch(Rectangle((0.415, 0.61), 0.095, 0.055, facecolor=TEAL_FILL, edgecolor="#b7ded6"))
    ax.text(0.392, 0.715, "A", ha="center", va="center", fontsize=14, weight="bold", color=TEAL)
    ax.text(0.462, 0.638, "B", ha="center", va="center", fontsize=14, weight="bold", color=TEAL)
    ax.text(0.44, 0.59, "low rank", ha="center", va="top", fontsize=7, color=MUTED)
    arrow(ax, (0.085, 0.72), (0.16, 0.72), GRAY)
    arrow(ax, (0.51, 0.72), (0.58, 0.72), GRAY)
    ax.text(0.075, 0.72, "x", ha="center", va="center", fontsize=10, color=INK)
    ax.text(0.605, 0.72, "h", ha="center", va="center", fontsize=10, color=INK)
    ax.text(0.31, 0.585, "h = Wx + (alpha / r)BAx", ha="center", va="center", fontsize=8, color=INK)

    bullets = [
        ("Pretrained weights stay frozen", GRAY),
        ("Two small low-rank matrices A, B are trainable", TEAL),
        ("A starts random and B starts at zero, so initial output is unchanged", TEAL),
        ("Because rank r is far smaller than the layer width, only a small fraction of parameters is updated", ORANGE),
    ]
    for i, (text, color) in enumerate(bullets):
        yy = 0.81 - i * 0.052
        ax.add_patch(Rectangle((0.675, yy - 0.014), 0.012, 0.012, facecolor=color, edgecolor=color))
        ax.text(0.695, yy - 0.008, text, ha="left", va="center", fontsize=7.3, color=INK)
    ax.text(0.675, 0.585, "At inference, BA can be merged into W; no extra image model is trained.", ha="left", va="center", fontsize=7, color=MUTED)

    rounded_box(ax, (0.035, 0.19), (0.93, 0.30), face="white", edge=BORDER)
    ax.text(0.055, 0.45, "(b) How we apply LoRA in this study", ha="left", va="center", fontsize=9, weight="bold", color=INK)
    stages = [
        (0.08, "Image + question", GRAY_FILL, BORDER),
        (0.23, "Vision\nencoder\nfrozen", GRAY_FILL, BORDER),
        (0.37, "Projector", "white", BORDER),
        (0.52, "Language model\nwith transformer blocks", BLUE_FILL, "#c9ddeb"),
    ]
    for x, text, face, edge in stages:
        rounded_box(ax, (x, 0.285), (0.11, 0.10), face=face, edge=edge)
        ax.text(x + 0.055, 0.335, text, ha="center", va="center", fontsize=7, color=INK, linespacing=1.15)
    for s, e in [((0.19, 0.335), (0.23, 0.335)), ((0.34, 0.335), (0.37, 0.335)), ((0.48, 0.335), (0.52, 0.335))]:
        arrow(ax, s, e)

    rounded_box(ax, (0.68, 0.25), (0.25, 0.17), face=GRAY_FILL, edge=BORDER)
    ax.text(0.70, 0.385, "Language-side LoRA target modules", ha="left", va="center", fontsize=7, weight="bold", color=INK)
    xs = [0.71, 0.755, 0.80, 0.845]
    for i, name in enumerate(["q", "k", "v", "o"]):
        ax.add_patch(Circle((xs[i], 0.335), 0.018, facecolor=TEAL_FILL, edgecolor="#b7ded6"))
        ax.text(xs[i], 0.335, name, ha="center", va="center", fontsize=7, color=TEAL, weight="bold")
    ax.text(0.78, 0.295, "self-attention projections", ha="center", va="center", fontsize=6.5, color=MUTED)
    for i, name in enumerate(["gate", "up", "down"]):
        xx = 0.705 + i * 0.07
        ax.add_patch(
            FancyBboxPatch(
                (xx, 0.255),
                0.047,
                0.03,
                boxstyle="round,pad=0.005",
                facecolor=ORANGE_FILL,
                edgecolor="#efd49a",
            )
        )
        ax.text(xx + 0.0235, 0.27, name, ha="center", va="center", fontsize=6, color=ORANGE, weight="bold")
    ax.text(0.81, 0.238, "feed-forward projections", ha="center", va="center", fontsize=6.5, color=MUTED)

    ax.text(
        0.68,
        0.22,
        "A trainable adapter is inserted on 7 linear projection families; everything else is frozen.",
        ha="left",
        va="center",
        fontsize=7,
        color=INK,
    )

    rounded_box(ax, (0.035, 0.055), (0.93, 0.075), face="#f7f6f2", edge="#e1ded6")
    ax.text(0.055, 0.093, "Configuration", ha="left", va="center", fontsize=7.5, weight="bold", color=INK)
    ax.text(
        0.16,
        0.093,
        "rank=16  alpha=32  dropout=0.05  learning rate=1e-4  cosine warmup=0.03  3 epochs  effective batch=16  bfloat16  seed=42",
        ha="left",
        va="center",
        fontsize=7,
        color=INK,
    )
    save(fig, path)


def pct(value: str | float) -> str:
    return f"{float(value) * 100:.0f}%"


def image_on_uniform_canvas(img: Image.Image, size: tuple[int, int] = (1000, 600)) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    fitted = ImageOps.contain(img, size, method=resample)
    canvas = Image.new("RGB", size, (10, 12, 14))
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def shorten_question(question: str, width: int = 58) -> str:
    lines = textwrap.wrap(question, width=width)
    if len(lines) <= 2:
        return "\n".join(lines)
    return "\n".join([lines[0], textwrap.shorten(" ".join(lines[1:]), width=width, placeholder="...")])


def sentence_initial_cap(text: str) -> str:
    text = str(text).strip()
    return text[:1].upper() + text[1:] if text else text


def make_failure_cases(repo: Path, path: Path) -> None:
    selected = [
        ("A", "case_008_qwen25vl_pathvqa"),
        ("B", "case_012_internvl_vqa_rad"),
        ("C", "case_019_llavaov_vqa_rad"),
        ("D", "case_038_medgemma_vqa_rad"),
        ("E", "case_050_huatuo_slake_en"),
        ("F", "case_036_smolvlm_pathvqa"),
    ]
    with (repo / "case" / "cases.csv").open(newline="", encoding="utf-8") as f:
        rows = {row["case_id"]: row for row in csv.DictReader(f)}

    fig = plt.figure(figsize=(7.3, 5.7))
    fig.patch.set_facecolor("white")
    fig.patches.append(
        Rectangle(
            (0.022, 0.04),
            0.956,
            0.91,
            transform=fig.transFigure,
            facecolor="#f3f8ef",
            edgecolor="#3f4547",
            linewidth=0.9,
            zorder=-10,
        )
    )
    panel_w = 0.275
    image_h = 0.235
    info_h = 0.145
    x_gap = 0.035
    y_positions = [0.500, 0.055]
    x_positions = [0.055, 0.055 + panel_w + x_gap, 0.055 + 2 * (panel_w + x_gap)]
    title_font = 8.3
    subtitle_font = 7.5
    question_font = 8.4
    result_font = 8.7
    zero_shot_font = 8.4

    for idx, (letter, case_id) in enumerate(selected):
        row = rows[case_id]
        col = idx % 3
        panel_row = idx // 3
        x0 = x_positions[col]
        y0 = y_positions[panel_row]
        model = row["model"]
        model_color = LANCET_PURPLE if model in {"medgemma", "huatuo"} else LANCET_BLUE
        compact_title = {
            "llavaov": "LLaVA-OV-7B",
            "huatuo": "HuatuoGPT-7B",
        }.get(model, MODEL_TITLES[model])
        title = f"({letter}) {compact_title}"
        subtitle = DATASET_TITLES[row["dataset"]]

        fig.text(
            x0,
            y0 + image_h + info_h + 0.034,
            title,
            ha="left",
            va="bottom",
            fontsize=title_font,
            weight="bold",
            color=LANCET_BLACK,
        )
        fig.text(
            x0 + panel_w,
            y0 + image_h + info_h + 0.034,
            subtitle,
            ha="right",
            va="bottom",
            fontsize=subtitle_font,
            color=MUTED,
        )
        fig.patches.append(
            Rectangle(
                (x0, y0 + image_h + info_h + 0.014),
                panel_w,
                0.006,
                transform=fig.transFigure,
                facecolor=model_color,
                edgecolor="none",
            )
        )

        ax_img = fig.add_axes([x0, y0 + info_h + 0.006, panel_w, image_h])
        img = image_on_uniform_canvas(Image.open(repo / "case" / row["image_file"]))
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")
            spine.set_linewidth(0.75)
        ax_img.set_facecolor("#0a0c0e")

        ax_info = fig.add_axes([x0, y0, panel_w, info_h])
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        question = shorten_question(sentence_initial_cap(row["question"]), width=31)
        ax_info.text(
            0.0,
            0.97,
            f"Q: {question}",
            ha="left",
            va="top",
            fontsize=question_font,
            color=LANCET_BLACK,
            linespacing=1.0,
        )
        ax_info.text(
            0.0,
            0.49,
            f"Ground truth: {sentence_initial_cap(row['gold'])}",
            ha="left",
            va="center",
            fontsize=result_font,
            color=LANCET_RED,
            weight="bold",
        )
        ax_info.text(
            0.0,
            0.30,
            f"Fine-tuned: {sentence_initial_cap(row['ft_pred'])} (wrong, {pct(row['ft_confidence'])})",
            ha="left",
            va="center",
            fontsize=result_font,
            color=LANCET_RED,
            weight="bold",
        )
        zs_state = "correct" if row["zero_shot_correct"] == "1" else "wrong"
        zs_color = LANCET_BLUE if zs_state == "correct" else LANCET_RED
        ax_info.text(
            0.0,
            0.10,
            f"Zero-shot: {sentence_initial_cap(row['zero_shot_pred'])} ({zs_state}, {pct(row['zero_shot_conf'])})",
            ha="left",
            va="center",
            fontsize=zero_shot_font,
            color=zs_color,
        )

    save(fig, path)


def make_metric_figures(repo: Path, figures_dir: Path, paper_images: Path) -> None:
    results = repo / "results"
    paper_plots._set_style()
    paper_plots._fig_reliability_diagram(str(results), str(figures_dir / "fig2_reliability_diagram.pdf"))

    master = pd.read_csv(results / "master_metrics.csv")
    corruption = pd.read_csv(results / "rq4_corruption_metrics.csv")
    rq5 = pd.read_csv(results / "rq5_modality_metrics.csv")
    paper_plots._fig_reliability_breakdown(
        master,
        corruption,
        rq5,
        str(figures_dir / "fig5_reliability_breakdown.pdf"),
    )
    for name in ["fig2_reliability_diagram.pdf", "fig5_reliability_breakdown.pdf"]:
        shutil.copy2(figures_dir / name, paper_images / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render styled manuscript figures without touching figure 1 or graphical abstracts.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--paper-dir", default=None)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    paper = Path(args.paper_dir).resolve() if args.paper_dir else repo.parent / "paper_overleaf"
    paper_images = paper / "images"
    figures_dir = repo / "figures"
    supp_figures = paper / "supplementary" / "figures"

    set_style()
    make_metric_figures(repo, figures_dir, paper_images)
    make_framework(paper_images / "framework.pdf")
    make_failure_cases(repo, paper_images / "fig_failure_cases.pdf")
    make_lora_supp(paper_images / "lora_supp.pdf")
    supp_figures.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paper_images / "lora_supp.pdf", supp_figures / "lora_supp.pdf")


if __name__ == "__main__":
    main()
