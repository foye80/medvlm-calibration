#!/usr/bin/env python3
"""Compute high-confidence error concentration, r_conf(alpha).

This script reads clean Phase 4 prediction CSVs and computes:
  r_conf(alpha) = P(confidence >= alpha | prediction is wrong)

It also pairs each fine-tuned cell against the matching zero-shot baseline
for the same model and evaluation dataset:
  delta_r_conf = r_conf_zero_shot - r_conf_fine_tuned

The script does not run model inference and does not need a GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["qwen25vl", "internvl", "llavaov", "smolvlm", "medgemma", "huatuo"]
DATASETS = ["vqa_rad", "slake_en", "pathvqa"]
ALPHAS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00]
SEED = 42
N_BOOTSTRAP = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def prediction_path(results_dir: Path, model: str, condition: str, eval_dataset: str) -> Path:
    return results_dir / f"phase4_{model}_{condition}_on_{eval_dataset}_test.csv"


def load_prediction(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["uid", "correct", "confidence"])


def r_conf_for_alpha(df: pd.DataFrame, alpha: float) -> dict[str, float | int]:
    correct = df["correct"].astype(int).to_numpy()
    confidence = df["confidence"].astype(float).to_numpy()
    wrong = correct == 0
    high_conf_wrong = wrong & (confidence >= alpha)
    n_error = int(wrong.sum())
    n_high_conf_error = int(high_conf_wrong.sum())
    r_conf = float(n_high_conf_error / n_error) if n_error else float("nan")
    return {
        "n_total": int(len(df)),
        "n_error": n_error,
        "n_high_conf_error": n_high_conf_error,
        "r_conf": r_conf,
        "accuracy": float(correct.mean()) if len(correct) else float("nan"),
        "mean_confidence": float(confidence.mean()) if len(confidence) else float("nan"),
    }


def compute_cell_values(results_dir: Path, models: list[str], datasets: list[str]) -> pd.DataFrame:
    rows = []
    missing = []
    for model in models:
        for eval_dataset in datasets:
            conditions = [("zero_shot", "none", "zero_shot")]
            conditions.extend(
                (
                    train_dataset,
                    train_dataset,
                    "ft_in_dataset" if train_dataset == eval_dataset else "ft_cross_dataset",
                )
                for train_dataset in datasets
            )
            for condition, train_dataset, setting in conditions:
                path = prediction_path(results_dir, model, condition, eval_dataset)
                if not path.exists():
                    missing.append(path)
                    continue
                df = load_prediction(path)
                for alpha in ALPHAS:
                    metric = r_conf_for_alpha(df, alpha)
                    rows.append(
                        {
                            "model": model,
                            "train_dataset": train_dataset,
                            "eval_dataset": eval_dataset,
                            "condition": condition,
                            "setting": setting,
                            "alpha": alpha,
                            **metric,
                            "source_file": str(path),
                        }
                    )
    if missing:
        missing_text = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing prediction files:\n{missing_text}")
    return pd.DataFrame(rows)


def attach_master_metrics(cell_df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    master_path = results_dir / "master_metrics.csv"
    if not master_path.exists():
        return cell_df
    master = pd.read_csv(master_path)
    keep = [
        "model",
        "condition",
        "eval_dataset",
        "ece",
        "adaptive_ece",
        "mce",
        "brier",
        "nll",
        "aurc",
        "e_aurc",
    ]
    present = [col for col in keep if col in master.columns]
    metric_cols = [col for col in present if col not in {"model", "condition", "eval_dataset"}]
    renamed = master[present].rename(columns={col: f"master_{col}" for col in metric_cols})
    return cell_df.merge(renamed, on=["model", "condition", "eval_dataset"], how="left")


def compute_delta_values(cell_df: pd.DataFrame) -> pd.DataFrame:
    baseline = cell_df[cell_df["setting"] == "zero_shot"][
        ["model", "eval_dataset", "alpha", "r_conf", "n_error", "n_high_conf_error"]
    ].rename(
        columns={
            "r_conf": "zero_shot_r_conf",
            "n_error": "zero_shot_n_error",
            "n_high_conf_error": "zero_shot_n_high_conf_error",
        }
    )
    ft = cell_df[cell_df["setting"].isin(["ft_in_dataset", "ft_cross_dataset"])].copy()
    out = ft.merge(baseline, on=["model", "eval_dataset", "alpha"], how="left")
    out = out.rename(
        columns={
            "r_conf": "ft_r_conf",
            "n_error": "ft_n_error",
            "n_high_conf_error": "ft_n_high_conf_error",
        }
    )
    out["delta_r_conf"] = out["zero_shot_r_conf"] - out["ft_r_conf"]
    out["expected_high_conf_errors_at_zero_shot_rate"] = out["zero_shot_r_conf"] * out["ft_n_error"]
    out["case_delta_high_conf_errors"] = out["delta_r_conf"] * out["ft_n_error"]
    out["excess_high_conf_errors_after_ft"] = -out["case_delta_high_conf_errors"]
    return out


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        draw = rng.choice(values, size=len(values), replace=True)
        samples[idx] = draw.mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def sign_flip_p_value(values: np.ndarray, seed: int) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    n = len(values)
    if n <= 20:
        signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * n))).T.reshape(-1, n)
        permuted = np.abs((signs * values).mean(axis=1))
        return float((np.sum(permuted >= observed) + 1) / (len(permuted) + 1))

    rng = np.random.default_rng(seed)
    draws = 200000
    signs = rng.choice([-1.0, 1.0], size=(draws, n), replace=True)
    permuted = np.abs((signs * values).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (draws + 1))


def summarize_r_conf(cell_df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for (setting, alpha), sub in cell_df.groupby(["setting", "alpha"], sort=True):
        values = sub["r_conf"].to_numpy(dtype=float)
        ci_lo, ci_hi = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=seed)
        total_errors = int(sub["n_error"].sum())
        total_high_conf_errors = int(sub["n_high_conf_error"].sum())
        rows.append(
            {
                "setting": setting,
                "alpha": alpha,
                "n_cells": int(len(sub)),
                "mean_r_conf": float(np.nanmean(values)),
                "median_r_conf": float(np.nanmedian(values)),
                "mean_r_conf_ci_low": ci_lo,
                "mean_r_conf_ci_high": ci_hi,
                "pooled_n_error": total_errors,
                "pooled_n_high_conf_error": total_high_conf_errors,
                "pooled_r_conf": float(total_high_conf_errors / total_errors) if total_errors else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_deltas(delta_df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for (setting, alpha), sub in delta_df.groupby(["setting", "alpha"], sort=True):
        values = sub["delta_r_conf"].to_numpy(dtype=float)
        ci_lo, ci_hi = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=seed)
        n_negative = int(np.sum(values < 0))
        n_positive = int(np.sum(values > 0))
        rows.append(
            {
                "setting": setting,
                "alpha": alpha,
                "n_paired_cells": int(len(sub)),
                "mean_delta_r_conf": float(np.nanmean(values)),
                "median_delta_r_conf": float(np.nanmedian(values)),
                "mean_delta_ci_low": ci_lo,
                "mean_delta_ci_high": ci_hi,
                "sign_flip_p_value": sign_flip_p_value(values, seed=seed),
                "n_delta_lt_0": n_negative,
                "frac_delta_lt_0": float(n_negative / len(sub)) if len(sub) else float("nan"),
                "n_delta_gt_0": n_positive,
                "frac_delta_gt_0": float(n_positive / len(sub)) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_cross_eval_mean(delta_df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """Summarize cross-dataset FT after averaging the two off-diagonal trains.

    Cross-dataset cells share the same zero-shot baseline for a fixed
    model/evaluation dataset. This collapsed summary uses one paired unit per
    model/evaluation dataset/alpha to avoid treating the shared baseline as
    fully independent.
    """
    collapsed = (
        delta_df[delta_df["setting"] == "ft_cross_dataset"]
        .groupby(["model", "eval_dataset", "alpha"], as_index=False)
        .agg(delta_r_conf=("delta_r_conf", "mean"))
    )
    collapsed["setting"] = "ft_cross_dataset_eval_mean"
    rows = []
    for (setting, alpha), sub in collapsed.groupby(["setting", "alpha"], sort=True):
        values = sub["delta_r_conf"].to_numpy(dtype=float)
        ci_lo, ci_hi = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=seed)
        n_negative = int(np.sum(values < 0))
        n_positive = int(np.sum(values > 0))
        rows.append(
            {
                "setting": setting,
                "alpha": alpha,
                "n_paired_cells": int(len(sub)),
                "mean_delta_r_conf": float(np.nanmean(values)),
                "median_delta_r_conf": float(np.nanmedian(values)),
                "mean_delta_ci_low": ci_lo,
                "mean_delta_ci_high": ci_hi,
                "sign_flip_p_value": sign_flip_p_value(values, seed=seed),
                "n_delta_lt_0": n_negative,
                "frac_delta_lt_0": float(n_negative / len(sub)) if len(sub) else float("nan"),
                "n_delta_gt_0": n_positive,
                "frac_delta_gt_0": float(n_positive / len(sub)) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_case_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, alpha), sub in delta_df.groupby(["setting", "alpha"], sort=True):
        ft_errors = int(sub["ft_n_error"].sum())
        ft_high_conf_errors = int(sub["ft_n_high_conf_error"].sum())
        expected_at_zs_rate = float(sub["expected_high_conf_errors_at_zero_shot_rate"].sum())
        case_delta = float(sub["case_delta_high_conf_errors"].sum())
        excess = float(sub["excess_high_conf_errors_after_ft"].sum())
        rows.append(
            {
                "setting": setting,
                "alpha": alpha,
                "n_cells": int(len(sub)),
                "ft_n_error_total": ft_errors,
                "ft_n_high_conf_error_total": ft_high_conf_errors,
                "expected_high_conf_errors_at_zero_shot_rate": expected_at_zs_rate,
                "case_delta_high_conf_errors": case_delta,
                "excess_high_conf_errors_after_ft": excess,
                "case_delta_per_100_errors": float(case_delta / ft_errors * 100.0) if ft_errors else float("nan"),
                "excess_per_100_errors": float(excess / ft_errors * 100.0) if ft_errors else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def make_paired_wide(cell_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    zero = cell_df[cell_df["setting"] == "zero_shot"][
        ["model", "eval_dataset", "alpha", "r_conf", "n_error", "n_high_conf_error"]
    ].rename(
        columns={
            "r_conf": "zero_shot_r_conf",
            "n_error": "zero_shot_n_error",
            "n_high_conf_error": "zero_shot_n_high_conf_error",
        }
    )
    in_ft = delta_df[delta_df["setting"] == "ft_in_dataset"][
        ["model", "eval_dataset", "alpha", "ft_r_conf", "delta_r_conf", "ft_n_error", "ft_n_high_conf_error"]
    ].rename(
        columns={
            "ft_r_conf": "in_dataset_ft_r_conf",
            "delta_r_conf": "in_dataset_delta_r_conf",
            "ft_n_error": "in_dataset_ft_n_error",
            "ft_n_high_conf_error": "in_dataset_ft_n_high_conf_error",
        }
    )
    cross = (
        delta_df[delta_df["setting"] == "ft_cross_dataset"]
        .groupby(["model", "eval_dataset", "alpha"], as_index=False)
        .agg(
            cross_dataset_ft_r_conf_mean=("ft_r_conf", "mean"),
            cross_dataset_delta_r_conf_mean=("delta_r_conf", "mean"),
            cross_dataset_ft_n_error_sum=("ft_n_error", "sum"),
            cross_dataset_ft_n_high_conf_error_sum=("ft_n_high_conf_error", "sum"),
            cross_dataset_n_train_conditions=("train_dataset", "count"),
        )
    )
    return zero.merge(in_ft, on=["model", "eval_dataset", "alpha"], how="left").merge(
        cross, on=["model", "eval_dataset", "alpha"], how="left"
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    cell_df = compute_cell_values(results_dir, models=args.models, datasets=args.datasets)
    cell_df = attach_master_metrics(cell_df, results_dir)
    delta_df = compute_delta_values(cell_df)
    summary_df = summarize_r_conf(cell_df, n_bootstrap=args.bootstrap, seed=args.seed)
    delta_summary_df = summarize_deltas(delta_df, n_bootstrap=args.bootstrap, seed=args.seed)
    cross_eval_mean_summary_df = summarize_cross_eval_mean(delta_df, n_bootstrap=args.bootstrap, seed=args.seed)
    case_delta_summary_df = summarize_case_deltas(delta_df)
    wide_df = make_paired_wide(cell_df, delta_df)

    outputs = {
        "r_conf_cell_values.csv": cell_df,
        "r_conf_delta_values.csv": delta_df,
        "r_conf_paired_wide.csv": wide_df,
        "r_conf_summary_by_alpha.csv": summary_df,
        "r_conf_delta_summary_by_alpha.csv": delta_summary_df,
        "r_conf_delta_summary_cross_eval_mean_by_alpha.csv": cross_eval_mean_summary_df,
        "r_conf_case_delta_summary_by_alpha.csv": case_delta_summary_df,
    }
    for name, df in outputs.items():
        path = results_dir / name
        df.to_csv(path, index=False)
        print(f"Wrote {path} ({len(df)} rows)")

    alpha90 = delta_summary_df[delta_summary_df["alpha"] == 0.90]
    print("\nPrimary threshold alpha=0.90")
    print(
        alpha90[
            [
                "setting",
                "n_paired_cells",
                "mean_delta_r_conf",
                "mean_delta_ci_low",
                "mean_delta_ci_high",
                "sign_flip_p_value",
                "frac_delta_lt_0",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
