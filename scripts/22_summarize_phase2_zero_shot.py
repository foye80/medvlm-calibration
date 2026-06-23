#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUTS = [
    ("qwen25vl", "Qwen/Qwen2.5-VL-7B-Instruct", "results/phase2_qwen25vl_vqarad50.csv", ""),
    ("smolvlm", "HuggingFaceTB/SmolVLM-Instruct", "results/phase2_smolvlm_vqarad50.csv", ""),
    (
        "llavaov",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "results/phase2_llavaov_vqarad50.csv",
        "",
    ),
    ("internvl", "OpenGVLab/InternVL2_5-8B", "results/phase2_internvl_vqarad50.csv", ""),
    (
        "huatuo",
        "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL",
        "results/phase2_huatuo_vqarad50.csv",
        "",
    ),
    (
        "medgemma",
        "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL",
        "results/phase2_medgemma_fallback_vqarad50.csv",
        "google/medgemma-4b-it inaccessible; used fallback",
    ),
]


def summarize_file(model: str, hf_id: str, csv_path: Path, note: str) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no prediction rows found: {csv_path}")
    correct = sum(int(row["correct"]) for row in rows)
    warnings = sum(1 for row in rows if row.get("image_load_warning"))
    mean_confidence = sum(float(row["confidence"]) for row in rows) / len(rows)
    return {
        "model": model,
        "hf_id": hf_id,
        "prediction_csv": str(csv_path),
        "n": str(len(rows)),
        "accuracy": f"{correct / len(rows):.4f}",
        "correct": str(correct),
        "mean_confidence": f"{mean_confidence:.4f}",
        "image_load_warnings": str(warnings),
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase 2 VQA-RAD zero-shot smoke outputs.")
    parser.add_argument("--output-csv", default="results/phase2_vqarad50_summary.csv")
    args = parser.parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize_file(model, hf_id, Path(csv_path), note)
        for model, hf_id, csv_path, note in DEFAULT_INPUTS
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['model']}: n={row['n']} accuracy={row['accuracy']} "
            f"mean_confidence={row['mean_confidence']} warnings={row['image_load_warnings']}"
        )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
