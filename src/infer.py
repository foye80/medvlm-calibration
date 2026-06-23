from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, setup_logging

logger = logging.getLogger(__name__)


def run_inference(
    *,
    model: str,
    dataset: str,
    condition: str,
    records_csv: Path,
    output_csv: Path,
    split: str,
    limit: int | None,
    adapter_path: Path | None = None,
    force: bool = False,
    load_in_4bit: bool = True,
    corruption: str | None = None,
    severity: int = 0,
    smoke: bool = False,
    max_image_edge: int | None = None,
) -> dict[str, int | str | bool | float]:
    if smoke:
        return {"model": model, "dataset": dataset, "condition": condition, "items": 8, "smoke": True}
    return run_prediction_eval(
        model_name=model,
        records_csv=records_csv,
        output_csv=output_csv,
        dataset=dataset,
        split=split,
        limit=limit,
        condition=condition,
        adapter_path=adapter_path,
        force=force,
        load_in_4bit=load_in_4bit,
        corruption=corruption,
        severity=severity,
        max_image_edge=max_image_edge,
    )


def read_items_csv(path: Path, *, dataset: str, split: str, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != dataset or row["split"] != split:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_rgb_image(image_path: str | Path, *, max_image_edge: int | None = None):
    from PIL import Image, ImageFile

    path = Path(image_path)
    try:
        img = Image.open(path).convert("RGB")
        warning = ""
    except OSError as exc:
        if "truncated" not in str(exc).lower():
            raise
        logger.warning("image file is truncated; loading with PIL truncated-image tolerance path=%s", path)
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img = Image.open(path).convert("RGB")
            warning = "truncated"
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
    if max_image_edge is not None:
        w, h = img.size
        if max(w, h) > max_image_edge:
            scale = max_image_edge / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img, warning


def _maybe_load_adapter(model_obj: Any, adapter_path: Path | None) -> Any:
    if adapter_path is None:
        return model_obj
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter_path does not exist: {adapter_path}")
    from peft import PeftModel

    logger.info("loading adapter adapter_path=%s", adapter_path)
    adapted_model = PeftModel.from_pretrained(model_obj, str(adapter_path))
    adapted_model.eval()
    return adapted_model


def _prediction_fieldnames() -> list[str]:
    return [
        "uid",
        "model",
        "resolved_hf_id",
        "model_family",
        "model_type",
        "condition",
        "corruption",
        "severity",
        "adapter_path",
        "dataset",
        "split",
        "gold_idx",
        "pred_idx",
        "correct",
        "confidence",
        "entropy",
        "options",
        "probabilities",
        "normalized_scores",
        "raw_log_likelihoods",
        "token_counts",
        "image_path",
        "image_load_warning",
        "question",
    ]


def _write_prediction_row(
    writer: csv.DictWriter,
    *,
    row: dict[str, str],
    model_name: str,
    resolved_hf_id: str,
    model_family: str,
    model_type: str,
    condition: str,
    corruption: str,
    severity: int,
    adapter_path: Path | None,
    prediction: Any,
    image_load_warning: str,
) -> int:
    from src.confidence import entropy

    options = json.loads(row["options"])
    probs = [score.probability for score in prediction.scores]
    gold_idx = int(row["gold_idx"])
    is_correct = int(prediction.pred_idx == gold_idx)
    writer.writerow(
        {
            "uid": row["uid"],
            "model": model_name,
            "resolved_hf_id": resolved_hf_id,
            "model_family": model_family,
            "model_type": model_type,
            "condition": condition,
            "corruption": corruption,
            "severity": severity,
            "adapter_path": str(adapter_path) if adapter_path is not None else "",
            "dataset": row["dataset"],
            "split": row["split"],
            "gold_idx": gold_idx,
            "pred_idx": prediction.pred_idx,
            "correct": is_correct,
            "confidence": prediction.confidence,
            "entropy": entropy(probs),
            "options": json.dumps(options, ensure_ascii=False),
            "probabilities": json.dumps(probs),
            "normalized_scores": json.dumps([score.normalized_score for score in prediction.scores]),
            "raw_log_likelihoods": json.dumps([score.raw_log_likelihood for score in prediction.scores]),
            "token_counts": json.dumps([score.token_count for score in prediction.scores]),
            "image_path": row["image_path"],
            "image_load_warning": image_load_warning,
            "question": row["question"],
        }
    )
    return is_correct


def run_prediction_eval(
    *,
    model_name: str,
    records_csv: Path,
    output_csv: Path,
    dataset: str,
    split: str,
    limit: int | None,
    condition: str,
    adapter_path: Path | None = None,
    force: bool = False,
    load_in_4bit: bool = True,
    corruption: str | None = None,
    severity: int = 0,
    max_image_edge: int | None = None,
) -> dict[str, float | int | str | bool]:
    from src.models.registry import load_model_bundle
    from src.models.score import score_vqa_options

    if corruption is None or corruption == "clean":
        corruption_name = "clean"
        severity = 0
    else:
        corruption_name = corruption
        if severity not in {1, 2, 3}:
            raise ValueError("severity must be 1, 2, or 3 when corruption is set")

    if limit is not None and limit <= 0:
        limit = None
    if output_csv.exists() and not force:
        logger.info("prediction file exists; skipping output_csv=%s", output_csv)
        return {
            "model": model_name,
            "dataset": dataset,
            "split": split,
            "condition": condition,
            "skipped": True,
            "output_csv": str(output_csv),
        }

    rows = read_items_csv(records_csv, dataset=dataset, split=split, limit=limit)
    if not rows:
        raise ValueError(f"no rows found for dataset={dataset} split={split}")

    bundle = load_model_bundle(model_name, smoke=False, load_in_4bit=load_in_4bit)
    bundle.model = _maybe_load_adapter(bundle.model, adapter_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    correct = 0
    started_at = time.time()
    with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_prediction_fieldnames())
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            options = json.loads(row["options"])
            image, image_load_warning = load_rgb_image(row["image_path"], max_image_edge=max_image_edge)
            if corruption_name != "clean":
                from src.corruptions import apply_corruption

                image = apply_corruption(image, corruption_name, severity)
            prediction = score_vqa_options(
                model=bundle.model,
                processor=bundle.processor,
                image=image,
                question=row["question"],
                options=options,
            )
            is_correct = _write_prediction_row(
                writer,
                row=row,
                model_name=model_name,
                resolved_hf_id=bundle.spec.hf_id,
                model_family=bundle.spec.family,
                model_type=bundle.spec.model_type,
                condition=condition,
                corruption=corruption_name,
                severity=severity,
                adapter_path=adapter_path,
                prediction=prediction,
                image_load_warning=image_load_warning,
            )
            correct += is_correct
            logger.info(
                "infer item=%s/%s uid=%s model=%s condition=%s pred=%s gold=%s conf=%.4f correct=%s",
                idx,
                len(rows),
                row["uid"],
                model_name,
                condition,
                prediction.pred_idx,
                int(row["gold_idx"]),
                prediction.confidence,
                is_correct,
            )
    tmp_csv.replace(output_csv)
    accuracy = correct / len(rows)
    duration_seconds = time.time() - started_at
    summary = {
        "model": model_name,
        "resolved_hf_id": bundle.spec.hf_id,
        "model_family": bundle.spec.family,
        "model_type": bundle.spec.model_type,
        "dataset": dataset,
        "split": split,
        "condition": condition,
        "corruption": corruption_name,
        "severity": severity,
        "adapter_path": str(adapter_path) if adapter_path is not None else "",
        "n": len(rows),
        "accuracy": accuracy,
        "duration_seconds": duration_seconds,
        "output_csv": str(output_csv),
    }
    summary_path = output_csv.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info(
        "infer_summary model=%s dataset=%s split=%s condition=%s n=%s accuracy=%.4f output_csv=%s",
        model_name,
        dataset,
        split,
        condition,
        len(rows),
        accuracy,
        output_csv,
    )
    return summary


def run_zero_shot_eval(
    *,
    model_name: str,
    records_csv: Path,
    output_csv: Path,
    dataset: str,
    split: str,
    limit: int,
) -> dict[str, float | int | str]:
    return run_prediction_eval(
        model_name=model_name,
        records_csv=records_csv,
        output_csv=output_csv,
        dataset=dataset,
        split=split,
        limit=limit,
        condition="zero_shot",
        adapter_path=None,
        force=True,
        load_in_4bit=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run model inference and confidence extraction.")
    parser.add_argument("--model", default="qwen25vl")
    parser.add_argument("--dataset", default="vqa_rad")
    parser.add_argument("--condition", default="zero_shot")
    parser.add_argument("--records-csv", default="data/vqa_items.csv")
    parser.add_argument("--output-csv", default="results/zero_shot_smoke.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--corruption", default="clean")
    parser.add_argument("--severity", type=int, default=0)
    parser.add_argument("--max-image-edge", type=int, default=None)
    parser.add_argument("--zero-shot-eval", action="store_true")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "inference")
    if args.zero_shot_eval and not config.smoke:
        run_zero_shot_eval(
            model_name=args.model,
            records_csv=Path(args.records_csv),
            output_csv=Path(args.output_csv),
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
        )
        return 0
    run_inference(
        model=args.model,
        dataset=args.dataset,
        condition=args.condition,
        records_csv=Path(args.records_csv),
        output_csv=Path(args.output_csv),
        split=args.split,
        limit=args.limit,
        adapter_path=Path(args.adapter_path) if args.adapter_path else None,
        force=bool(args.force),
        load_in_4bit=not bool(args.no_4bit),
        corruption=args.corruption,
        severity=args.severity,
        smoke=config.smoke,
        max_image_edge=args.max_image_edge,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
