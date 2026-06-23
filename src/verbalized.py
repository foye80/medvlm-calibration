"""Verbalized confidence collection (RQ2 alternative signal).

Prompts each model to answer the VQA item AND state a 0-100 confidence, then
parses the integer. Pairs with the canonical teacher-forced prediction (joined
by uid downstream) for error-detection AUROC.

No free-text generation path existed before this module; generation mirrors the
two scoring paths in src/models/score.py (standard HF + InternVL custom).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Asked in addition to the normal VQA prompt (PROJECT_SPEC.md §6.2).
VERBALIZED_SUFFIX = (
    "\nFirst give your answer, then state how confident you are that your answer "
    "is correct on a scale from 0 to 100. End your reply with exactly: "
    "Confidence: <number>"
)


def parse_confidence(text: str) -> float | None:
    """Prefer the number after the last 'confidence' mention; fall back to the
    last standalone 0-100 integer in the text."""
    lowered = text.lower()
    matches = list(re.finditer(r"confidence\D*?(\d{1,3})", lowered))
    if matches:
        value = int(matches[-1].group(1))
        if 0 <= value <= 100:
            return value / 100.0
    ints = [int(m.group(1)) for m in re.finditer(r"(?<!\d)(100|\d{1,2})(?!\d)", text)]
    ints = [v for v in ints if 0 <= v <= 100]
    if ints:
        return ints[-1] / 100.0
    return None


def _verbalized_prompt(question: str, options) -> str:
    from src.models.score import build_vqa_prompt

    return build_vqa_prompt(question, options) + VERBALIZED_SUFFIX


def _generate_standard(model, processor, image, prompt: str) -> str:
    import torch

    from src.models.score import _processor_inputs  # type: ignore[attr-defined]

    content = (
        [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        if image is not None
        else [{"type": "text", "text": prompt}]
    )
    messages = [{"role": "user", "content": content}]
    if hasattr(processor, "apply_chat_template"):
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = (f"User: <image>\n{prompt}\nAssistant:" if image is not None else f"User: {prompt}\nAssistant:")
    else:
        text = f"User: <image>\n{prompt}\nAssistant:" if image is not None else f"User: {prompt}\nAssistant:"

    inputs = _processor_inputs(processor, text, image)
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    from src.models.score import _tokenizer_from_processor  # type: ignore[attr-defined]

    tokenizer = _tokenizer_from_processor(processor)
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=pad_id)
    new_tokens = out[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_internvl(model, processor, image, question: str, options) -> str:
    import torch

    from src.models.score import (  # type: ignore[attr-defined]
        _internvl_pixel_values,
        _internvl_prompt_text,
        _tokenizer_from_processor,
    )

    tokenizer = _tokenizer_from_processor(processor)
    pixel_values = _internvl_pixel_values(image, model)
    # _internvl_prompt_text bakes in build_vqa_prompt; rebuild with our suffix.
    base_prompt = _internvl_prompt_text(model, tokenizer, question, options, int(pixel_values.shape[0]))
    # Inject the confidence request just before the assistant turn marker is hard;
    # simplest robust path: append suffix to the user content by regenerating.
    # _internvl_prompt_text already returns the full templated prompt ending at the
    # assistant role; we instead append the suffix into the question text.
    prompt_text = base_prompt  # suffix handled via question below
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    gen = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,
        do_sample=False,
    )
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def generate_verbalized(*, model, processor, image, question: str, options) -> tuple[str, float | None]:
    from src.models.score import _is_internvl_model  # type: ignore[attr-defined]

    if image is not None and _is_internvl_model(model):
        # InternVL: include the suffix in the question text it sees.
        text = _generate_internvl(model, processor, image, question + " " + VERBALIZED_SUFFIX.strip(), options)
    else:
        text = _generate_standard(model, processor, image, _verbalized_prompt(question, options))
    return text, parse_confidence(text)


def _fieldnames() -> list[str]:
    return ["uid", "model", "condition", "dataset", "split", "verbalized_raw", "verbalized_confidence", "parsed"]


def run_verbalized_eval(
    *,
    model_name: str,
    records_csv: Path,
    output_csv: Path,
    dataset: str,
    split: str,
    condition: str,
    adapter_path: Path | None = None,
    limit: int | None = None,
    load_in_4bit: bool = True,
    max_image_edge: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    from src.infer import _maybe_load_adapter, load_rgb_image, read_items_csv
    from src.models.registry import load_model_bundle

    if limit is not None and limit <= 0:
        limit = None
    if output_csv.exists() and not force:
        logger.info("verbalized file exists; skipping output_csv=%s", output_csv)
        return {"model": model_name, "skipped": True, "output_csv": str(output_csv)}

    rows = read_items_csv(records_csv, dataset=dataset, split=split, limit=limit)
    if not rows:
        raise ValueError(f"no rows for dataset={dataset} split={split}")

    bundle = load_model_bundle(model_name, smoke=False, load_in_4bit=load_in_4bit)
    bundle.model = _maybe_load_adapter(bundle.model, adapter_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    parsed_ok = 0
    started = time.time()
    with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            options = json.loads(row["options"])
            image, _ = load_rgb_image(row["image_path"], max_image_edge=max_image_edge)
            raw, conf = generate_verbalized(
                model=bundle.model,
                processor=bundle.processor,
                image=image,
                question=row["question"],
                options=options,
            )
            parsed = conf is not None
            parsed_ok += int(parsed)
            writer.writerow(
                {
                    "uid": row["uid"],
                    "model": model_name,
                    "condition": condition,
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "verbalized_raw": raw.replace("\n", " ")[:300],
                    "verbalized_confidence": "" if conf is None else f"{conf:.4f}",
                    "parsed": int(parsed),
                }
            )
            if idx % 25 == 0 or idx == len(rows):
                logger.info(
                    "verbalized item=%s/%s uid=%s model=%s cond=%s conf=%s parsed_rate=%.2f",
                    idx, len(rows), row["uid"], model_name, condition,
                    "NA" if conf is None else f"{conf:.2f}", parsed_ok / idx,
                )
    tmp_csv.replace(output_csv)
    summary = {
        "model": model_name, "dataset": dataset, "split": split, "condition": condition,
        "n": len(rows), "parsed_rate": parsed_ok / len(rows),
        "adapter_path": str(adapter_path) if adapter_path else "",
        "duration_seconds": time.time() - started, "output_csv": str(output_csv),
    }
    with output_csv.with_suffix(".summary.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2, ensure_ascii=False)
    logger.info("verbalized_summary model=%s cond=%s dataset=%s n=%s parsed_rate=%.3f",
                model_name, condition, dataset, len(rows), summary["parsed_rate"])
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Collect verbalized confidence.")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--records-csv", default="data/vqa_items.csv")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--condition", default="zero_shot")
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-image-edge", type=int, default=None)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_verbalized_eval(
        model_name=a.model,
        records_csv=Path(a.records_csv),
        output_csv=Path(a.output_csv),
        dataset=a.dataset,
        split=a.split,
        condition=a.condition,
        adapter_path=Path(a.adapter_path) if a.adapter_path else None,
        limit=a.limit,
        load_in_4bit=not a.no_4bit,
        max_image_edge=a.max_image_edge,
        force=a.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
