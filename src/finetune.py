from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from src.infer import load_rgb_image
from src.models.registry import load_model_bundle, resolve_lora_target_modules
from src.models.score import prepare_teacher_forced_inputs
from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, set_seed, setup_logging

logger = logging.getLogger(__name__)

VISION_NAME_MARKERS = ("visual", "vision", "vision_tower", "vision_model", "image_encoder")
DEFAULT_MAX_IMAGE_EDGE = 768


def load_lora_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_train_rows(records_csv: Path, *, dataset: str, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with records_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != dataset or row["split"] != "train":
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no train rows found for dataset={dataset}")
    return rows


def shuffled_train_rows(rows: Sequence[dict[str, str]], *, seed: int, epoch: int) -> list[dict[str, str]]:
    epoch_rows = list(rows)
    rng = random.Random(seed + epoch * 1_000_003)
    rng.shuffle(epoch_rows)
    return epoch_rows


def freeze_vision_parameters(model: Any) -> int:
    frozen = 0
    for name, parameter in model.named_parameters():
        lower_name = name.lower()
        if any(marker in lower_name for marker in VISION_NAME_MARKERS):
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def trainable_parameter_count(model: Any) -> tuple[int, int]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def package_versions() -> dict[str, str]:
    import torch
    import transformers
    import peft

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


def resize_image_for_training(image: Any, *, max_edge: int) -> tuple[Any, bool]:
    if max_edge <= 0:
        return image, False
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= max_edge:
        return image, False
    from PIL import Image

    scale = max_edge / longest_edge
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), True


def run_finetune(
    *,
    model: str,
    dataset: str,
    records_csv: Path,
    output_root: Path,
    lora_config_path: Path,
    max_train_items: int | None,
    max_steps: int | None,
    force: bool,
    no_qlora: bool,
    max_image_edge: int,
    smoke: bool = False,
    seed: int = 42,
) -> dict[str, int | str | bool | float]:
    if smoke:
        return {"model": model, "dataset": dataset, "steps": 1, "smoke": True}
    set_seed(seed)
    lora_cfg = load_lora_config(lora_config_path)
    output_dir = output_root / f"{model}_{dataset}_lora"
    adapter_config = output_dir / "adapter_config.json"
    summary_path = output_dir / "training_summary.json"
    if adapter_config.exists() and summary_path.exists() and not force:
        logger.info("adapter already exists; skipping output_dir=%s", output_dir)
        return {"model": model, "dataset": dataset, "skipped": True, "output_dir": str(output_dir)}

    max_items = max_train_items
    rows = read_train_rows(records_csv, dataset=dataset, limit=max_items)
    bits = int(lora_cfg.get("bits", 4))
    load_in_4bit = bits == 4 and not no_qlora
    bundle = load_model_bundle(model, smoke=False, load_in_4bit=load_in_4bit)
    model_obj = bundle.model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

    if load_in_4bit:
        model_obj = prepare_model_for_kbit_training(model_obj)
    target_modules = resolve_lora_target_modules(bundle.spec)
    peft_kwargs: dict[str, Any] = {
        "r": int(lora_cfg["r"]),
        "lora_alpha": int(lora_cfg["lora_alpha"]),
        "lora_dropout": float(lora_cfg["lora_dropout"]),
        "bias": "none",
        "target_modules": target_modules,
    }
    if bundle.spec.family != "internvl":
        peft_kwargs["task_type"] = "CAUSAL_LM"
    peft_config = LoraConfig(**peft_kwargs)
    model_obj = get_peft_model(model_obj, peft_config)
    frozen_vision_params = freeze_vision_parameters(model_obj)
    trainable_params, total_params = trainable_parameter_count(model_obj)
    logger.info(
        "trainable_params=%s total_params=%s frozen_vision_params=%s target_modules=%s",
        trainable_params,
        total_params,
        frozen_vision_params,
        ",".join(target_modules),
    )

    learning_rate = float(lora_cfg["learning_rate"])
    optimizer = torch.optim.AdamW((p for p in model_obj.parameters() if p.requires_grad), lr=learning_rate)
    grad_accum = int(lora_cfg["grad_accum"])
    epochs = int(lora_cfg["num_train_epochs"])
    effective_max_steps = max_steps
    if effective_max_steps is None:
        effective_max_steps = (len(rows) * epochs + grad_accum - 1) // grad_accum

    output_dir.mkdir(parents=True, exist_ok=True)
    model_obj.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    seen_examples = 0
    resized_images = 0
    skipped_oom_uids: list[str] = []
    losses: list[float] = []
    started_at = time.time()
    for epoch in range(epochs):
        epoch_rows = shuffled_train_rows(rows, seed=seed, epoch=epoch)
        logger.info("epoch_start=%s shuffled_train_rows=%s seed=%s", epoch + 1, len(epoch_rows), seed)
        for row in epoch_rows:
            if optimizer_steps >= effective_max_steps:
                break
            options = json.loads(row["options"])
            answer = options[int(row["gold_idx"])]
            image, image_warning = load_rgb_image(row["image_path"])
            if image_warning:
                logger.warning("training item uses image_load_warning=%s uid=%s", image_warning, row["uid"])
            image, resized = resize_image_for_training(image, max_edge=max_image_edge)
            if resized:
                resized_images += 1
                logger.info("resized training image uid=%s max_image_edge=%s", row["uid"], max_image_edge)
            try:
                inputs = prepare_teacher_forced_inputs(
                    model=model_obj,
                    processor=bundle.processor,
                    image=image,
                    question=row["question"],
                    options=options,
                    answer=answer,
                )
                outputs = model_obj(**inputs)
                loss = outputs.loss / grad_accum
                loss.backward()
            except torch.OutOfMemoryError:
                skipped_oom_uids.append(row["uid"])
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning("skipping training item after CUDA OOM uid=%s", row["uid"])
                continue
            losses.append(float(loss.detach().cpu().item() * grad_accum))
            del inputs, outputs, loss
            seen_examples += 1
            if seen_examples % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_obj.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                logger.info(
                    "train_step=%s/%s epoch=%s examples=%s loss=%.4f",
                    optimizer_steps,
                    effective_max_steps,
                    epoch + 1,
                    seen_examples,
                    losses[-1],
                )
        if optimizer_steps >= effective_max_steps:
            break
    if seen_examples % grad_accum != 0 and optimizer_steps < effective_max_steps:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    model_obj.save_pretrained(output_dir)
    try:
        bundle.processor.save_pretrained(output_dir / "processor")
    except Exception as exc:
        logger.warning("could not save processor: %s", exc)
    duration_seconds = time.time() - started_at
    summary = {
        "model": model,
        "resolved_hf_id": bundle.spec.hf_id,
        "dataset": dataset,
        "records_csv": str(records_csv),
        "output_dir": str(output_dir),
        "examples": seen_examples,
        "optimizer_steps": optimizer_steps,
        "epochs_configured": epochs,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "shuffle_each_epoch": True,
        "shuffle_seed": seed,
        "load_in_4bit": load_in_4bit,
        "max_image_edge": max_image_edge,
        "resized_images": resized_images,
        "skipped_oom_count": len(skipped_oom_uids),
        "skipped_oom_uids": skipped_oom_uids[:200],
        "target_modules": target_modules,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "frozen_vision_params": frozen_vision_params,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "last_loss": losses[-1] if losses else None,
        "duration_seconds": duration_seconds,
        "versions": package_versions(),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info(
        "finetune_complete model=%s dataset=%s steps=%s examples=%s output_dir=%s",
        model,
        dataset,
        optimizer_steps,
        seen_examples,
        output_dir,
    )
    return {
        "model": model,
        "dataset": dataset,
        "steps": optimizer_steps,
        "examples": seen_examples,
        "mean_loss": float(summary["mean_loss"] or 0.0),
        "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run QLoRA fine-tuning.")
    parser.add_argument("--model", default="qwen25vl")
    parser.add_argument("--dataset", default="vqa_rad")
    parser.add_argument("--records-csv", default="data/vqa_items.csv")
    parser.add_argument("--output-root", default="adapters")
    parser.add_argument("--lora-config", default="configs/lora.yaml")
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-image-edge", type=int, default=DEFAULT_MAX_IMAGE_EDGE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-qlora", action="store_true")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "fine-tuning")
    run_finetune(
        model=args.model,
        dataset=args.dataset,
        records_csv=Path(args.records_csv),
        output_root=Path(args.output_root),
        lora_config_path=Path(args.lora_config),
        max_train_items=args.max_train_items,
        max_steps=args.max_steps,
        force=bool(args.force),
        no_qlora=bool(args.no_qlora),
        max_image_edge=int(args.max_image_edge),
        smoke=config.smoke,
        seed=config.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
