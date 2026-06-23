from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptionScore:
    option: str
    raw_log_likelihood: float
    token_count: int
    normalized_score: float
    probability: float


@dataclass(frozen=True)
class ScoredPrediction:
    pred_idx: int
    confidence: float
    scores: list[OptionScore]


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not values:
        raise ValueError("softmax requires at least one value")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [float(v) / temperature for v in values]
    max_value = max(scaled)
    exp_values = [math.exp(v - max_value) for v in scaled]
    total = sum(exp_values)
    return [v / total for v in exp_values]


def score_options_from_log_likelihoods(
    options: Sequence[str],
    log_likelihoods: Sequence[float],
    token_counts: Sequence[int],
    *,
    length_normalized: bool = True,
    temperature: float = 1.0,
) -> list[OptionScore]:
    if len(options) != len(log_likelihoods) or len(options) != len(token_counts):
        raise ValueError("options, log_likelihoods, and token_counts must have equal length")
    if any(count <= 0 for count in token_counts):
        raise ValueError("token_counts must be positive")
    ranking_scores = [
        ll / count if length_normalized else ll
        for ll, count in zip(log_likelihoods, token_counts, strict=True)
    ]
    probabilities = softmax(ranking_scores, temperature=temperature)
    return [
        OptionScore(
            option=option,
            raw_log_likelihood=float(ll),
            token_count=int(count),
            normalized_score=float(ll) / int(count),
            probability=float(prob),
        )
        for option, ll, count, prob in zip(options, log_likelihoods, token_counts, probabilities, strict=True)
    ]


def build_vqa_prompt(question: str, options: Sequence[str]) -> str:
    if len(options) == 2 and {option.lower() for option in options} == {"yes", "no"}:
        return f"{question.strip()}\nAnswer yes or no."
    option_lines = [f"{chr(ord('A') + idx)}) {option}" for idx, option in enumerate(options)]
    return f"{question.strip()}\nOptions:\n" + "\n".join(option_lines) + "\nAnswer with the best option."


def build_chat_text(processor: Any, question: str, options: Sequence[str], image: Any | None) -> str:
    prompt = build_vqa_prompt(question, options)
    content: list[dict[str, Any]]
    if image is None:
        content = [{"type": "text", "text": prompt}]
    else:
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    if image is None:
        return f"User: {prompt}\nAssistant:"
    return f"User: <image>\n{prompt}\nAssistant:"


def _tokenizer_from_processor(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return processor


def _find_last_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start
    return None


def _processor_inputs(processor: Any, text: str, image: Any | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = [image]
    return processor(**kwargs)


def _is_internvl_model(model: Any) -> bool:
    return all(hasattr(_internvl_base_model(model), attr) for attr in ["extract_feature", "language_model", "num_image_token"])


def _internvl_base_model(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        if base_model is not model and all(
            hasattr(base_model, attr) for attr in ["extract_feature", "language_model", "num_image_token"]
        ):
            return base_model
    base_model = getattr(model, "base_model", None)
    nested_model = getattr(base_model, "model", None)
    if all(hasattr(nested_model, attr) for attr in ["extract_feature", "language_model", "num_image_token"]):
        return nested_model
    if all(hasattr(model, attr) for attr in ["extract_feature", "language_model", "num_image_token"]):
        return model
    return model


def _pil_to_internvl_tensor(image: Any, image_size: int):
    import numpy as np
    import torch
    from PIL import Image

    resized = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(resized).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _internvl_grid(width: int, height: int, max_tiles: int) -> tuple[int, int]:
    aspect = width / height
    candidates = [
        (cols, rows)
        for tiles in range(1, max_tiles + 1)
        for cols in range(1, tiles + 1)
        for rows in range(1, tiles + 1)
        if cols * rows == tiles
    ]
    return min(candidates, key=lambda grid: (abs((grid[0] / grid[1]) - aspect), grid[0] * grid[1]))


def _internvl_pixel_values(image: Any, model: Any):
    import torch
    from PIL import Image

    base_model = _internvl_base_model(model)
    image_size = int(getattr(base_model.config, "force_image_size", None) or base_model.config.vision_config.image_size)
    max_tiles = int(getattr(base_model.config, "max_dynamic_patch", 12) or 12)
    cols, rows = _internvl_grid(image.width, image.height, max_tiles)
    resized = image.resize((cols * image_size, rows * image_size), Image.Resampling.BICUBIC)
    tiles = [
        resized.crop(
            (
                col * image_size,
                row * image_size,
                (col + 1) * image_size,
                (row + 1) * image_size,
            )
        )
        for row in range(rows)
        for col in range(cols)
    ]
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    pixel_values = torch.stack([_pil_to_internvl_tensor(tile, image_size) for tile in tiles])
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return pixel_values.to(device=device, dtype=dtype)


def _internvl_prompt_text(model: Any, tokenizer: Any, question: str, options: Sequence[str], num_patches: int) -> str:
    import copy

    base_model = _internvl_base_model(model)
    prompt = build_vqa_prompt(question, options)
    question_text = prompt if "<image>" in prompt else f"<image>\n{prompt}"
    template = copy.deepcopy(base_model.conv_template)
    template.system_message = base_model.system_message
    template.append_message(template.roles[0], question_text)
    template.append_message(template.roles[1], None)
    text = template.get_prompt()
    image_tokens = "<img>" + "<IMG_CONTEXT>" * base_model.num_image_token * num_patches + "</img>"
    text = text.replace("<image>", image_tokens, 1)
    image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    base_model.img_context_token_id = image_token_id
    if hasattr(model, "img_context_token_id"):
        model.img_context_token_id = image_token_id
    return text


def internvl_option_log_likelihood(
    *,
    model: Any,
    processor: Any,
    image: Any,
    question: str,
    options: Sequence[str],
    option: str,
) -> tuple[float, int]:
    import torch

    tokenizer = _tokenizer_from_processor(processor)
    pixel_values = _internvl_pixel_values(image, model)
    prompt_text = _internvl_prompt_text(model, tokenizer, question, options, int(pixel_values.shape[0]))
    answer_text = f" {option}"
    full_text = prompt_text + answer_text
    model_inputs = tokenizer(full_text, return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    start = _find_last_subsequence(input_ids[0].tolist(), answer_ids)
    if start is None:
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        start = int(prompt_ids.shape[-1])
        answer_ids = input_ids[0, start:].tolist()
    end = start + len(answer_ids)
    if start <= 0 or end > input_ids.shape[-1]:
        raise ValueError("could not locate InternVL answer tokens in model input")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            return_dict=True,
        )
    logits = outputs.logits[0]
    log_probs = torch.log_softmax(logits[start - 1 : end - 1], dim=-1)
    target_ids = input_ids[0, start:end].to(log_probs.device)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return float(token_log_probs.sum().item()), int(target_ids.numel())


def prepare_teacher_forced_inputs(
    *,
    model: Any,
    processor: Any,
    image: Any | None,
    question: str,
    options: Sequence[str],
    answer: str,
) -> dict[str, Any]:
    import torch

    if image is not None and _is_internvl_model(model):
        tokenizer = _tokenizer_from_processor(processor)
        pixel_values = _internvl_pixel_values(image, model)
        prompt_text = _internvl_prompt_text(model, tokenizer, question, options, int(pixel_values.shape[0]))
        answer_text = f" {answer}"
        full_text = prompt_text + answer_text
        model_inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = model_inputs["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
        start = _find_last_subsequence(input_ids[0].tolist(), answer_ids)
        if start is None:
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
            start = int(prompt_ids.shape[-1])
            answer_ids = input_ids[0, start:].tolist()
        end = start + len(answer_ids)
        if start <= 0 or end > input_ids.shape[-1]:
            raise ValueError("could not locate InternVL answer tokens in model input")
        labels = torch.full_like(input_ids, -100)
        labels[0, start:end] = input_ids[0, start:end]
        device = next(model.parameters()).device
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids.to(device),
            "attention_mask": model_inputs["attention_mask"].to(device),
            "image_flags": torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=device),
            "labels": labels.to(device),
        }

    tokenizer = _tokenizer_from_processor(processor)
    prompt_text = build_chat_text(processor, question, options, image)
    answer_text = f" {answer}"
    full_text = prompt_text + answer_text
    inputs = _processor_inputs(processor, full_text, image)
    input_ids = inputs["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    start = _find_last_subsequence(input_ids[0].tolist(), answer_ids)
    if start is None:
        prompt_ids = _processor_inputs(processor, prompt_text, image)["input_ids"][0]
        start = int(prompt_ids.shape[-1])
        answer_ids = input_ids[0, start:].tolist()
    end = start + len(answer_ids)
    if start <= 0 or end > input_ids.shape[-1]:
        raise ValueError("could not locate answer tokens in model input")
    labels = torch.full_like(input_ids, -100)
    labels[0, start:end] = input_ids[0, start:end]
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    inputs["labels"] = labels.to(device)
    return inputs


def option_log_likelihood(
    *,
    model: Any,
    processor: Any,
    image: Any | None,
    question: str,
    options: Sequence[str],
    option: str,
) -> tuple[float, int]:
    import torch

    if image is not None and _is_internvl_model(model):
        return internvl_option_log_likelihood(
            model=model,
            processor=processor,
            image=image,
            question=question,
            options=options,
            option=option,
        )

    tokenizer = _tokenizer_from_processor(processor)
    prompt_text = build_chat_text(processor, question, options, image)
    answer_text = f" {option}"
    full_text = prompt_text + answer_text
    inputs = _processor_inputs(processor, full_text, image)
    input_ids = inputs["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    start = _find_last_subsequence(input_ids[0].tolist(), answer_ids)
    if start is None:
        prompt_ids = _processor_inputs(processor, prompt_text, image)["input_ids"][0]
        start = int(prompt_ids.shape[-1])
        answer_ids = input_ids[0, start:].tolist()
    end = start + len(answer_ids)
    if start <= 0 or end > input_ids.shape[-1]:
        raise ValueError("could not locate answer tokens in model input")

    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0]
    log_probs = torch.log_softmax(logits[start - 1 : end - 1], dim=-1)
    target_ids = input_ids[0, start:end].to(log_probs.device)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return float(token_log_probs.sum().item()), int(target_ids.numel())


def score_vqa_options(
    *,
    model: Any,
    processor: Any,
    image: Any | None,
    question: str,
    options: Sequence[str],
    length_normalized: bool = True,
    temperature: float = 1.0,
) -> ScoredPrediction:
    log_likelihoods: list[float] = []
    token_counts: list[int] = []
    for option in options:
        ll, count = option_log_likelihood(
            model=model,
            processor=processor,
            image=image,
            question=question,
            options=options,
            option=option,
        )
        log_likelihoods.append(ll)
        token_counts.append(count)
    scores = score_options_from_log_likelihoods(
        options,
        log_likelihoods,
        token_counts,
        length_normalized=length_normalized,
        temperature=temperature,
    )
    probabilities = [score.probability for score in scores]
    pred_idx = max(range(len(probabilities)), key=lambda idx: probabilities[idx])
    return ScoredPrediction(pred_idx=pred_idx, confidence=probabilities[pred_idx], scores=scores)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score answer options by teacher-forced likelihood.")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "option scoring")
    if config.smoke:
        scores = score_options_from_log_likelihoods(["yes", "no"], [-0.2, -1.4], [1, 1])
        logger.info("smoke_scores=%s", scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
