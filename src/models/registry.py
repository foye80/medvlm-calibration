from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from typing import Any, Sequence

from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    short_name: str
    hf_id: str
    family: str
    size: str
    model_type: str
    dtype: str = "bf16"
    gated: bool = False
    fallback_hf_id: str | None = None


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    spec: ModelSpec
    device: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen25vl": ModelSpec("qwen25vl", "Qwen/Qwen2.5-VL-7B-Instruct", "qwen_vl", "7B", "general"),
    "internvl": ModelSpec("internvl", "OpenGVLab/InternVL2_5-8B", "internvl", "8B", "general"),
    "llavaov": ModelSpec(
        "llavaov",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "llava_ov",
        "7B",
        "general",
    ),
    "smolvlm": ModelSpec("smolvlm", "HuggingFaceTB/SmolVLM-Instruct", "smolvlm", "2.2B", "general_small"),
    "medgemma": ModelSpec(
        "medgemma",
        "google/medgemma-4b-it",
        "gemma",
        "4B",
        "medical",
        gated=True,
    ),
    "huatuo": ModelSpec(
        "huatuo",
        "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL",
        "qwen_vl",
        "7B",
        "medical",
    ),
}


def list_model_specs() -> list[ModelSpec]:
    return list(MODEL_SPECS.values())


def get_model_spec(short_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[short_name]
    except KeyError as exc:
        raise KeyError(f"unknown model short_name: {short_name}") from exc


def resolve_lora_target_modules(spec: ModelSpec) -> list[str]:
    if spec.family == "smolvlm":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if spec.family == "internvl":
        return ["wqkv", "wo", "w1", "w2", "w3"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def torch_dtype_from_spec(spec: ModelSpec) -> Any:
    import torch

    if spec.dtype == "bf16" and torch.cuda.is_available():
        return torch.bfloat16
    if spec.dtype == "fp16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def fallback_spec_for(spec: ModelSpec) -> ModelSpec | None:
    if spec.fallback_hf_id is None:
        return None
    family = "qwen_vl" if "qwen2.5vl" in spec.fallback_hf_id.lower() else "llava_like"
    return replace(
        spec,
        hf_id=spec.fallback_hf_id,
        family=family,
        model_type=f"{spec.model_type}_fallback",
        gated=False,
    )


def _looks_like_access_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = ["401", "403", "gated", "restricted", "unauthorized", "forbidden", "access"]
    return any(marker in message for marker in markers)


def model_class_for_spec(spec: ModelSpec) -> Any:
    import transformers

    if spec.family == "qwen_vl" and hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        return transformers.Qwen2_5_VLForConditionalGeneration
    if spec.family == "internvl" and hasattr(transformers, "AutoModel"):
        return transformers.AutoModel
    for class_name in ["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"]:
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            return model_class
    raise RuntimeError("no compatible Transformers model class found")


def _load_model_and_processor_from_spec(
    spec: ModelSpec,
    *,
    load_in_4bit: bool = False,
) -> tuple[Any, Any, ModelSpec]:
    import torch
    from transformers import AutoProcessor

    model_class = model_class_for_spec(spec)
    dtype = torch_dtype_from_spec(spec)
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    logger.info("loading processor model=%s hf_id=%s", spec.short_name, spec.hf_id)
    processor = AutoProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)
    logger.info("loading model model=%s dtype=%s load_in_4bit=%s", spec.short_name, dtype, load_in_4bit)
    model = model_class.from_pretrained(
        spec.hf_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor, spec


def load_model_and_processor(
    short_name: str,
    *,
    smoke: bool = False,
    load_in_4bit: bool = False,
) -> tuple[Any, Any, ModelSpec]:
    spec = get_model_spec(short_name)
    if smoke:
        return None, None, spec
    try:
        return _load_model_and_processor_from_spec(spec, load_in_4bit=load_in_4bit)
    except Exception as exc:
        fallback_spec = fallback_spec_for(spec)
        if fallback_spec is None or not _looks_like_access_error(exc):
            raise
        logger.warning(
            "model=%s hf_id=%s unavailable because of access error; using fallback_hf_id=%s",
            spec.short_name,
            spec.hf_id,
            fallback_spec.hf_id,
        )
        return _load_model_and_processor_from_spec(fallback_spec, load_in_4bit=load_in_4bit)


def load_model_bundle(short_name: str, *, smoke: bool = False, load_in_4bit: bool = False) -> LoadedModel:
    model, processor, spec = load_model_and_processor(short_name, smoke=smoke, load_in_4bit=load_in_4bit)
    device = "cuda" if not smoke else "smoke"
    if model is not None:
        try:
            device = str(next(model.parameters()).device)
        except StopIteration:
            device = "unknown"
    return LoadedModel(model=model, processor=processor, spec=spec, device=device)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or load model registry entries.")
    parser.add_argument("--model", default="qwen25vl")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "model registry")
    bundle = load_model_bundle(args.model, smoke=config.smoke)
    logger.info(
        "model=%s hf_id=%s family=%s gated=%s device=%s lora_targets=%s",
        bundle.spec.short_name,
        bundle.spec.hf_id,
        bundle.spec.family,
        bundle.spec.gated,
        bundle.device,
        ",".join(resolve_lora_target_modules(bundle.spec)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
