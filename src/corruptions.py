from __future__ import annotations

import argparse
import io
import logging
from collections.abc import Sequence
from typing import Any

from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, setup_logging

logger = logging.getLogger(__name__)

CORRUPTIONS = (
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "brightness_shift",
    "contrast_shift",
    "jpeg_compression",
    "downscale_upscale",
)


def _horizontal_motion_blur(image: Any, kernel_size: int) -> Any:
    import numpy as np
    from PIL import Image

    arr = np.asarray(image).astype(np.float32)
    pad = kernel_size // 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    blurred = np.zeros_like(arr, dtype=np.float32)
    for offset in range(kernel_size):
        blurred += padded[:, offset : offset + arr.shape[1], :]
    blurred /= float(kernel_size)
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8), mode="RGB")


def apply_corruption(image: Any, corruption: str, severity: int) -> Any:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    if corruption not in CORRUPTIONS:
        raise ValueError(f"unknown corruption: {corruption}")
    if severity not in {1, 2, 3}:
        raise ValueError("severity must be 1, 2, or 3")

    image = image.convert("RGB")
    if corruption == "gaussian_noise":
        std = {1: 8.0, 2: 16.0, 3: 32.0}[severity]
        rng = np.random.default_rng(10_000 + severity)
        arr = np.asarray(image).astype(np.float32)
        noisy = np.clip(arr + rng.normal(0.0, std, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")

    if corruption == "gaussian_blur":
        radius = {1: 1.0, 2: 2.0, 3: 4.0}[severity]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    if corruption == "motion_blur":
        kernel_size = {1: 5, 2: 9, 3: 15}[severity]
        return _horizontal_motion_blur(image, kernel_size)

    if corruption == "brightness_shift":
        factor = {1: 0.75, 2: 0.55, 3: 0.35}[severity]
        return ImageEnhance.Brightness(image).enhance(factor)

    if corruption == "contrast_shift":
        factor = {1: 0.75, 2: 0.55, 3: 0.35}[severity]
        return ImageEnhance.Contrast(image).enhance(factor)

    if corruption == "jpeg_compression":
        quality = {1: 45, 2: 25, 3: 10}[severity]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if corruption == "downscale_upscale":
        scale = {1: 0.75, 2: 0.5, 3: 0.25}[severity]
        width, height = image.size
        small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(small_size, Image.Resampling.BICUBIC).resize(
            (width, height),
            Image.Resampling.BICUBIC,
        )

    raise AssertionError(f"unhandled corruption: {corruption}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply test-time image corruptions.")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "image corruption")
    logger.info("registered_corruptions=%s", ",".join(CORRUPTIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
