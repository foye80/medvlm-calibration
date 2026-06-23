import pytest
from PIL import Image

from src.corruptions import CORRUPTIONS, apply_corruption


def test_apply_corruptions_preserve_rgb_size() -> None:
    image = Image.new("RGB", (16, 12), color=(128, 64, 32))
    for corruption in CORRUPTIONS:
        for severity in (1, 2, 3):
            out = apply_corruption(image, corruption, severity=severity)
            assert out.mode == "RGB"
            assert out.size == image.size


def test_apply_corruption_rejects_invalid_inputs() -> None:
    image = Image.new("RGB", (4, 4))
    with pytest.raises(ValueError):
        apply_corruption(image, "not_a_corruption", severity=1)
    with pytest.raises(ValueError):
        apply_corruption(image, "gaussian_noise", severity=0)
