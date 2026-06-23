from PIL import Image

from src.finetune import resize_image_for_training, shuffled_train_rows


def test_resize_image_for_training_keeps_small_image() -> None:
    image = Image.new("RGB", (320, 240))

    resized, changed = resize_image_for_training(image, max_edge=768)

    assert resized.size == (320, 240)
    assert changed is False


def test_resize_image_for_training_limits_longest_edge() -> None:
    image = Image.new("RGB", (1600, 800))

    resized, changed = resize_image_for_training(image, max_edge=800)

    assert resized.size == (800, 400)
    assert changed is True


def test_shuffled_train_rows_is_deterministic_without_mutating_input() -> None:
    rows = [{"uid": str(idx), "gold_idx": "0" if idx < 5 else "1"} for idx in range(10)]

    first = shuffled_train_rows(rows, seed=42, epoch=0)
    second = shuffled_train_rows(rows, seed=42, epoch=0)
    next_epoch = shuffled_train_rows(rows, seed=42, epoch=1)

    assert [row["uid"] for row in first] == [row["uid"] for row in second]
    assert [row["uid"] for row in first] != [row["uid"] for row in rows]
    assert [row["uid"] for row in next_epoch] != [row["uid"] for row in first]
    assert [row["uid"] for row in rows] == [str(idx) for idx in range(10)]
