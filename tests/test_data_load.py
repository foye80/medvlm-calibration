from src.data.load import (
    assert_no_split_leakage,
    make_choice_item,
    split_train_calib_test,
)
from src.data.schema import VQAItem


def test_make_choice_item_parses_answer_label() -> None:
    item = make_choice_item(
        uid="pmc_0",
        dataset="pmc_vqa",
        modality="mixed",
        image_path="image.jpg",
        question="Which option is correct?",
        choices=[" A: first ", " B: second ", " C: third ", " D: fourth "],
        answer_label="C",
        split="test",
    )

    assert item is not None
    assert item.options == ["first", "second", "third", "fourth"]
    assert item.gold_idx == 2


def test_question_group_split_has_no_leakage() -> None:
    records = []
    for idx in range(40):
        question = f"Question group {idx // 2}?"
        records.append(
            VQAItem(
                uid=f"item_{idx}",
                dataset="toy",
                modality="xray",
                image_path=f"{idx}.png",
                question=question,
                options=["yes", "no"],
                gold_idx=idx % 2,
                split="train",
            )
        )

    split_records = split_train_calib_test(records, seed=42)

    assert_no_split_leakage(split_records)
    assert {item.split for item in split_records} == {"train", "calib", "test"}
