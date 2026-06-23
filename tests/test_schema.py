from src.data.schema import VQAItem


def test_vqa_item_round_trip() -> None:
    item = VQAItem(
        uid="vqa_rad_0",
        dataset="vqa_rad",
        modality="xray",
        image_path="data/vqa_rad/images/0.png",
        question="Is there cardiomegaly?",
        options=["yes", "no"],
        gold_idx=1,
        split="train",
    )

    restored = VQAItem.from_dict(item.to_dict())

    assert restored == item
    assert restored.gold_answer == "no"


def test_vqa_item_rejects_bad_gold_index() -> None:
    try:
        VQAItem(
            uid="bad",
            dataset="vqa_rad",
            modality="xray",
            image_path="x.png",
            question="Question?",
            options=["yes", "no"],
            gold_idx=2,
            split="train",
        )
    except ValueError as exc:
        assert "gold_idx" in str(exc)
    else:
        raise AssertionError("expected ValueError")
