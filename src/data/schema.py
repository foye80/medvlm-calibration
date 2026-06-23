from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class VQAItem:
    uid: str
    dataset: str
    modality: str
    image_path: str
    question: str
    options: list[str]
    gold_idx: int
    split: str

    VALID_SPLITS: ClassVar[set[str]] = {"train", "calib", "test"}

    def __post_init__(self) -> None:
        if not self.uid:
            raise ValueError("uid must be non-empty")
        if not self.dataset:
            raise ValueError("dataset must be non-empty")
        if not self.question:
            raise ValueError("question must be non-empty")
        if not self.options:
            raise ValueError("options must be non-empty")
        if self.gold_idx < 0 or self.gold_idx >= len(self.options):
            raise ValueError("gold_idx must point to one option")
        if self.split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(self.VALID_SPLITS)}")

    @property
    def gold_answer(self) -> str:
        return self.options[self.gold_idx]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "VQAItem":
        return cls(
            uid=str(row["uid"]),
            dataset=str(row["dataset"]),
            modality=str(row["modality"]),
            image_path=str(row["image_path"]),
            question=str(row["question"]),
            options=list(row["options"]),
            gold_idx=int(row["gold_idx"]),
            split=str(row["split"]),
        )
