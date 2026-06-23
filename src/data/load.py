from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.schema import VQAItem
from src.runtime import add_runtime_args, config_from_args, log_smoke_stub, setup_logging

logger = logging.getLogger(__name__)

YES_VALUES = {"yes", "y"}
NO_VALUES = {"no", "n"}
TRAIN_DATASETS = {"vqa_rad", "slake_en", "pathvqa"}
TEST_ONLY_DATASETS = {"pmc_vqa", "omnimedvqa"}
DEFAULT_DATASETS = ["vqa_rad", "slake_en", "pathvqa", "pmc_vqa", "omnimedvqa"]


@dataclass(frozen=True)
class DatasetBuildResult:
    dataset: str
    records: list[VQAItem]
    raw_count: int
    kept_count: int
    dropped_count: int
    drop_reasons: dict[str, int]


class ZipImageCache:
    def __init__(self, zip_path: str | Path) -> None:
        self.zip_path = Path(zip_path)
        self.archive = zipfile.ZipFile(self.zip_path)
        self.names = self.archive.namelist()
        self.by_exact = {self._norm_name(name): name for name in self.names}
        self.by_basename: dict[str, list[str]] = defaultdict(list)
        for name in self.names:
            self.by_basename[Path(name).name].append(name)

    @staticmethod
    def _norm_name(name: str) -> str:
        return name.replace("\\", "/").lstrip("./")

    def find_member(self, image_ref: str) -> str | None:
        ref = self._norm_name(image_ref)
        candidates = [
            ref,
            f"Images/{ref}",
            f"OmniMedVQA/{ref}",
            f"OmniMedVQA/Images/{ref}",
            f"imgs/{ref}",
        ]
        for candidate in candidates:
            hit = self.by_exact.get(candidate)
            if hit is not None:
                return hit
        suffix = f"/{ref}"
        for name in self.names:
            if self._norm_name(name).endswith(suffix):
                return name
        basename_hits = self.by_basename.get(Path(ref).name, [])
        if len(basename_hits) == 1:
            return basename_hits[0]
        return None

    def extract(self, image_ref: str, output_path: Path) -> bool:
        member = self.find_member(image_ref)
        if member is None:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.archive.open(member) as src, output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return True


def normalize_text(value: Any) -> str:
    return str(value).strip()


def normalize_answer(value: Any) -> str:
    text = normalize_text(value).lower()
    text = text.strip(" \t\n\r.。:;")
    return text


def is_yes_no_answer(value: Any) -> bool:
    answer = normalize_answer(value)
    return answer in YES_VALUES or answer in NO_VALUES


def yes_no_gold_idx(value: Any) -> int:
    answer = normalize_answer(value)
    if answer in YES_VALUES:
        return 0
    if answer in NO_VALUES:
        return 1
    raise ValueError(f"not a yes/no answer: {value}")


def clean_choice_text(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"^[A-Da-d]\s*[:.)]\s*", "", text)
    return text.strip()


def choice_gold_idx(value: Any) -> int | None:
    text = normalize_text(value).strip()
    if not text:
        return None
    letter = text[0].upper()
    if letter in {"A", "B", "C", "D"}:
        return ord(letter) - ord("A")
    return None


def safe_uid_part(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:120] or "item"


def cache_pil_image(image: Any, output_path: Path) -> bool:
    if output_path.exists():
        return True
    if image is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(image, "mode", None) not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    image.save(output_path)
    return True


def make_yes_no_item(
    *,
    uid: str,
    dataset: str,
    modality: str,
    image_path: str,
    question: Any,
    answer: Any,
    split: str,
) -> VQAItem | None:
    if not is_yes_no_answer(answer):
        return None
    return VQAItem(
        uid=uid,
        dataset=dataset,
        modality=modality,
        image_path=image_path,
        question=normalize_text(question),
        options=["yes", "no"],
        gold_idx=yes_no_gold_idx(answer),
        split=split,
    )


def make_choice_item(
    *,
    uid: str,
    dataset: str,
    modality: str,
    image_path: str,
    question: Any,
    choices: Sequence[Any],
    answer_label: Any,
    split: str,
) -> VQAItem | None:
    if len(choices) != 4:
        return None
    options = [clean_choice_text(choice) for choice in choices]
    if any(not option for option in options):
        return None
    gold_idx = choice_gold_idx(answer_label)
    if gold_idx is None:
        return None
    return VQAItem(
        uid=uid,
        dataset=dataset,
        modality=modality,
        image_path=image_path,
        question=normalize_text(question),
        options=options,
        gold_idx=gold_idx,
        split=split,
    )


def split_train_calib_test(records: Sequence[VQAItem], *, seed: int) -> list[VQAItem]:
    if not records:
        return []
    from sklearn.model_selection import train_test_split

    grouped: dict[str, list[VQAItem]] = defaultdict(list)
    for item in records:
        group_key = f"{item.dataset}:{normalize_text(item.question).lower()}"
        grouped[group_key].append(item)

    group_keys = list(grouped)
    group_labels = []
    for key in group_keys:
        labels = [item.gold_idx for item in grouped[key]]
        group_labels.append(Counter(labels).most_common(1)[0][0])

    if len(group_keys) < 3:
        return [
            VQAItem(
                uid=item.uid,
                dataset=item.dataset,
                modality=item.modality,
                image_path=item.image_path,
                question=item.question,
                options=item.options,
                gold_idx=item.gold_idx,
                split="train",
            )
            for item in records
        ]

    def can_stratify(labels: Sequence[int]) -> bool:
        counts = Counter(labels)
        return len(counts) > 1 and min(counts.values()) >= 2

    stratify = group_labels if can_stratify(group_labels) else None
    train_keys, tmp_keys, train_labels, tmp_labels = train_test_split(
        group_keys,
        group_labels,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
    tmp_stratify = tmp_labels if can_stratify(tmp_labels) else None
    calib_keys, test_keys = train_test_split(
        tmp_keys,
        test_size=0.5,
        random_state=seed,
        stratify=tmp_stratify,
    )

    split_by_key = {key: "train" for key in train_keys}
    split_by_key.update({key: "calib" for key in calib_keys})
    split_by_key.update({key: "test" for key in test_keys})

    output: list[VQAItem] = []
    for key, items in grouped.items():
        split = split_by_key[key]
        for item in items:
            output.append(
                VQAItem(
                    uid=item.uid,
                    dataset=item.dataset,
                    modality=item.modality,
                    image_path=item.image_path,
                    question=item.question,
                    options=item.options,
                    gold_idx=item.gold_idx,
                    split=split,
                )
            )
    return output


def _load_hf_yes_no_dataset(
    *,
    dataset_key: str,
    hf_id: str,
    modality: str,
    output_dir: Path,
    cache_images: bool,
    smoke: bool,
    max_items: int | None,
) -> DatasetBuildResult:
    from datasets import get_dataset_split_names, load_dataset

    raw_count = 0
    kept: list[VQAItem] = []
    drop_reasons: Counter[str] = Counter()
    split_names = get_dataset_split_names(hf_id, trust_remote_code=True)
    for native_split in split_names:
        ds = load_dataset(hf_id, split=native_split, trust_remote_code=True)
        for idx, row in enumerate(ds):
            if max_items is not None and raw_count >= max_items:
                break
            raw_count += 1
            answer = row.get("answer")
            if not is_yes_no_answer(answer):
                drop_reasons["not_yes_no"] += 1
                continue
            uid = f"{dataset_key}_{native_split}_{idx}"
            image_path = ""
            if cache_images and row.get("image") is not None:
                local_path = output_dir / dataset_key / "images" / f"{uid}.png"
                if cache_pil_image(row.get("image"), local_path):
                    image_path = str(local_path)
            if not image_path:
                image_path = f"missing_image/{dataset_key}/{uid}.png"
            item = make_yes_no_item(
                uid=uid,
                dataset=dataset_key,
                modality=modality,
                image_path=image_path,
                question=row.get("question"),
                answer=answer,
                split="train",
            )
            if item is None:
                drop_reasons["invalid_yes_no"] += 1
                continue
            kept.append(item)
        if max_items is not None and raw_count >= max_items:
            break
    if dataset_key in TRAIN_DATASETS:
        kept = split_train_calib_test(kept, seed=42)
    return DatasetBuildResult(
        dataset=dataset_key,
        records=kept,
        raw_count=raw_count,
        kept_count=len(kept),
        dropped_count=raw_count - len(kept),
        drop_reasons=dict(drop_reasons),
    )


def _load_slake(
    *,
    output_dir: Path,
    cache_images: bool,
    smoke: bool,
    max_items: int | None,
) -> DatasetBuildResult:
    from datasets import get_dataset_split_names, load_dataset
    from huggingface_hub import hf_hub_download

    raw_count = 0
    kept: list[VQAItem] = []
    drop_reasons: Counter[str] = Counter()
    image_cache: ZipImageCache | None = None
    if cache_images and not smoke:
        image_cache = ZipImageCache(hf_hub_download("BoKelvin/SLAKE", "imgs.zip", repo_type="dataset"))

    for native_split in get_dataset_split_names("BoKelvin/SLAKE", trust_remote_code=True):
        ds = load_dataset("BoKelvin/SLAKE", split=native_split, trust_remote_code=True)
        for idx, row in enumerate(ds):
            if max_items is not None and raw_count >= max_items:
                break
            raw_count += 1
            if normalize_text(row.get("q_lang", "en")).lower() != "en":
                drop_reasons["non_english"] += 1
                continue
            if not is_yes_no_answer(row.get("answer")):
                drop_reasons["not_yes_no"] += 1
                continue
            uid = f"slake_en_{native_split}_{idx}"
            image_ref = normalize_text(row.get("img_name", ""))
            local_path = output_dir / "slake_en" / "images" / safe_uid_part(image_ref)
            image_path = str(local_path)
            if image_cache is not None and not local_path.exists():
                if not image_cache.extract(image_ref, local_path):
                    drop_reasons["missing_image"] += 1
                    continue
            item = make_yes_no_item(
                uid=uid,
                dataset="slake_en",
                modality=normalize_text(row.get("modality", "radiology")).lower() or "radiology",
                image_path=image_path,
                question=row.get("question"),
                answer=row.get("answer"),
                split="train",
            )
            if item is None:
                drop_reasons["invalid_yes_no"] += 1
                continue
            kept.append(item)
        if max_items is not None and raw_count >= max_items:
            break

    kept = split_train_calib_test(kept, seed=42)
    return DatasetBuildResult(
        dataset="slake_en",
        records=kept,
        raw_count=raw_count,
        kept_count=len(kept),
        dropped_count=raw_count - len(kept),
        drop_reasons=dict(drop_reasons),
    )


def _load_pmc_vqa(
    *,
    output_dir: Path,
    cache_images: bool,
    smoke: bool,
    max_items: int | None,
) -> DatasetBuildResult:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    csv_files = ["test_clean.csv", "test.csv", "test_2.csv"]
    image_caches: list[ZipImageCache] = []
    if cache_images and not smoke:
        for filename in ["images.zip", "images_2.zip"]:
            image_caches.append(ZipImageCache(hf_hub_download("xmcmic/PMC-VQA", filename, repo_type="dataset")))

    raw_count = 0
    kept: list[VQAItem] = []
    drop_reasons: Counter[str] = Counter()
    seen_uid: set[str] = set()

    for csv_file in csv_files:
        csv_path = hf_hub_download("xmcmic/PMC-VQA", csv_file, repo_type="dataset")
        frame = pd.read_csv(csv_path)
        for idx, row in frame.iterrows():
            if max_items is not None and raw_count >= max_items:
                break
            raw_count += 1
            answer_label = row.get("Answer_label", row.get("Answer"))
            choices = [row.get(f"Choice {letter}") for letter in ["A", "B", "C", "D"]]
            figure_path = normalize_text(row.get("Figure_path", ""))
            if not figure_path:
                drop_reasons["missing_figure_path"] += 1
                continue
            uid = f"pmc_vqa_{safe_uid_part(csv_file)}_{idx}_{safe_uid_part(figure_path)}"
            if uid in seen_uid:
                drop_reasons["duplicate_uid"] += 1
                continue
            seen_uid.add(uid)
            local_path = output_dir / "pmc_vqa" / "images" / figure_path
            if image_caches and not local_path.exists():
                if not any(cache.extract(figure_path, local_path) for cache in image_caches):
                    drop_reasons["missing_image"] += 1
                    continue
            item = make_choice_item(
                uid=uid,
                dataset="pmc_vqa",
                modality="mixed",
                image_path=str(local_path),
                question=row.get("Question"),
                choices=choices,
                answer_label=answer_label,
                split="test",
            )
            if item is None:
                drop_reasons["invalid_multiple_choice"] += 1
                continue
            kept.append(item)
        if max_items is not None and raw_count >= max_items:
            break

    return DatasetBuildResult(
        dataset="pmc_vqa",
        records=kept,
        raw_count=raw_count,
        kept_count=len(kept),
        dropped_count=raw_count - len(kept),
        drop_reasons=dict(drop_reasons),
    )


def _load_omnimedvqa(
    *,
    output_dir: Path,
    cache_images: bool,
    smoke: bool,
    max_items: int | None,
) -> DatasetBuildResult:
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download("foreverbeliever/OmniMedVQA", "OmniMedVQA.zip", repo_type="dataset")
    archive = ZipImageCache(zip_path)
    raw_count = 0
    kept: list[VQAItem] = []
    drop_reasons: Counter[str] = Counter()
    json_members = [
        name
        for name in archive.names
        if "QA_information/Open-access/" in name and name.lower().endswith(".json")
    ]
    for member in json_members:
        with archive.archive.open(member) as handle:
            payload = json.load(handle)
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        for idx, row in enumerate(rows):
            if max_items is not None and raw_count >= max_items:
                break
            raw_count += 1
            image_ref = normalize_text(row.get("image_path", ""))
            choices = [row.get(f"option_{letter}") for letter in ["A", "B", "C", "D"]]
            gt_answer = normalize_text(row.get("gt_answer", ""))
            answer_label = None
            cleaned_choices = [clean_choice_text(choice) for choice in choices]
            for letter_idx, option in enumerate(cleaned_choices):
                if normalize_answer(option) == normalize_answer(gt_answer):
                    answer_label = chr(ord("A") + letter_idx)
                    break
            if answer_label is None:
                drop_reasons["gt_not_in_options"] += 1
                continue
            uid = f"omnimedvqa_{safe_uid_part(row.get('question_id', f'{Path(member).stem}_{idx}'))}"
            local_path = output_dir / "omnimedvqa" / "images" / image_ref
            if cache_images and not local_path.exists():
                if not archive.extract(image_ref, local_path):
                    drop_reasons["missing_image"] += 1
                    continue
            item = make_choice_item(
                uid=uid,
                dataset="omnimedvqa",
                modality=normalize_text(row.get("modality_type", "unknown")) or "unknown",
                image_path=str(local_path),
                question=row.get("question"),
                choices=choices,
                answer_label=answer_label,
                split="test",
            )
            if item is None:
                drop_reasons["invalid_multiple_choice"] += 1
                continue
            kept.append(item)
        if max_items is not None and raw_count >= max_items:
            break
    return DatasetBuildResult(
        dataset="omnimedvqa",
        records=kept,
        raw_count=raw_count,
        kept_count=len(kept),
        dropped_count=raw_count - len(kept),
        drop_reasons=dict(drop_reasons),
    )


def load_dataset_records(dataset_name: str, *, split: str | None = None, smoke: bool = False) -> list[VQAItem]:
    """Load one dataset into the unified schema.

    In smoke mode this avoids network and returns synthetic records.
    """
    if smoke:
        return [
            VQAItem(
                uid=f"{dataset_name}_smoke_{idx}",
                dataset=dataset_name,
                modality="smoke",
                image_path=f"data/{dataset_name}/images/smoke_{idx}.png",
                question="Is this a smoke-test item?",
                options=["yes", "no"],
                gold_idx=0,
                split=split or "train",
            )
            for idx in range(8)
        ]
    result = build_dataset(
        dataset_name,
        output_dir=Path("data"),
        cache_images=True,
        smoke=False,
        max_items=None,
    )
    records = result.records
    if split is not None:
        records = [item for item in records if item.split == split]
    return records


def build_dataset(
    dataset_name: str,
    *,
    output_dir: Path,
    cache_images: bool,
    smoke: bool,
    max_items: int | None,
) -> DatasetBuildResult:
    if smoke:
        records = load_dataset_records(dataset_name, smoke=True)
        return DatasetBuildResult(dataset_name, records, 8, len(records), 0, {})
    if dataset_name == "vqa_rad":
        return _load_hf_yes_no_dataset(
            dataset_key="vqa_rad",
            hf_id="flaviagiammarino/vqa-rad",
            modality="radiology",
            output_dir=output_dir,
            cache_images=cache_images,
            smoke=smoke,
            max_items=max_items,
        )
    if dataset_name == "pathvqa":
        return _load_hf_yes_no_dataset(
            dataset_key="pathvqa",
            hf_id="flaviagiammarino/path-vqa",
            modality="pathology",
            output_dir=output_dir,
            cache_images=cache_images,
            smoke=smoke,
            max_items=max_items,
        )
    if dataset_name == "slake_en":
        return _load_slake(output_dir=output_dir, cache_images=cache_images, smoke=smoke, max_items=max_items)
    if dataset_name == "pmc_vqa":
        return _load_pmc_vqa(output_dir=output_dir, cache_images=cache_images, smoke=smoke, max_items=max_items)
    if dataset_name == "omnimedvqa":
        return _load_omnimedvqa(output_dir=output_dir, cache_images=cache_images, smoke=smoke, max_items=max_items)
    raise KeyError(f"unknown dataset: {dataset_name}")


def assert_no_split_leakage(records: Sequence[VQAItem]) -> None:
    """Check that a uid and a question group appear in only one split."""
    uid_seen: dict[str, str] = {}
    question_seen: dict[tuple[str, str], str] = {}
    for item in records:
        old_split = uid_seen.get(item.uid)
        if old_split is not None and old_split != item.split:
            raise ValueError(f"uid {item.uid} appears in both {old_split} and {item.split}")
        uid_seen[item.uid] = item.split
        key = (item.dataset, normalize_text(item.question).lower())
        old_question_split = question_seen.get(key)
        if old_question_split is not None and old_question_split != item.split:
            raise ValueError(
                f"question group {key} appears in both {old_question_split} and {item.split}"
            )
        question_seen[key] = item.split


def write_records_csv(records: Sequence[VQAItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["uid", "dataset", "modality", "image_path", "question", "options", "gold_idx", "split"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in records:
            row = item.to_dict()
            row["options"] = json.dumps(row["options"], ensure_ascii=False)
            writer.writerow(row)


def write_count_report(results: Sequence[DatasetBuildResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "raw_count",
                "kept_count",
                "dropped_count",
                "split",
                "split_count",
                "drop_reasons",
            ],
        )
        writer.writeheader()
        for result in results:
            split_counts = Counter(item.split for item in result.records)
            for split, count in sorted(split_counts.items()):
                writer.writerow(
                    {
                        "dataset": result.dataset,
                        "raw_count": result.raw_count,
                        "kept_count": result.kept_count,
                        "dropped_count": result.dropped_count,
                        "split": split,
                        "split_count": count,
                        "drop_reasons": json.dumps(result.drop_reasons, sort_keys=True),
                    }
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare medical VQA datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["vqa_rad"],
        help="Dataset keys from configs/datasets.yaml, or 'all'.",
    )
    parser.add_argument("--output-dir", default="data", help="Output data directory.")
    parser.add_argument("--records-out", default="data/vqa_items.csv")
    parser.add_argument("--counts-out", default="results/dataset_counts.csv")
    parser.add_argument("--max-items-per-dataset", type=int, default=None)
    parser.add_argument("--no-cache-images", action="store_true")
    add_runtime_args(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    setup_logging(config.log_level)
    log_smoke_stub(__name__, config, "data preparation")
    dataset_names = DEFAULT_DATASETS if "all" in args.datasets else args.datasets
    output_dir = Path(args.output_dir)
    results = [
        build_dataset(
            dataset_name,
            output_dir=output_dir,
            cache_images=not args.no_cache_images,
            smoke=config.smoke,
            max_items=config.smoke_items if config.smoke else args.max_items_per_dataset,
        )
        for dataset_name in dataset_names
    ]
    records = [item for result in results for item in result.records]
    assert_no_split_leakage(records)
    write_records_csv(records, Path(args.records_out))
    write_count_report(results, Path(args.counts_out))
    for result in results:
        logger.info(
            "dataset=%s raw=%s kept=%s dropped=%s reasons=%s",
            result.dataset,
            result.raw_count,
            result.kept_count,
            result.dropped_count,
            result.drop_reasons,
        )
    logger.info("wrote_records=%s path=%s", len(records), args.records_out)
    logger.info("wrote_counts path=%s", args.counts_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
