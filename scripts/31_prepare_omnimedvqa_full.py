#!/usr/bin/env python3
"""Prepare full OmniMedVQA items CSV for RQ5 (per-modality calibration).

Samples up to --max-per-modality items per modality (default 200), extracts images
from the HF-cached zip, and writes data/omnimedvqa_items.csv.

Usage:
    python scripts/31_prepare_omnimedvqa_full.py [--max-per-modality 200] [--out data/omnimedvqa_items.csv]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
SKIP_MODALITIES = {"unknown"}

def find_zip() -> Path:
    env_zip = os.environ.get("OMNIMEDVQA_ZIP")
    if env_zip and os.path.exists(env_zip):
        return Path(env_zip)
    log.info("local cache not found, downloading via hf_hub_download...")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("foreverbeliever/OmniMedVQA", "OmniMedVQA.zip", repo_type="dataset")
    return Path(path)


def normalize(v: object) -> str:
    return str(v).strip()


def normalize_answer(v: object) -> str:
    return normalize(v).lower().strip(" \t\n\r.。:;\"'")


def safe_uid(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-modality", type=int, default=200)
    parser.add_argument("--out", default="data/omnimedvqa_items.csv")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    random.seed(SEED)
    zip_path = find_zip()
    log.info("zip_path=%s", zip_path)

    data_root = Path(args.data_root)
    image_base = data_root / "omnimedvqa" / "images"

    # ── collect all valid items grouped by modality ──────────────────────────
    by_modality: dict[str, list[dict]] = defaultdict(list)

    with zipfile.ZipFile(zip_path) as z:
        zip_names = set(z.namelist())
        json_members = sorted(
            n for n in zip_names
            if "QA_information/Open-access/" in n and n.lower().endswith(".json")
        )
        log.info("found %d QA JSON files", len(json_members))

        for member in json_members:
            with z.open(member) as fh:
                payload = json.load(fh)
            rows = payload if isinstance(payload, list) else payload.get("data", [])
            for idx, row in enumerate(rows):
                modality = normalize(row.get("modality_type", "unknown")) or "unknown"
                if modality in SKIP_MODALITIES:
                    continue

                image_ref = normalize(row.get("image_path", ""))
                image_in_zip = f"OmniMedVQA/{image_ref}"
                if image_in_zip not in zip_names:
                    continue

                gt = row.get("gt_answer", "")
                choices = [row.get(f"option_{l}") for l in "ABCD"]
                gold_idx = None
                for li, ch in enumerate(choices):
                    if ch is not None and normalize_answer(ch) == normalize_answer(gt):
                        gold_idx = li
                        break
                if gold_idx is None:
                    continue

                qid = safe_uid(normalize(row.get("question_id", f"{Path(member).stem}_{idx}")))
                uid = f"omnimedvqa_{qid}"
                local_path = image_base / image_ref

                by_modality[modality].append({
                    "uid": uid,
                    "dataset": "omnimedvqa",
                    "modality": modality,
                    "image_path": str(local_path),
                    "question": normalize(row.get("question", "")),
                    "options": json.dumps([normalize(c) for c in choices], ensure_ascii=False),
                    "gold_idx": gold_idx,
                    "split": "test",
                    "_image_in_zip": image_in_zip,
                })

        log.info("modalities: %s", {m: len(v) for m, v in sorted(by_modality.items())})

        # ── sample ────────────────────────────────────────────────────────────
        selected: list[dict] = []
        for modality, items in sorted(by_modality.items()):
            sample = random.sample(items, min(args.max_per_modality, len(items)))
            sample.sort(key=lambda r: r["uid"])
            selected.extend(sample)
            log.info("modality=%-50s  total=%5d  sampled=%d", modality, len(items), len(sample))

        log.info("total selected: %d", len(selected))

        # ── extract images ────────────────────────────────────────────────────
        log.info("extracting images...")
        extracted = skipped = 0
        for item in selected:
            dest = Path(item["image_path"])
            if dest.exists():
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item["_image_in_zip"]) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted += 1
        log.info("extracted=%d  skipped_existing=%d", extracted, skipped)

    # ── write CSV ─────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["uid", "dataset", "modality", "image_path", "question", "options", "gold_idx", "split"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            writer.writerow({k: item[k] for k in fieldnames})

    log.info("wrote %d rows → %s", len(selected), out_path)


if __name__ == "__main__":
    main()
