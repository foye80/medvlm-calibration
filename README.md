# Medical VLM Confidence Reliability

This repository contains the public code and derived artifacts for the study:

**Fine-Tuning Improves but Does Not Guarantee Confidence Reliability in Medical Vision-Language Models**

The project audits whether confidence scores from open-weight medical
vision-language models remain reliable after parameter-efficient fine-tuning.
It is not an accuracy leaderboard and does not introduce a new model. The core
focus is calibration, selective prediction, high-confidence errors, and
cross-dataset shift.

## What Is Included

- `src/`: dataset schema/loading utilities, model registry, option-likelihood
  scoring, fine-tuning, inference, calibration, metrics, corruption transforms,
  verbalized-confidence parsing, and plotting code.
- `scripts/`: reproducibility scripts for data preparation, fine-tuning,
  inference, calibration, metric aggregation, and figure generation.
- `tests/`: unit tests for core confidence, scoring, metrics, corruption, data,
  fine-tuning, and calibration utilities.
- `results/`: final aggregate metric tables used for manuscript analyses.
- `results/supplementary_tables/`: CSV versions of supplementary tables.
- `figures/`: generated manuscript and supplementary figures.
- `data/split_manifest.csv`: split identifiers without original questions,
  answer text, image paths, or image files.
- `DATA_LICENSES.md`: dataset access and license notes.
- `PROJECT_SPEC.md`: original project specification.

## What Is Not Included

This repository intentionally does **not** include original medical images,
dataset mirrors, model weights, LoRA/QLoRA adapters, Hugging Face caches, logs,
or large raw per-item prediction CSVs. The source datasets should be obtained
from their original providers under their own terms. Raw prediction files are
large and will be better suited to an archival data repository if public release
is required by the journal.

## Main Result Tables

- `results/master_metrics.csv`: clean zero-shot, in-distribution fine-tuned,
  and cross-dataset evaluation metrics.
- `results/temperature_scaling_calib.csv`: temperature scaling fitted on
  held-out calibration splits and evaluated on test splits.
- `results/rq2_verbalized_metrics.csv`: verbalized-confidence analysis.
- `results/rq4_corruption_metrics.csv`: same-domain image-corruption metrics.
- `results/rq5_modality_metrics.csv`: OmniMedVQA modality-level metrics.
- `results/findings.md`: narrative result summary.

## Reproduction Outline

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run tests:

```bash
PYTHONPATH="$PWD" python -m pytest -q tests
```

Regenerate aggregate metrics from existing prediction files, if available:

```bash
python scripts/phase5_aggregate.py
python scripts/36_temperature_scaling_from_calib.py --require-all \
  --out results/temperature_scaling_calib.csv
python scripts/37_rq4_corruption_metrics.py
python scripts/33_metrics_rq5.py
python scripts/38_rq2_verbalized_metrics.py
```

Regenerate manuscript-style figures:

```bash
python scripts/41_make_paper_style_figures.py
python scripts/42_reliability_fingerprint.py
python scripts/43_r_conf_analysis.py
python scripts/44_plot_r_conf_alpha_delta.py
python scripts/45_plot_r_conf_case_delta.py
```

Full model inference and QLoRA fine-tuning require local GPU resources and
access to the relevant open-weight checkpoints. Some checkpoints or datasets may
require users to accept provider terms before download.

## Dataset Notes

The study uses public benchmark datasets and model checkpoints. Dataset licenses
and access notes are summarized in `DATA_LICENSES.md`. Because several dataset
cards do not declare a single clear license, this repository releases derived
split identifiers and aggregate outputs rather than redistributing original
questions, answer text, images, or full dataset mirrors.

## Citation

Please cite the manuscript when it becomes available. Until then, cite this
repository as the code and derived-results release for the above study.
