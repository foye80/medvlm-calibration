# Data Licenses

This file records dataset access and license notes for the calibration study.
When a Hugging Face dataset card does not clearly declare a license, use the
dataset only under the original source terms and state this limitation in the
paper.

## VQA-RAD

- Resolved source: `flaviagiammarino/vqa-rad`
- Access: public Hugging Face dataset
- License on inspected dataset card: not clearly declared in Phase 1
- Use note: cite the original VQA-RAD source and follow its terms
- Study handling: strict loader keeps yes/no items only, because non-yes/no
  closed items do not expose a full candidate option set in this mirror

## SLAKE English

- Resolved source: `BoKelvin/SLAKE`
- Access: public Hugging Face dataset
- License on inspected dataset card: not clearly declared in Phase 1
- Files inspected: `train.json`, `validation.json`, `test.json`, `imgs.zip`
- Use note: cite the original SLAKE source and follow its terms
- Study handling: strict loader keeps English yes/no items only, because other
  closed labels do not expose explicit answer options in this mirror

## PathVQA

- Resolved source: `flaviagiammarino/path-vqa`
- Access: public Hugging Face dataset
- License on inspected dataset card: not clearly declared in Phase 1
- Use note: cite the original PathVQA source and follow its terms
- Study handling: strict loader keeps yes/no items only

## PMC-VQA

- Resolved source: `xmcmic/PMC-VQA`
- Access: public Hugging Face dataset
- License on inspected dataset card: not clearly declared in Phase 1
- Files inspected: `train.csv`, `test.csv`, `train_2.csv`, `test_2.csv`,
  `test_clean.csv`, `images.zip`, `images_2.zip`
- Use note: PMC-derived figures may overlap with model pretraining corpora;
  this must be stated as a contamination limitation
- Study handling: manual CSV reader is required because the default dataset
  builder fails on mismatched CSV columns. The active loader uses only
  `test_clean.csv`, `test.csv`, and `test_2.csv`, because PMC-VQA is a test
  dataset in this project rather than an adapter-training dataset

## OmniMedVQA

- Resolved source: `foreverbeliever/OmniMedVQA`
- Access: public Hugging Face zip release
- License on inspected dataset card: no single clear license; dataset card says
  users must follow the original dataset licenses and ethical constraints
- Files inspected: `OmniMedVQA.zip`, `README.md`
- Use note: use open-access QA/image subset where images are provided; restricted
  access rows should not be used unless original images are separately available
- Study handling: main zip is preferred over small parquet mirrors because its
  QA JSON includes `modality_type`, which is required for per-modality analysis
