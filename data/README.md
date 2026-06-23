# Data Manifest

`split_manifest.csv` contains derived split identifiers used in the study.

Columns:

- `uid`: project item identifier.
- `dataset`: source benchmark label.
- `modality`: broad modality label used by the project.
- `split`: train, calibration, or test split used by the project.
- `gold_idx`: index of the correct option in the original processed item.

The manifest intentionally excludes original questions, answer-option text,
image paths, and image files. Obtain the original datasets from their providers
and follow the terms summarized in `../DATA_LICENSES.md`.
