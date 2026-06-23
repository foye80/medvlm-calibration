# Result Artifacts

This directory contains aggregate metrics and manuscript-supporting tables.

Included top-level files:

- `master_metrics.csv`
- `temperature_scaling_calib.csv`
- `rq2_verbalized_metrics.csv`
- `rq4_corruption_metrics.csv`
- `rq5_modality_metrics.csv`
- `dataset_counts.csv`
- `findings.md`
- `novelty_check.md`

The `supplementary_tables/` directory contains CSV versions of the supplementary
tables prepared for journal submission.

Large raw per-item prediction CSVs are not included in this GitHub release.
They can be regenerated with the inference scripts if datasets and model
checkpoints are available locally.
