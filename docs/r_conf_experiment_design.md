# High-Confidence Error Concentration Analysis

## Objective

Quantify whether fine-tuning changes how often model errors are made with high confidence. This analysis is intentionally different from expected calibration error (ECE): ECE measures average calibration gap, whereas this analysis asks whether the remaining errors are concentrated in high-confidence predictions.

## Metric

For model `m`, training condition `t`, evaluation dataset `e`, and threshold `alpha`:

```text
r_conf(alpha) =
  #{i: prediction_i is wrong and confidence_i >= alpha}
  / #{i: prediction_i is wrong}
```

Equivalently:

```text
r_conf(alpha) = P(confidence >= alpha | wrong)
```

Interpretation: among all wrong answers, the fraction that were assigned confidence at least `alpha`.

## Thresholds

The alpha sweep is:

```text
0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99
```

The primary manuscript threshold is `alpha = 0.90`; the remaining thresholds are a sensitivity curve.

## Analysis Cells

Models:

```text
qwen25vl, internvl, llavaov, smolvlm, medgemma, huatuo
```

Evaluation datasets:

```text
vqa_rad, slake_en, pathvqa
```

Conditions:

```text
zero-shot:       model m evaluated on dataset e
in-dataset FT:  model m fine-tuned on e and evaluated on e
cross-dataset FT: model m fine-tuned on t and evaluated on e, where t != e
```

Expected clean-test cells:

```text
zero-shot:       6 models * 3 eval datasets = 18 cells
in-dataset FT:  6 models * 3 eval datasets = 18 cells
cross-dataset FT: 6 models * 3 eval datasets * 2 off-diagonal train datasets = 36 cells
```

## Delta Definition

Each fine-tuned cell is paired against the same model and evaluation dataset in zero-shot mode:

```text
delta_r_conf(alpha) =
  r_conf_zero_shot(model=m, eval=e, alpha=alpha)
  - r_conf_fine_tuned(model=m, train=t, eval=e, alpha=alpha)
```

Direction:

```text
delta_r_conf > 0: fine-tuning reduced high-confidence error concentration
delta_r_conf = 0: no change
delta_r_conf < 0: fine-tuning increased high-confidence error concentration
```

For a case-count interpretation, convert the proportion difference back to the number of fine-tuned errors:

```text
case_delta_high_conf_errors =
  delta_r_conf(alpha) * n_errors_fine_tuned
```

Direction:

```text
case_delta > 0: fewer high-confidence errors than expected under the matched zero-shot concentration
case_delta < 0: excess high-confidence errors after fine-tuning
```

The complementary positive burden is:

```text
excess_high_conf_errors_after_ft = -case_delta_high_conf_errors
```

## Statistical Summary

For each `alpha` and setting (`ft_in_dataset`, `ft_cross_dataset`), report:

```text
number of paired cells
mean delta_r_conf
median delta_r_conf
bootstrap 95% CI for mean delta_r_conf
paired sign-flip permutation p-value for mean delta_r_conf
number and fraction of cells with delta_r_conf < 0
```

Because the two cross-dataset fine-tuned conditions for the same model and evaluation dataset share one zero-shot baseline, the analysis also reports a collapsed cross-dataset summary. In that summary, the two off-diagonal cross-dataset deltas are averaged within each `model x eval_dataset x alpha` unit before inference, yielding 18 paired units rather than 36 condition-level units.

For descriptive comparison, also report cell-level mean and pooled `r_conf(alpha)` for zero-shot, in-dataset FT, and cross-dataset FT.

## Output Files

The analysis script writes:

```text
results/r_conf_cell_values.csv
results/r_conf_delta_values.csv
results/r_conf_paired_wide.csv
results/r_conf_summary_by_alpha.csv
results/r_conf_delta_summary_by_alpha.csv
results/r_conf_delta_summary_cross_eval_mean_by_alpha.csv
results/r_conf_case_delta_summary_by_alpha.csv
```
