# Findings — Medical VLM Calibration & Selective Prediction after LoRA Fine-tuning

Generated 2026-06-09. Numbers sourced from `results/master_metrics.csv` (clean grid,
72 cells, 1000× bootstrap CIs), `results/temperature_scaling_calib.csv` (RQ3),
`results/rq4_corruption_metrics.csv` (RQ4b, 396 cells), `results/rq5_modality_metrics.csv`
(RQ5). 6 models × 3 datasets (VQA-RAD, SLAKE-en, PathVQA), closed-ended VQA, QLoRA r=16.

---

## RQ1 — Are fine-tuned medical VLMs calibrated? Does FT help or hurt?

**Paired zero-shot → FT on in-distribution (ID) test (18 model×dataset cells):**

| | Accuracy | ECE |
|---|---|---|
| Zero-shot | 0.665 | 0.179 |
| FT (ID) | 0.776 | 0.132 |
| Δ (FT − zero-shot) | **+0.112** | **−0.047** |

- Fine-tuning **improves both accuracy and average calibration** — the *opposite* of the
  common "FT makes models overconfident" assumption.
- But the effect is **heterogeneous**: ECE *worsens* in **4/18** ID cells. The pattern is
  driven by initial calibration: models poorly calibrated zero-shot improve most; models
  already well-calibrated are at risk of degradation.
  - Risky case: `qwen25vl + vqa_rad`, Δacc ≈ −1.6pp yet ECE jumps (well-calibrated → worse).
  - Catastrophic archived case: HuatuoGPT + PathVQA adapter collapse (acc 67%→37%) — fixed
    by grad-clipping + ~1 epoch; underscores per-run QC before analysis.

### Key contribution: medical pre-training ≠ better calibration

Zero-shot ECE per model (lower = better):

| Model | Type | ECE |
|---|---|---|
| huatuo | **medical** 7B | 0.151 |
| internvl | general 8B | 0.177 |
| qwen25vl | general 7B | 0.179 |
| medgemma | **medical** 4B | 0.223 |
| smolvlm | general 2.2B | 0.239 |
| llavaov | general 7B | 0.240 |

The two medical models sit at opposite extremes (huatuo best-but-one, medgemma worst-but-two).
**Calibration quality tracks initial calibration level and model scale/training, not the
medical/general domain split.** Discussion should frame the axis as initial-ECE, not domain.

---

## RQ2 — Which confidence signal best detects errors?

Error-detection AUROC, averaged over FT-ID cells:

| Signal | AUROC |
|---|---|
| Option-softmax (max prob) | 0.707 |
| Entropy | 0.707 |
| Mean token logprob | 0.707 |

- Option-softmax and entropy are **mathematically near-equivalent for binary yes/no** items,
  which dominate the datasets — hence identical AUROC.
- Selective prediction works: abstaining at 70% coverage raises accuracy by **+6.6pp** on
  average (up to +11.5pp), but not in every cell.
- **Verbalized confidence:** implemented (`src/verbalized.py`) and a 36-cell sweep is running.
  Smoke results: internvl/qwen/medgemma/huatuo follow the "0–100 confidence" instruction
  (parse rate ≈1.0); **llavaov degenerates (always ~100, skips the answer) and smolvlm
  refuses to verbalize at all (parse rate ≈0)** — a finding in itself: smaller/some VLMs
  cannot self-report usable confidence. AUROC integration pending sweep completion.
- **Self-consistency:** deferred (N=10 sampling too expensive); documented in Limitations.

---

## RQ3 — Does temperature scaling recover calibration?

Fit single scalar T on the held-out **calib** split, applied to clean test (72 cells):

| | Before | After |
|---|---|---|
| Mean ECE | 0.201 | **0.081** |
| Mean NLL | 0.952 | **0.587** |

- ECE improved in **87.5%** of cells, NLL in **91.7%**. Mean T ≈ 6.53.
- Temperature scaling is a **cheap, effective post-hoc fix** — recovers most calibration at
  zero accuracy cost (T scaling is monotonic, preserves argmax/coverage).

---

## RQ4 — How does calibration degrade under distribution shift?

### RQ4a — Cross-dataset shift (severe)

| FT condition | Accuracy | ECE |
|---|---|---|
| ID (matching test) | 0.776 | 0.132 |
| Cross-dataset | 0.612 | **0.247** |

Cross-dataset deployment costs −16.4pp accuracy and **+11.5pp ECE** — the dominant
failure mode for safe deployment.

### RQ4b — Image corruptions (mild, graceful)

Mean over 6 models × 7 corruptions × 3 ID datasets (`fig3_corruption_degradation.png`):

| Severity | Accuracy | ECE | AURC | Sel-Acc@70% |
|---|---|---|---|---|
| 0 (clean) | 0.776 | 0.132 | 0.138 | 0.839 |
| 1 | 0.778 | 0.133 | 0.138 | 0.838 |
| 2 | 0.774 | 0.134 | 0.140 | 0.833 |
| 3 | 0.765 | 0.137 | 0.146 | 0.824 |

- Image corruptions cause only **mild, graceful** degradation (severity 3: −1.2pp acc,
  +0.5pp ECE). **Cross-dataset shift hurts ~20× more than pixel-level corruption** — the
  clean headline for RQ4.

---

## RQ5 — Does calibration differ across imaging modalities?

Per-modality (OmniMedVQA, zero-shot averaged over models; `fig4_modality_calibration.png`):

| Modality | Accuracy | ECE | AURC |
|---|---|---|---|
| OCT | 0.563 | **0.092** | 0.214 |
| Fundus Photography | 0.487 | 0.158 | 0.452 |
| Dermoscopy | 0.506 | 0.240 | 0.345 |
| Microscopy | 0.423 | 0.252 | 0.547 |
| MR | 0.392 | 0.275 | 0.547 |
| X-Ray | 0.383 | 0.321 | 0.647 |
| CT | 0.204 | 0.580 | 0.865 |
| Ultrasound | 0.083 | **0.698** | 0.932 |

- Calibration varies **dramatically by modality**: ECE spans 0.09 (OCT) to 0.70 (ultrasound),
  an ~8× range. Modalities with low accuracy (ultrasound, CT) are also wildly overconfident
  (high ECE + near-1.0 AURC = confidence useless for abstention).
- Per-modality reliability must be audited before clinical deployment; a single aggregate
  calibration number hides catastrophic per-modality failure.

---

## Headline narrative

1. FT generally *improves* calibration on ID data, but heterogeneously (initial-ECE driven).
2. **Medical pre-training does not guarantee calibration.**
3. Temperature scaling cheaply recovers calibration (ECE 0.20→0.08).
4. The real danger is **distribution shift** — cross-dataset (ECE +11.5pp) ≫ image corruption (+0.5pp).
5. Calibration is **strongly modality-dependent** (ECE 0.09–0.70).

## Excluded (Limitations)
- **PMC-VQA**: OmniMedVQA covers cross-modality (RQ5); PMC-VQA's A–D format diverges and has
  pretraining-contamination risk.
- **Self-consistency**: N=10 sampling too expensive for the full grid.
