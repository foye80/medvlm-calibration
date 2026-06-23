# Project Spec — Calibration & Selective Prediction of PEFT-Fine-Tuned Medical VLMs

> **Hand this whole file to Claude Code.** It is the single source of truth for the project. Work phase by phase, check off tasks as you go, and stop to report at the end of each phase. Do not invent scope beyond what is written here without flagging it first.

---

## 0. One-paragraph summary

We do **not** propose a new method. We run a **rigorous empirical study** of how trustworthy the *confidence* of open-weight medical Vision-Language Models (VLMs) is **after LoRA fine-tuning** on closed-ended medical VQA. We measure, across multiple model families and imaging modalities: (1) calibration (does fine-tuning fix or break it?), (2) how well simple confidence signals support selective prediction / abstention, (3) whether cheap post-hoc calibration (temperature scaling) recovers reliability, and (4) how all of this degrades under distribution shift (cross-dataset + image corruptions). The headline expected story is a **safety-flavored cautionary result**: fine-tuning raises accuracy but tends to make models **overconfident**, naive confidence is unsafe for clinical deployment, yet a 1-parameter temperature fit recovers most calibration. Negative/cautionary results are fully publishable here.

**Target venues :** *Lancet Digital Health*, *Nature Communications*, *npj Digital Medicine*, *IEEE-TPAMI*, *Journal of Imaging*, *IEEE Access*. Stretch: *Nature Machine Intelligence* (robustness/safety angle).

---

## 1. Why this is novel (novelty guard — READ BEFORE STARTING)

The "fine-tune N VLMs on modality X and report accuracy" template is **saturated** (radiology VQA-RAD/SLAKE, ROCOv2 captioning, GI endoscopy/Kvasir, ultrasound, dermatology, dental — all done in 2025–2026). **Do not frame this as an accuracy leaderboard.** The gap we occupy is the **evaluation axis**: a cross-modality, cross-model, PEFT-before-vs-after **descriptive study of calibration and selective-prediction reliability**. Related but distinct prior work:
- Medical-VLM **hallucination mitigation** benchmarks (e.g. MedHallTune) — those *propose a method/dataset to reduce* hallucination; we *measure* confidence reliability, not propose mitigation.
- **Conformal abstention / selective prediction** papers — general-domain LLM/VLM **method** papers, not a medical cross-modality empirical characterization of PEFT effects.

**Action item (Phase 6, before writing):** re-run a quick literature check for any paper titled/abstracted as "calibration of fine-tuned medical VLMs" or "selective prediction medical VQA fine-tuning." If a near-duplicate appeared, escalate to the human; we then pivot the framing to the **distribution-shift** axis (RQ4) or the **confidence-signal comparison** as the primary contribution.

---

## 2. Research questions (each maps to a figure/table)

- **RQ1 — Calibration.** Are LoRA-fine-tuned medical VLMs calibrated? Does fine-tuning improve or worsen calibration relative to the zero-shot base model? → *Fig. 1: ECE before vs after fine-tuning (per model × dataset); reliability diagrams.*
- **RQ2 — Selective prediction.** How well do simple confidence signals (option-softmax max prob, entropy, verbalized confidence, self-consistency) support abstention? Which signal best separates correct vs incorrect? → *Fig. 2: risk–coverage curves + AURC; Table: error-detection AUROC per signal.*
- **RQ3 — Post-hoc calibration.** Does temperature scaling (1 param) / Platt scaling cheaply recover calibration? At what cost to coverage? → *Table: ECE/AURC before vs after temperature scaling.*
- **RQ4 — Distribution shift.** How do calibration and selective prediction degrade under (a) cross-dataset shift and (b) image corruptions? → *Fig. 3: ECE / selective-accuracy degradation vs corruption severity; cross-dataset transfer matrix.*
- **RQ5 (secondary) — Per-modality.** Using OmniMedVQA modality labels, does calibration differ systematically across imaging modalities? → *Fig. 4: per-modality ECE/AURC.*

---

## 3. Scope decisions (locked — do not change silently)

- **Closed-ended / multiple-choice items only.** We need a fixed answer set per question so that "confidence" = a proper probability over candidate answers and correctness is unambiguous. Drop open-ended free-text items from all datasets. This is a deliberate, defensible scope; state it explicitly in the paper Limitations.
- **Open-weight local models only** (we need logits / per-option likelihoods; closed APIs cannot give clean option probabilities). No GPT-4o/Gemini in the core study; optionally cite their zero-shot numbers from literature only.
- **LoRA/QLoRA PEFT only.** Vision encoder frozen. Identical PEFT hyperparameters across all models for fairness.

---

## 4. Datasets (all public; verify exact HF IDs at runtime, they drift)

Use the **closed-ended / yes-no / multiple-choice** subset of each. Build a unified schema (see §7.2).

| Short name | HF id (verify) | Modality | Answer type | Role |
|---|---|---|---|---|
| VQA-RAD | `flaviagiammarino/vqa-rad` | Radiology (X-ray/CT/MRI) | yes-no + closed | train + ID test |
| SLAKE-en | `BoKelvin/SLAKE` (English split) | Radiology | closed | train + cross test |
| PathVQA | `flaviagiammarino/path-vqa` | Histopathology | yes-no | train + cross test |
| PMC-VQA | `xmcmic/PMC-VQA` | Mixed (figures) | multiple-choice (A–D) | ID test |
| OmniMedVQA | official release on HF | **Many** modalities | multiple-choice | per-modality test (RQ5) |

**Gating / license notes:** record each dataset's license in `DATA_LICENSES.md`. PMC-VQA and OmniMedVQA derive from PMC figures — note potential train/pretrain contamination in Limitations. If a dataset id 404s, search HF for the canonical mirror and log the resolved id in `configs/datasets.yaml`.

**Train/calibration/test splits:** for each train dataset create `train / calib / test` = 80/10/10 by **question**, stratified by answer label, fixed seed `42`. The `calib` split is used **only** to fit temperature scaling (never for gradient updates).

---

## 5. Models (6; mix of general/medical and sizes)

| Short name | HF id (verify) | Family | Size | Type |
|---|---|---|---|---|
| qwen25vl | `Qwen/Qwen2.5-VL-7B-Instruct` | Qwen-VL | 7B | general |
| internvl | `OpenGVLab/InternVL2_5-8B` | InternVL | 8B | general |
| llavaov | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | LLaVA-OV | 7B | general |
| smolvlm | `HuggingFaceTB/SmolVLM-Instruct` | SmolVLM | ~2.2B | general (small) |
| medgemma | `google/medgemma-4b-it` | Gemma | 4B | **medical** (gated — accept license) |
| huatuo | `FreedomIntelligence/HuatuoGPT-Vision-7B` | LLaVA-ish | 7B | **medical** |

- `medgemma` is **gated**: the human must accept the license on HF and provide an `HF_TOKEN`. If unavailable, fall back to `microsoft/llava-med-v1.5-mistral-7b` (verify checkpoint loads) as the second medical model.
- The size axis (smolvlm 2.2B vs 7–8B) and the general-vs-medical axis are both analysis dimensions — keep them tracked in metadata.

---

## 6. Method — confidence extraction (the technically critical part)

For a closed/MC question with candidate answer strings `A = {a_1, ..., a_k}` (k=2 for yes/no), build the chat-formatted prompt `P` = image + question (+ enumerated options for MC). Then:

### 6.1 Primary signal — option log-likelihood softmax
For each candidate `a_i`, run a **teacher-forced** forward pass on `concat(P, a_i)`, mask labels so loss is computed **only over the answer tokens**, and take the **length-normalized** sum of token log-probs:
```
LL_i = (1 / |tokens(a_i)|) * sum_t log p(a_i_t | P, a_i_<t)
```
Define the answer distribution and prediction:
```
p_i = softmax_i(LL / 1.0)            # temperature T=1 at eval time
pred = argmax_i p_i
conf = max_i p_i                      # primary confidence
```
- Implement both length-normalized (primary) and raw-sum (ablation) variants behind a flag.
- For yes/no this is a clean 2-way softmax.

### 6.2 Alternative confidence signals (for RQ2 comparison)
1. **Entropy** of `p` (lower entropy = higher confidence).
2. **Verbalized confidence**: additionally prompt the model "On a scale 0–100, how confident are you?" parse the integer. (Free-form generation; robust parser with fallback to NaN.)
3. **Self-consistency**: sample `N=10` generations at `temp=0.7`, confidence = fraction agreeing with the majority answer.
4. **Mean token logprob** of the greedily generated answer.

All signals are computed per item and stored; RQ2 compares them by error-detection AUROC.

### 6.3 Temperature scaling (RQ3)
Fit a single scalar `T` on the **calib** split by minimizing NLL of `softmax(LL / T)` against gold labels (1-D optimization, `scipy.optimize.minimize_scalar` or LBFGS). Apply `T` at test time. Also implement Platt scaling as a secondary baseline.

---

## 7. Metrics (implement all in `src/metrics.py`, with bootstrap CIs)

### 7.1 Calibration
- **ECE** (M=15 equal-width bins): `ECE = Σ_b (|B_b|/N) · |acc(B_b) − conf(B_b)|`.
- **Adaptive ECE** (equal-mass bins), **MCE**, **Brier score**, **NLL**.
- **Reliability diagram** data (bin centers, acc, conf, counts).

### 7.2 Selective prediction
- **Risk–coverage curve**: sort by `conf` desc; at coverage `c`, risk = error rate on top-c fraction. **AURC** = area under it (lower better). Also **E-AURC** (excess over optimal).
- **Selective accuracy @ coverage** ∈ {100, 90, 80, 70, 50}%.
- **Coverage @ risk ≤ 5%** (i.e., max coverage maintaining ≥95% selective accuracy).
- **Error-detection AUROC**: positive class = "prediction is wrong", score = `−conf`. Computed per confidence signal (§6.2).

### 7.3 Reporting
Every metric reported with **95% bootstrap CIs** (1000 resamples over test items). Paired comparisons (zero-shot vs fine-tuned; pre- vs post-temperature) via bootstrap difference CIs. Save raw per-item predictions so any metric can be recomputed without re-running models.

---

## 8. Corruptions (RQ4b) — `src/corruptions.py`

Test-time only (never in training). Medical-relevant, severity ∈ {1,2,3}:
`gaussian_noise`, `gaussian_blur`, `motion_blur`, `brightness_shift`, `contrast_shift`, `jpeg_compression`, `downscale_upscale` (resolution loss). Use clean, well-documented implementations (imagecorruptions lib or hand-rolled with PIL/numpy). Apply to the **ID test set** of each dataset.

---

## 9. Experiment matrix

- **Adapters to train:** 6 models × 3 train datasets {VQA-RAD, SLAKE-en, PathVQA} = **18 LoRA adapters**.
- **Zero-shot baselines:** 6 models, no fine-tuning.
- **Evaluation grid:** for each (model, condition) where condition ∈ {zero-shot, FT-on-D for D in 3 train sets}, evaluate on:
  - **ID test** of the matching train dataset,
  - **cross-dataset** test sets (the other radiology/pathology sets),
  - **corrupted** ID test (7 corruptions × 3 severities),
  - **OmniMedVQA / PMC-VQA** for cross-modality (RQ5, primarily on zero-shot + best FT).
- Keep it tractable: full corruption sweep only for FT-on-ID condition; cross-dataset matrix at clean severity only.

---

## 10. Repository layout (create this)

```
medvlm-calibration/
├── README.md                      # quickstart + reproduce instructions
├── PROJECT_SPEC.md                # this file
├── DATA_LICENSES.md               # filled during Phase 1
├── requirements.txt
├── configs/
│   ├── datasets.yaml              # resolved HF ids, splits, answer-set rules
│   ├── models.yaml                # resolved HF ids, chat templates, dtype
│   ├── lora.yaml                  # shared PEFT hyperparams
│   └── experiments.yaml           # the matrix in §9
├── src/
│   ├── data/
│   │   ├── load.py                # → unified schema records
│   │   └── schema.py              # dataclass: id, image, question, options, gold, modality, dataset
│   ├── models/
│   │   ├── registry.py            # load model + processor + chat template per id
│   │   └── score.py               # option log-likelihood scoring (§6.1)
│   ├── finetune.py                # QLoRA SFT (trl SFTTrainer)
│   ├── infer.py                   # produce per-item predictions + all confidence signals
│   ├── confidence.py              # signals §6.2
│   ├── calibrate.py               # temperature + Platt scaling §6.3
│   ├── corruptions.py             # §8
│   ├── metrics.py                 # §7, with bootstrap CIs
│   └── plots.py                   # Figs 1–4
├── scripts/
│   ├── 00_prepare_data.sh
│   ├── 10_finetune_all.sh
│   ├── 20_infer_all.sh
│   ├── 30_compute_metrics.sh
│   └── 40_make_figures.sh
├── results/                       # per-item parquet + aggregated csv (gitignored if large)
├── figures/
└── tests/                         # unit tests for metrics + scoring
```

### Unified record schema (`src/data/schema.py`)
```python
@dataclass
class VQAItem:
    uid: str            # f"{dataset}_{idx}"
    dataset: str
    modality: str       # e.g. "xray","ct","mri","pathology","mixed", or OmniMed label
    image_path: str
    question: str
    options: list[str]  # e.g. ["yes","no"] or ["A) ...","B) ...",...]
    gold_idx: int
    split: str          # train | calib | test
```

---

## 11. Environment

```
python>=3.10
torch (CUDA build matching the box)
transformers>=4.49, peft>=0.13, trl>=0.12, accelerate, bitsandbytes
datasets, huggingface_hub
pillow, numpy, scipy, scikit-learn, pandas, pyarrow
matplotlib, seaborn
imagecorruptions   # or hand-rolled fallback
pytest
```
- Use **QLoRA (4-bit, bitsandbytes)** so 7–8B models fit on a single 24–48 GB GPU.
- Set `HF_TOKEN` env var for gated models. Pin all versions in `requirements.txt`.
- Global seed = 42 everywhere; log `transformers`/`torch` versions and GPU into each result file.

---

## 12. Shared LoRA config (`configs/lora.yaml`) — identical across models

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
freeze_vision_encoder: true
bits: 4                  # QLoRA
learning_rate: 1.0e-4
lr_scheduler: cosine
warmup_ratio: 0.03
num_train_epochs: 3
per_device_batch_size: 4
grad_accum: 4
max_seq_len: 1024
bf16: true
seed: 42
```
Note: `target_modules` names differ per architecture — `registry.py` must resolve the correct linear-layer names for each model (Gemma/Qwen/InternVL/LLaVA differ). Implement a per-model override map.

---

## 13. Phased task list for Claude Code

> Work top-to-bottom. After each phase: run `pytest`, commit, and print a short status report (what ran, key numbers, anything that needs a human). **Do not** proceed past a phase that has failing smoke tests.

### Phase 0 — Scaffold ☐
- [ ] Create repo layout (§10), `requirements.txt`, `README.md` skeleton.
- [ ] Implement `schema.py`, stub all modules with type-hinted signatures.
- [ ] Set up logging + a `--smoke` flag everywhere that runs on 8 items / 1 step.

### Phase 1 — Data ☐
- [ ] In `configs/datasets.yaml`, resolve and **verify** each HF dataset id (search HF if 404).
- [ ] `src/data/load.py`: download, filter to closed/MC items, normalize to `VQAItem`, build `train/calib/test` (80/10/10, stratified, seed 42), cache images to `data/<dataset>/images/`.
- [ ] For OmniMedVQA, preserve the per-item **modality** label (needed for RQ5).
- [ ] Write `DATA_LICENSES.md`. Print counts per dataset/split/answer-type.
- [ ] Unit test: schema round-trip, no train/test image leakage.

### Phase 2 — Model loading + scoring ☐
- [ ] `src/models/registry.py`: load each model + processor; resolve chat template + correct image token handling per family; resolve LoRA `target_modules` per family.
- [ ] `src/models/score.py`: implement option log-likelihood scoring (§6.1), length-norm + raw variants. **Validate** on a 2-option toy where the obviously-correct answer must score higher.
- [ ] Smoke test: zero-shot accuracy of each model on 50 VQA-RAD items is non-trivial (> chance).

### Phase 3 — Fine-tuning ☐
- [ ] `src/finetune.py`: QLoRA SFT via trl `SFTTrainer`, frozen vision encoder, config from `lora.yaml`. Save adapter + training log per (model, dataset).
- [ ] `scripts/10_finetune_all.sh`: loop the 18 adapters. Resume-safe (skip existing).
- [ ] Smoke: 1 model × 1 dataset × 1 epoch trains and loss decreases.
- [ ] Log GPU-hours per run.

### Phase 4 — Inference + confidence ☐
- [ ] `src/infer.py`: for each (model, condition, test set) produce a **parquet** of per-item rows: `uid, gold_idx, pred_idx, p_vector, conf_optsoftmax, entropy, verbalized_conf, selfconsistency, mean_logprob, dataset, modality, corruption, severity`.
- [ ] `src/confidence.py`: the alternative signals (§6.2) with robust parsers.
- [ ] `src/corruptions.py`: §8; wire into `infer.py` via flags.
- [ ] `scripts/20_infer_all.sh`: full eval grid (§9). Resume-safe.

### Phase 5 — Metrics + calibration + plots ☐
- [ ] `src/metrics.py`: §7 with bootstrap CIs. **Unit-test** ECE/AURC against hand-computed tiny examples.
- [ ] `src/calibrate.py`: temperature + Platt scaling, fit on `calib` split only.
- [ ] `scripts/30_compute_metrics.sh`: aggregate all parquets → `results/master_metrics.csv` (one row per model×condition×testset×severity, all metrics + CIs).
- [ ] `src/plots.py` + `scripts/40_make_figures.sh`: Figs 1–4 (§2) + reliability-diagram grid.

### Phase 6 — Analysis, novelty re-check, writeup ☐
- [ ] Produce `results/findings.md`: state whether each RQ's hypothesis held, with the key numbers.
- [ ] **Re-run the novelty check (§1).** Report any near-duplicate to the human before writing.
- [ ] Draft `paper/` per the outline in §15 (Markdown first; the human will move to LaTeX/Word).
- [ ] Final reproducibility pass: fresh clone + `README` steps reproduce one figure end-to-end.

---

## 14. Compute budget (estimate; refine after Phase 3 smoke)

- LoRA/QLoRA run on a few-k-item dataset: ~1–4 GPU-h on one 40–80 GB GPU.
- 18 adapters → ~40–70 GPU-h. Inference grid (incl. corruptions, self-consistency sampling) → ~20–40 GPU-h.
- **Total ≈ 60–120 single-GPU hours.** Fits a single A100/H100 over a few days, or parallelize across GPUs. If budget is tight, drop SmolVLM corruption sweep and self-consistency (N=10 is the most expensive signal).

---

## 15. Paper outline (map each section to a figure/table)

1. **Intro** — clinical VLMs are being fine-tuned everywhere, but deployment needs *trustworthy confidence*, not just accuracy. Gap: nobody has systematically characterized calibration/selective-prediction of PEFT-fine-tuned medical VLMs across modalities.
2. **Related work** — medical VLM fine-tuning (saturated accuracy studies); hallucination/abstention method papers (distinct: we measure, not mitigate); calibration of LLMs/VLMs (general-domain).
3. **Methods** — datasets (§4), models (§5), PEFT (§12), confidence signals (§6), metrics (§7), corruptions (§8).
4. **Results** —
   - 4.1 RQ1 calibration before/after FT (Fig. 1).
   - 4.2 RQ2 selective prediction + signal comparison (Fig. 2, Table).
   - 4.3 RQ3 temperature scaling (Table).
   - 4.4 RQ4 distribution shift (Fig. 3).
   - 4.5 RQ5 per-modality (Fig. 4).
5. **Discussion** — what this means for clinical deployment; cheap fixes that work; which signal to trust.
6. **Limitations** — closed/MC scope, open-weight only, dataset contamination risk, no human reader study.
7. **Conclusion.**

---

## 16. Gotchas / do-not-screw-up list

- **Chat templates differ per model** — wrong image-token placement silently tanks accuracy. Validate zero-shot accuracy > chance per model in Phase 2 before trusting anything.
- **Option scoring length bias** — always report length-normalized as primary; keep raw as ablation.
- **Calibration leakage** — temperature `T` is fit on `calib` only, applied to `test`. Never touch `test` labels during fitting.
- **Contamination** — PMC-derived datasets may overlap medical-VLM pretraining; do not over-claim; flag in Limitations.
- **Determinism** — fix seeds, but log that sampling-based signals (self-consistency) are inherently stochastic; fix the sampling seed too.
- **Save per-item raw outputs** — so any metric can be recomputed without re-running the GPU jobs.
- **Resume-safety** — every long script must skip already-completed (model, condition, testset) cells.
- **Gated model** — if `medgemma` can't be accessed, fall back to LLaVA-Med and record the swap; do not silently drop a medical model.

---

## 17. Definition of done

- `results/master_metrics.csv` populated for the full §9 grid with CIs.
- Figs 1–4 + reliability grid rendered in `figures/`.
- `results/findings.md` answers RQ1–RQ5 with numbers.
- Novelty re-check done and logged.
- Fresh-clone reproduction of at least one figure verified.
- Paper draft in `paper/` following §15.
