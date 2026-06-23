# Novelty Check — Phase 6 Pre-Writing

**Date:** 2026-06-08  
**Required by:** PROJECT_SPEC.md §1 ("re-run a quick literature check before writing")  
**Verdict:** No near-duplicate found. Original framing stands. Two papers require explicit differentiation in Related Work and Discussion.

---

## Search queries run

1. "calibration fine-tuned medical vision language model PEFT LoRA 2024 2025 2026"
2. "selective prediction abstention medical VQA confidence calibration 2024 2025"
3. "confidence calibration medical VLM fine-tuning overconfidence ECE 2024 2025 2026"
4. "LoRA PEFT calibration before after fine-tuning medical VLM VQA ECE reliability 2025 2026"
5. "MICCAI 2025 confidence calibration multimodal medical VLM empirical fine-tuning"

---

## Papers reviewed

### 1. Byun et al. (2026) — CLOSEST RELATED WORK, must cite and differentiate

**Citation:** Ji Young Byun, Young-Jin Park, Jean-Philippe Corbeil, Asma Ben Abacha. "Overconfidence and Calibration in Medical VQA: Empirical Findings and Hallucination-Aware Mitigation." *arXiv:2604.02543*, April 2026.

**What they do:**
- Empirical study of zero-shot calibration in medical VLMs
- Models: Qwen3-VL, InternVL3, LLaVA-NeXT across scales 2B–38B
- Datasets: VQA-RAD, SLAKE-EN, VQA-Med (3 benchmarks)
- Confidence signals: chain-of-thought, verbalized confidence variants
- Post-hoc calibration: Platt scaling
- Novel contribution: Hallucination-Aware Calibration (HAC) — uses vision-grounded hallucination detection signals as auxiliary input to refine confidence estimates

**What they do NOT do (= our contribution):**

| Dimension | Byun et al. | Ours |
|---|---|---|
| PEFT/LoRA fine-tuning effect on calibration | ❌ zero-shot only | ✅ core RQ1 |
| Before-after calibration comparison | ❌ | ✅ 6 models × 3 datasets × 72-cell grid |
| Selective prediction / risk–coverage curves | ❌ | ✅ RQ2 |
| Image corruption degradation of calibration | ❌ | ✅ RQ4b, 7 types × 3 severities |
| Temperature scaling as post-hoc calibration | ❌ (uses Platt+HAC instead) | ✅ RQ3 |
| Training collapse as safety risk documentation | ❌ | ✅ (huatuo/pathvqa gradient explosion case) |
| Proposes new calibration method | ✅ HAC | ❌ intentionally not — we are empirical |

**How to differentiate in Discussion:**

> Byun et al. (2026) establish that zero-shot medical VLMs are systematically overconfident and propose HAC to mitigate this at inference time. Our work addresses a complementary and clinically prior question: what does LoRA fine-tuning — the standard adaptation step before deployment — do to calibration reliability? We show this effect is heterogeneous and non-monotonic: fine-tuning improves accuracy across the board but changes calibration in directions that cannot be predicted from accuracy alone, including cases of "accuracy-neutral reliability degradation" and outright training collapse. This means the overconfidence problem documented by Byun et al. for zero-shot models can be *exacerbated or introduced* by fine-tuning, motivating calibration evaluation as a mandatory post-fine-tuning audit step rather than a zero-shot diagnostic.

---

### 2. MICCAI 2025 Paper 1840 — METHOD PAPER, different track

**Citation:** (Authors TBC from camera-ready.) "Confidence Calibration for Multimodal LLMs: An Empirical Study Through Medical VQA." *MICCAI 2025*, Paper 1840.

**What they do:**
- Proposes MS-FBI (Multi-Strategy Fusion-Based Interrogation) + auxiliary expert LLM
- Reduces ECE by average 40% across 3 medical VQA datasets
- Zero-shot / prompting focused; no PEFT fine-tuning

**Key difference:** Method-proposal paper. Our study is purely empirical — we measure what happens, not propose a new fix. Cite as prior calibration method work; our temperature-scaling RQ3 is a simpler, cheaper baseline that we deliberately test first.

---

### 3. ScienceDirect LoRA/AdaLoRA comparison (2025) — NO THREAT

"Optimizing multimodal models for medical VQA: A comparative study of LoRA and AdaLoRA on VQA-RAD and SLAKE-VQA." *Computers in Biology and Medicine*, 2025.

Reports only accuracy. No ECE, no confidence reliability, no selective prediction. Cite as evidence that PEFT for medical VQA is mainstream; we add the reliability dimension they omit.

---

### 4. Variational VQA (arXiv 2505.09591) — NO THREAT

"Variational Visual Question Answering for Uncertainty-Aware Selective Prediction." *TMLR*, 2026. General (non-medical) VQA, proposes variational learning for selective prediction. No PEFT, no medical domain. Cite as selective prediction methodology background.

---

## Literature gap confirmed

No existing paper combines:
1. LoRA/PEFT fine-tuning **before-after calibration comparison** in medical VLMs
2. Systematic **selective prediction / risk–coverage** analysis post fine-tuning
3. **Image corruption** degradation of calibration post fine-tuning
4. **Temperature scaling** as low-cost recovery of post-fine-tuning calibration
5. Documentation of **training instability** (collapse to single-option prediction) as a calibration safety risk

This combination is our contribution. The framing "fine-tuning and calibration reliability are not aligned" is not addressed in the literature.

---

## Discussion section writing notes

When writing the Discussion, explicitly compare against Byun et al. (2026) on these axes:

1. **Complementary findings on overconfidence:** Both studies find overconfidence, but ours shows fine-tuning *modulates* it non-monotonically (some models worse, some better) while theirs shows it persists at zero-shot across scales. Together: overconfidence is present before fine-tuning AND can be made worse by it.

2. **On post-hoc calibration:** They use Platt + HAC; we use temperature scaling. If our RQ3 shows temperature scaling is insufficient, this supports their argument that more sophisticated methods (e.g., HAC) are needed even for fine-tuned models.

3. **On selective prediction (RQ2):** If our AUROC results show that post-fine-tuning confidence signals still separate correct from incorrect at AUROC ~0.72 on average, this is actionable for deployment (defer low-confidence items) even when ECE is poor — a nuance Byun et al. do not address.

4. **On distribution shift / corruptions (RQ4):** Byun et al. evaluate in-distribution only. Our RQ4 adds the out-of-distribution story: fine-tuned models that appear well-calibrated in-distribution may degrade in both accuracy and calibration under corruption, sometimes remaining confident while becoming wrong — the highest clinical risk scenario.

---

## Action required before submission

- [ ] Obtain full author list and journal/conference details for MICCAI 2025 Paper 1840 (camera-ready not yet indexed on arXiv as of 2026-06-08).
- [ ] At camera-ready stage, re-check arXiv for any new preprints matching "PEFT calibration medical VQA" — the field is moving fast (Byun et al. appeared only 2 months ago).
