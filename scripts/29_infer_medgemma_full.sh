#!/usr/bin/env bash
# Run all 12 medgemma clean-grid cells (zero-shot + 3 FT adapters × 3 eval datasets).
# Waits for adapters that are still training.
# Usage: CUDA_VISIBLE_DEVICES=4 bash scripts/29_infer_medgemma_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}/.phase2_pydeps:${PWD}/.phase1_deps:${PWD}/.test_deps:${PWD}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cuda_visible="${CUDA_VISIBLE_DEVICES:-}"
visible_gpu="${cuda_visible%%,*}"
max_busy_memory_mb="${PHASE4_MAX_BUSY_MEMORY_MB:-6000}"
LOG="logs/infer_medgemma_full.log"
mkdir -p logs

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

log "=== 29_infer_medgemma_full started (GPU=${CUDA_VISIBLE_DEVICES:-unset}) ==="

wait_for_gpu_capacity() {
  [[ -z "${cuda_visible}" || -z "${visible_gpu}" ]] && return
  command -v nvidia-smi >/dev/null 2>&1 || return
  while true; do
    local used_mb
    used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="${visible_gpu}" 2>/dev/null | head -n 1 | tr -d ' ')"
    [[ -z "${used_mb}" || "${used_mb}" -le "${max_busy_memory_mb}" ]] && break
    log "waiting for GPU${visible_gpu} capacity used_mb=${used_mb} threshold=${max_busy_memory_mb}"
    sleep 300
  done
}

run_cell() {
  local condition="$1"
  local eval_dataset="$2"
  local output_csv="$3"
  local adapter_path="${4:-}"

  if [[ -f "${output_csv}" ]]; then
    log "skip exists: ${output_csv}"
    return
  fi
  # Remove stale .tmp if present
  [[ -f "${output_csv}.tmp" ]] && rm -f "${output_csv}.tmp"

  wait_for_gpu_capacity
  mkdir -p .run_locks
  local lock=".run_locks/infer_medgemma_${condition}_on_${eval_dataset}_test.lock"
  log "infer medgemma condition=${condition} eval=${eval_dataset}"
  if [[ -n "${adapter_path}" ]]; then
    flock "${lock}" scripts/20_infer_all.sh \
      --model medgemma \
      --dataset "${eval_dataset}" \
      --split test \
      --condition "${condition}" \
      --adapter-path "${adapter_path}" \
      --output-csv "${output_csv}" \
      --limit 0 2>&1 | tee -a "$LOG"
  else
    flock "${lock}" scripts/20_infer_all.sh \
      --model medgemma \
      --dataset "${eval_dataset}" \
      --split test \
      --condition "${condition}" \
      --output-csv "${output_csv}" \
      --limit 0 2>&1 | tee -a "$LOG"
  fi
  log "done: ${output_csv}"
}

eval_datasets=(vqa_rad slake_en pathvqa)

# ── Zero-shot (3 cells) ──────────────────────────────────────────────────────
for ev in "${eval_datasets[@]}"; do
  run_cell "zero_shot" "${ev}" "results/phase4_medgemma_zero_shot_on_${ev}_test.csv"
done

# ── FT adapters (3 × 3 = 9 cells, waits for training_summary.json) ──────────
for train_ds in vqa_rad slake_en pathvqa; do
  adapter_dir="adapters/medgemma_${train_ds}_lora"
  log "waiting for ${adapter_dir}/training_summary.json"
  until [[ -f "${adapter_dir}/training_summary.json" ]]; do sleep 60; done
  log "adapter ready: ${adapter_dir}"
  for ev in "${eval_datasets[@]}"; do
    run_cell "ft_${train_ds}" "${ev}" \
      "results/phase4_medgemma_${train_ds}_on_${ev}_test.csv" \
      "${adapter_dir}"
  done
done

# ── Re-aggregate master_metrics.csv ─────────────────────────────────────────
log "re-running phase5_aggregate.py"
${PYTHON_BIN:-python3} scripts/phase5_aggregate.py \
  --results-dir results \
  --out results/master_metrics.csv 2>&1 | tee -a "$LOG"

log "=== ALL DONE ==="
