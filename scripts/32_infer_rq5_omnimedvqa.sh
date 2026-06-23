#!/usr/bin/env bash
# RQ5 inference: 6 models × {zero_shot, ft_vqa_rad} × OmniMedVQA
# Prerequisite: run scripts/31_prepare_omnimedvqa_full.py first.
# Usage: CUDA_VISIBLE_DEVICES=X bash scripts/32_infer_rq5_omnimedvqa.sh [model ...]
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$#" -eq 0 ]]; then
  set -- qwen25vl internvl llavaov smolvlm medgemma huatuo
fi

RECORDS_CSV="data/omnimedvqa_items.csv"
MAX_IMAGE_EDGE="${RQ5_MAX_IMAGE_EDGE:-1120}"
if [[ ! -f "${RECORDS_CSV}" ]]; then
  echo "ERROR: ${RECORDS_CSV} not found. Run scripts/31_prepare_omnimedvqa_full.py first."
  exit 1
fi

cuda_visible="${CUDA_VISIBLE_DEVICES:-}"
visible_gpu="${cuda_visible%%,*}"
max_busy_memory_mb="${PHASE4_MAX_BUSY_MEMORY_MB:-6000}"

wait_for_gpu() {
  [[ -z "${cuda_visible}" || -z "${visible_gpu}" ]] && return
  command -v nvidia-smi >/dev/null 2>&1 || return
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="${visible_gpu}" 2>/dev/null | head -1 | tr -d ' ')"
    [[ -z "${used}" || "${used}" -le "${max_busy_memory_mb}" ]] && break
    echo "$(date -Is) waiting GPU${visible_gpu} used=${used}MB threshold=${max_busy_memory_mb}MB"
    sleep 300
  done
}

run_cell() {
  local model="$1" condition="$2" adapter_path="$3" output_csv="$4"
  if [[ -f "${output_csv}" ]]; then
    echo "$(date -Is) skip exists output_csv=${output_csv}"
    return
  fi
  wait_for_gpu
  echo "$(date -Is) infer model=${model} condition=${condition}"
  if [[ -n "${adapter_path}" ]]; then
    scripts/20_infer_all.sh \
      --model "${model}" \
      --dataset omnimedvqa \
      --records-csv "${RECORDS_CSV}" \
      --split test \
      --condition "${condition}" \
      --adapter-path "${adapter_path}" \
      --output-csv "${output_csv}" \
      --max-image-edge "${MAX_IMAGE_EDGE}" \
      --limit 0
  else
    scripts/20_infer_all.sh \
      --model "${model}" \
      --dataset omnimedvqa \
      --records-csv "${RECORDS_CSV}" \
      --split test \
      --condition "${condition}" \
      --output-csv "${output_csv}" \
      --max-image-edge "${MAX_IMAGE_EDGE}" \
      --limit 0
  fi
  echo "$(date -Is) done model=${model} condition=${condition} output=${output_csv}"
}

for model in "$@"; do
  # zero_shot
  run_cell "${model}" "zero_shot" "" \
    "results/rq5_${model}_zero_shot_omnimedvqa_test.csv"

  # ft_vqa_rad (vqa_rad adapter — broadest medical training set)
  adapter_dir="adapters/${model}_vqa_rad_lora"
  if [[ -f "${adapter_dir}/training_summary.json" ]]; then
    run_cell "${model}" "ft_vqa_rad" "${adapter_dir}" \
      "results/rq5_${model}_ft_vqa_rad_omnimedvqa_test.csv"
  else
    echo "$(date -Is) WARN adapter missing skip ft_vqa_rad model=${model}"
  fi
done
echo "$(date -Is) RQ5 inference complete"
