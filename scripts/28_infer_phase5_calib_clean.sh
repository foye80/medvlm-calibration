#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$#" -eq 0 ]]; then
  set -- qwen25vl internvl llavaov smolvlm medgemma huatuo
fi

train_datasets=(vqa_rad slake_en pathvqa)
eval_datasets=(vqa_rad slake_en pathvqa)

cuda_visible="${CUDA_VISIBLE_DEVICES:-}"
visible_gpu="${cuda_visible%%,*}"
max_busy_memory_mb="${PHASE5_MAX_BUSY_MEMORY_MB:-6000}"

wait_for_gpu_capacity() {
  if [[ -z "${cuda_visible}" || -z "${visible_gpu}" ]]; then
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  while true; do
    local used_mb
    used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="${visible_gpu}" 2>/dev/null | head -n 1 | tr -d ' ')"
    if [[ -z "${used_mb}" || "${used_mb}" -le "${max_busy_memory_mb}" ]]; then
      break
    fi
    echo "$(date -Is) waiting for GPU${visible_gpu} capacity used_mb=${used_mb} threshold_mb=${max_busy_memory_mb}"
    sleep 300
  done
}

run_calib_cell() {
  local model="$1"
  local condition_name="$2"
  local condition_arg="$3"
  local eval_dataset="$4"
  local adapter_path="$5"
  local output_csv="$6"

  if [[ -f "${output_csv}" ]]; then
    echo "$(date -Is) skip output exists output_csv=${output_csv}"
    return
  fi

  wait_for_gpu_capacity
  mkdir -p .run_locks
  local lock_path
  lock_path=".run_locks/infer_calib_${model}_${condition_name}_on_${eval_dataset}.lock"
  echo "$(date -Is) infer calib model=${model} condition=${condition_name} eval_dataset=${eval_dataset}"
  if [[ -n "${adapter_path}" ]]; then
    flock "${lock_path}" scripts/20_infer_all.sh \
      --model "${model}" \
      --dataset "${eval_dataset}" \
      --split calib \
      --condition "${condition_arg}" \
      --adapter-path "${adapter_path}" \
      --output-csv "${output_csv}" \
      --limit 0
  else
    flock "${lock_path}" scripts/20_infer_all.sh \
      --model "${model}" \
      --dataset "${eval_dataset}" \
      --split calib \
      --condition "${condition_arg}" \
      --output-csv "${output_csv}" \
      --limit 0
  fi
  echo "$(date -Is) done calib model=${model} condition=${condition_name} eval_dataset=${eval_dataset}"
}

for model in "$@"; do
  for eval_dataset in "${eval_datasets[@]}"; do
    run_calib_cell \
      "${model}" \
      "zero_shot" \
      "zero_shot" \
      "${eval_dataset}" \
      "" \
      "results/phase5_calib_${model}_zero_shot_on_${eval_dataset}_calib.csv"
  done

  for train_dataset in "${train_datasets[@]}"; do
    adapter_dir="adapters/${model}_${train_dataset}_lora"
    adapter_summary="${adapter_dir}/training_summary.json"
    echo "$(date -Is) waiting for ${adapter_summary}"
    while [[ ! -f "${adapter_summary}" ]]; do
      sleep 300
    done

    for eval_dataset in "${eval_datasets[@]}"; do
      run_calib_cell \
        "${model}" \
        "${train_dataset}" \
        "ft_${train_dataset}" \
        "${eval_dataset}" \
        "${adapter_dir}" \
        "results/phase5_calib_${model}_${train_dataset}_on_${eval_dataset}_calib.csv"
    done
  done
done
