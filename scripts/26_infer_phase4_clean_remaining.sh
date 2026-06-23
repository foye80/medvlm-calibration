#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="all"
case "${1:-}" in
  --zero-shot-only)
    mode="zero_shot"
    shift
    ;;
  --cross-only)
    mode="cross"
    shift
    ;;
esac

if [[ "$#" -eq 0 ]]; then
  set -- qwen25vl internvl llavaov smolvlm medgemma huatuo
fi

train_datasets=(vqa_rad slake_en pathvqa)
eval_datasets=(vqa_rad slake_en pathvqa)

cuda_visible="${CUDA_VISIBLE_DEVICES:-}"
visible_gpu="${cuda_visible%%,*}"
max_busy_memory_mb="${PHASE4_MAX_BUSY_MEMORY_MB:-6000}"

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

run_infer_cell() {
  local model="$1"
  local train_dataset="$2"
  local eval_dataset="$3"
  local condition="$4"
  local adapter_path="$5"
  local output_csv="$6"

  if [[ -f "${output_csv}" ]]; then
    echo "$(date -Is) skip output exists output_csv=${output_csv}"
    return
  fi

  wait_for_gpu_capacity
  mkdir -p .run_locks
  local lock_path
  lock_path=".run_locks/infer_${model}_${condition}_on_${eval_dataset}_test.lock"
  echo "$(date -Is) infer model=${model} train_dataset=${train_dataset} eval_dataset=${eval_dataset} condition=${condition}"
  if [[ -n "${adapter_path}" ]]; then
    flock "${lock_path}" scripts/20_infer_all.sh \
      --model "${model}" \
      --dataset "${eval_dataset}" \
      --split test \
      --condition "${condition}" \
      --adapter-path "${adapter_path}" \
      --output-csv "${output_csv}" \
      --limit 0
  else
    flock "${lock_path}" scripts/20_infer_all.sh \
      --model "${model}" \
      --dataset "${eval_dataset}" \
      --split test \
      --condition "${condition}" \
      --output-csv "${output_csv}" \
      --limit 0
  fi
  echo "$(date -Is) done model=${model} train_dataset=${train_dataset} eval_dataset=${eval_dataset} condition=${condition}"
}

for model in "$@"; do
  if [[ "${mode}" == "all" || "${mode}" == "zero_shot" ]]; then
    for eval_dataset in "${eval_datasets[@]}"; do
      run_infer_cell \
        "${model}" \
        "zero_shot" \
        "${eval_dataset}" \
        "zero_shot" \
        "" \
        "results/phase4_${model}_zero_shot_on_${eval_dataset}_test.csv"
    done
  fi

  if [[ "${mode}" == "all" || "${mode}" == "cross" ]]; then
    for train_dataset in "${train_datasets[@]}"; do
      adapter_dir="adapters/${model}_${train_dataset}_lora"
      adapter_summary="${adapter_dir}/training_summary.json"
      echo "$(date -Is) waiting for ${adapter_summary}"
      while [[ ! -f "${adapter_summary}" ]]; do
        sleep 300
      done
      for eval_dataset in "${eval_datasets[@]}"; do
        if [[ "${eval_dataset}" == "${train_dataset}" ]]; then
          continue
        fi
        run_infer_cell \
          "${model}" \
          "${train_dataset}" \
          "${eval_dataset}" \
          "ft_${train_dataset}" \
          "${adapter_dir}" \
          "results/phase4_${model}_${train_dataset}_on_${eval_dataset}_test.csv"
      done
    done
  fi
done
