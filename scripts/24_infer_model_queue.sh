#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

model="${1:?usage: scripts/24_infer_model_queue.sh MODEL [TRAIN_DATASET...]}"
shift || true
if [[ "$#" -eq 0 ]]; then
  set -- vqa_rad slake_en pathvqa
fi

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

for train_dataset in "$@"; do
  adapter_dir="adapters/${model}_${train_dataset}_lora"
  adapter_summary="${adapter_dir}/training_summary.json"
  echo "$(date -Is) waiting for ${adapter_summary}"
  while [[ ! -f "$adapter_summary" ]]; do
    sleep 300
  done

  output_csv="results/phase4_${model}_${train_dataset}_on_${train_dataset}_test.csv"
  if [[ -f "$output_csv" ]]; then
    echo "$(date -Is) skip model=${model} train_dataset=${train_dataset}; output exists"
    continue
  fi

  wait_for_gpu_capacity
  mkdir -p .run_locks
  lock_path=".run_locks/infer_${model}_${train_dataset}_on_${train_dataset}_test.lock"
  echo "$(date -Is) infer model=${model} train_dataset=${train_dataset} eval_dataset=${train_dataset}"
  flock "${lock_path}" scripts/20_infer_all.sh \
    --model "$model" \
    --dataset "$train_dataset" \
    --split test \
    --condition "ft_${train_dataset}" \
    --adapter-path "$adapter_dir" \
    --output-csv "$output_csv" \
    --limit 0
  echo "$(date -Is) done infer model=${model} train_dataset=${train_dataset}"
done
