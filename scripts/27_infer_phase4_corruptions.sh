#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

train_datasets=(vqa_rad slake_en pathvqa)

# Optional: --train-datasets ds1,ds2 to restrict which adapters to run
if [[ "${1:-}" == "--train-datasets" ]]; then
  IFS=',' read -ra train_datasets <<< "$2"
  shift 2
fi

if [[ "$#" -eq 0 ]]; then
  set -- qwen25vl internvl llavaov smolvlm medgemma huatuo
fi
corruptions=(
  gaussian_noise
  gaussian_blur
  motion_blur
  brightness_shift
  contrast_shift
  jpeg_compression
  downscale_upscale
)
severities=(1 2 3)

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

for model in "$@"; do
  for train_dataset in "${train_datasets[@]}"; do
    adapter_dir="adapters/${model}_${train_dataset}_lora"
    adapter_summary="${adapter_dir}/training_summary.json"
    echo "$(date -Is) waiting for ${adapter_summary}"
    while [[ ! -f "${adapter_summary}" ]]; do
      sleep 300
    done

    for corruption in "${corruptions[@]}"; do
      for severity in "${severities[@]}"; do
        output_csv="results/phase4_${model}_${train_dataset}_on_${train_dataset}_test_${corruption}_s${severity}.csv"
        if [[ -f "${output_csv}" ]]; then
          echo "$(date -Is) skip output exists output_csv=${output_csv}"
          continue
        fi

        wait_for_gpu_capacity
        mkdir -p .run_locks
        lock_path=".run_locks/infer_${model}_${train_dataset}_on_${train_dataset}_test_${corruption}_s${severity}.lock"
        echo "$(date -Is) infer model=${model} train_dataset=${train_dataset} eval_dataset=${train_dataset} corruption=${corruption} severity=${severity}"
        flock "${lock_path}" scripts/20_infer_all.sh \
          --model "${model}" \
          --dataset "${train_dataset}" \
          --split test \
          --condition "ft_${train_dataset}" \
          --adapter-path "${adapter_dir}" \
          --corruption "${corruption}" \
          --severity "${severity}" \
          --output-csv "${output_csv}" \
          --limit 0
        echo "$(date -Is) done model=${model} train_dataset=${train_dataset} corruption=${corruption} severity=${severity}"
      done
    done
  done
done
