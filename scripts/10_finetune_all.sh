#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/.phase2_pydeps:${PWD}/.phase1_deps:${PWD}/.test_deps:${PWD}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python_bin="${PYTHON_BIN:-python3}"

model="qwen25vl"
dataset="vqa_rad"
args=("$@")
idx=0
while [[ "$idx" -lt "${#args[@]}" ]]; do
  case "${args[$idx]}" in
    --model)
      idx=$((idx + 1))
      model="${args[$idx]:-${model}}"
      ;;
    --dataset)
      idx=$((idx + 1))
      dataset="${args[$idx]:-${dataset}}"
      ;;
  esac
  idx=$((idx + 1))
done

mkdir -p .run_locks
lock_path=".run_locks/finetune_${model}_${dataset}.lock"
echo "$(date -Is) waiting for finetune lock model=${model} dataset=${dataset} lock=${lock_path}"
flock "$lock_path" "${python_bin}" -m src.finetune "$@"
