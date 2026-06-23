#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

wait_for=""
if [[ "${1:-}" == "--wait-for" ]]; then
  wait_for="$2"
  shift 2
fi

model="${1:?usage: scripts/11_finetune_model_queue.sh [--wait-for path] MODEL DATASET...}"
shift
if [[ "$#" -eq 0 ]]; then
  set -- vqa_rad slake_en pathvqa
fi

if [[ -n "$wait_for" ]]; then
  echo "$(date -Is) waiting for ${wait_for}"
  while [[ ! -f "$wait_for" ]]; do
    sleep 300
  done
fi

for dataset in "$@"; do
  summary="adapters/${model}_${dataset}_lora/training_summary.json"
  if [[ -f "$summary" ]]; then
    echo "$(date -Is) skip model=${model} dataset=${dataset}; summary exists"
    continue
  fi
  echo "$(date -Is) start model=${model} dataset=${dataset}"
  scripts/10_finetune_all.sh --model "$model" --dataset "$dataset"
  echo "$(date -Is) done model=${model} dataset=${dataset}"
done
