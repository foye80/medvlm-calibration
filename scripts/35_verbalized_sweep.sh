#!/usr/bin/env bash
# Work-stealing sweep for verbalized confidence (RQ2 signal).
# 36-cell pool: 6 models × (zero_shot ×3 datasets + ft_id ×3 datasets), clean test.
# Coordinates via .run_locks (flock -n) so any number of GPUs drain the pool.
# CPU capped to 4 threads everywhere.
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/35_verbalized_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

models=(qwen25vl internvl llavaov smolvlm medgemma huatuo)
datasets=(vqa_rad slake_en pathvqa)

declare -a CELLS=()
for m in "${models[@]}"; do
  # zero_shot on all 3 datasets
  for ds in "${datasets[@]}"; do
    out="results/verbalized_${m}_zero_shot_on_${ds}_test.csv"
    CELLS+=("${out}|--model ${m} --dataset ${ds} --split test --condition zero_shot")
  done
  # ft on its own (ID) dataset
  for ds in "${datasets[@]}"; do
    adp="adapters/${m}_${ds}_lora"
    out="results/verbalized_${m}_ft_${ds}_on_${ds}_test.csv"
    CELLS+=("${out}|--model ${m} --dataset ${ds} --split test --condition ft_${ds} --adapter-path ${adp}")
  done
done

mkdir -p .run_locks results

run_cell() {
  local out="$1"; shift
  local lock=".run_locks/verbalized_$(basename "${out}").lock"
  (
    exec 9>"${lock}"
    flock -n 9 || exit 2
    [[ -f "${out}" ]] && exit 0
    # skip ft cells whose adapter is missing
    for a in "$@"; do :; done
    echo "$(date -Is) RUN ${out}"
    bash scripts/34_verbalized.sh "$@" --output-csv "${out}" --limit 0
    echo "$(date -Is) DONE ${out}"
  )
}

echo "$(date -Is) verbalized sweep start gpu=${CUDA_VISIBLE_DEVICES:-?} pool=${#CELLS[@]}"
while true; do
  remaining=0; progressed=0
  for entry in "${CELLS[@]}"; do
    out="${entry%%|*}"; args="${entry#*|}"
    [[ -f "${out}" ]] && continue
    # guard: ft cell needs its adapter dir
    if [[ "${args}" == *"--adapter-path "* ]]; then
      adp="${args##*--adapter-path }"; adp="${adp%% *}"
      if [[ ! -f "${adp}/training_summary.json" && ! -d "${adp}" ]]; then
        echo "$(date -Is) SKIP missing adapter ${adp} for ${out}"; continue
      fi
    fi
    remaining=1
    # shellcheck disable=SC2086
    run_cell "${out}" ${args}; rc=$?
    [[ ${rc} -eq 0 ]] && progressed=1
  done
  [[ ${remaining} -eq 0 ]] && break
  [[ ${progressed} -eq 0 ]] && { echo "$(date -Is) all remaining locked, sleep 30"; sleep 30; }
done
echo "$(date -Is) verbalized sweep done gpu=${CUDA_VISIBLE_DEVICES:-?} — pool drained"
