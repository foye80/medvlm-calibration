#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

restart=false
interval=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --restart)
      restart=true
      shift
      ;;
    --interval)
      interval="${2:?--interval requires seconds}"
      shift 2
      ;;
    *)
      echo "usage: scripts/12_finetune_watchdog.sh [--restart] [--interval seconds]" >&2
      exit 2
      ;;
  esac
done

models=(qwen25vl smolvlm llavaov internvl huatuo medgemma)
datasets=(vqa_rad slake_en pathvqa)

gpu_for_model() {
  case "$1" in
    qwen25vl) echo 6 ;;
    smolvlm) echo 1 ;;
    llavaov) echo 5 ;;
    internvl) echo 4 ;;
    huatuo|medgemma) echo 7 ;;
    *) echo "" ;;
  esac
}

queue_session_for_model() {
  case "$1" in
    qwen25vl) echo medvlm_phase3_qwen_queue ;;
    smolvlm) echo medvlm_phase3_smolvlm_queue ;;
    llavaov) echo medvlm_phase3_llavaov_queue ;;
    internvl) echo medvlm_phase3_internvl_queue ;;
    huatuo) echo medvlm_phase3_huatuo_queue ;;
    medgemma) echo medvlm_phase3_medgemma_queue ;;
    *) echo "" ;;
  esac
}

is_training_process_active() {
  local model="$1"
  local dataset="$2"
  pgrep -af "src[.]finetune --model ${model} --dataset ${dataset}" >/dev/null
}

is_queue_active() {
  local model="$1"
  local session
  session="$(queue_session_for_model "$model")"
  [[ -n "$session" ]] && tmux has-session -t "$session" 2>/dev/null
}

start_recovery() {
  local model="$1"
  local dataset="$2"
  local gpu="$3"
  local stamp
  local session
  local log
  stamp="$(date +%Y%m%d_%H%M%S)"
  session="medvlm_recover_${model}_${dataset}_${stamp}"
  log="logs/phase3_recover_${model}_${dataset}_${stamp}.log"
  echo "$(date -Is) restart model=${model} dataset=${dataset} gpu=${gpu} session=${session} log=${log}"
  tmux new-session -d -s "$session" \
    "CUDA_VISIBLE_DEVICES=${gpu} scripts/10_finetune_all.sh --model ${model} --dataset ${dataset} --max-image-edge 512 > ${log} 2>&1"
}

audit_once() {
  local any_issue=false
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      local summary="adapters/${model}_${dataset}_lora/training_summary.json"
      if [[ -f "$summary" ]]; then
        echo "$(date -Is) ok summary model=${model} dataset=${dataset}"
        continue
      fi

      if [[ "$model" == "medgemma" && ! -f adapters/huatuo_pathvqa_lora/training_summary.json ]]; then
        echo "$(date -Is) wait dependency model=medgemma dataset=${dataset} dependency=huatuo_pathvqa"
        continue
      fi

      if is_training_process_active "$model" "$dataset"; then
        echo "$(date -Is) ok active model=${model} dataset=${dataset}"
        continue
      fi

      if [[ "$model" == "medgemma" ]] && is_queue_active "$model"; then
        echo "$(date -Is) ok queued model=${model} dataset=${dataset}"
        continue
      fi

      if [[ "$dataset" != "vqa_rad" ]] && is_queue_active "$model"; then
        echo "$(date -Is) ok queued model=${model} dataset=${dataset}"
        continue
      fi

      any_issue=true
      echo "$(date -Is) missing model=${model} dataset=${dataset} summary=${summary}"
      if [[ "$restart" == true ]]; then
        local gpu
        gpu="$(gpu_for_model "$model")"
        start_recovery "$model" "$dataset" "$gpu"
      fi
    done
  done
  if [[ "$any_issue" == false ]]; then
    echo "$(date -Is) audit clean"
  fi
}

if [[ "$interval" -gt 0 ]]; then
  while true; do
    audit_once
    sleep "$interval"
  done
else
  audit_once
fi
