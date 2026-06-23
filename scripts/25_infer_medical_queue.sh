#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

scripts/24_infer_model_queue.sh huatuo &
huatuo_pid="$!"

scripts/24_infer_model_queue.sh medgemma &
medgemma_pid="$!"

wait "${huatuo_pid}"
wait "${medgemma_pid}"
