#!/usr/bin/env bash
# Thin wrapper around src.verbalized (mirrors scripts/20_infer_all.sh env setup).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/.phase2_pydeps:${PWD}/.phase1_deps:${PWD}/.test_deps:${PWD}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
python_bin="${PYTHON_BIN:-python3}"
"${python_bin}" -m src.verbalized "$@"
