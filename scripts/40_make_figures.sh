#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/.phase1_deps:${PWD}/.test_deps:${PWD}:${PYTHONPATH:-}"
python3 -m src.plots "$@"
