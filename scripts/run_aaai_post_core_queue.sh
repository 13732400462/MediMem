#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/syh/A-mem-aaai}"
CORE_PID="${CORE_PID:?CORE_PID is required}"
MAX_WORKERS="${MAX_WORKERS:-16}"
export PATH="/root/miniconda3/envs/amem_eval/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

while kill -0 "$CORE_PID" 2>/dev/null; do
  sleep 60
done

test -s "$ROOT/runs/aaai_core_three_seed_20260706/statistical_analysis/stratified_metrics.csv"

ROOT="$ROOT" MAX_WORKERS="$MAX_WORKERS" bash "$ROOT/scripts/run_aaai_budget_sweep_three_seed.sh"
ROOT="$ROOT" MAX_WORKERS="$MAX_WORKERS" bash "$ROOT/scripts/run_aaai_transfer_three_seed.sh"
ROOT="$ROOT" MAX_WORKERS="$MAX_WORKERS" bash "$ROOT/scripts/run_aaai_contamination_three_seed.sh"
