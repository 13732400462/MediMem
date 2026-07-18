#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/syh/A-mem-aaai}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/nonmedical_timeline_formal_20260719_extractive_v8}"
LOG_ROOT="$RUN_ROOT/logs"
SUMMARY="$RUN_ROOT/queue_status.tsv"
LETTA_PY="/home/ymu/miniconda3/envs/letta-official/bin/python"
GPU0_ENDPOINT="http://127.0.0.1:8002/v1"
GPU1_ENDPOINT="http://127.0.0.1:8001/v1"
SEED=20260716

export PYTHONPATH="$ROOT"
export DEEPSEEK_API_KEY="EMPTY"
export DEEPSEEK_MODEL="qwen3-vl-8b"
export DEEPSEEK_TIMEOUT="240"
export DEEPSEEK_MAX_TOKENS="700"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOG_ROOT"
printf 'step\tendpoint\tstart_utc\tend_utc\texit_code\n' > "$SUMMARY"

run_dataset() {
  local dataset="$1"
  local endpoint="$2"
  local start end rc
  local args
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$start] START letta_$dataset endpoint=$endpoint" >> "$LOG_ROOT/queue.log"
  args=(--dataset "$dataset" --workers 8 --endpoint "$endpoint"
        --output-root "$RUN_ROOT/letta" --require-api --judge-answers --qa-max-steps 3
        --random-seed "$SEED" --ingestion-mode extractive)
  case "$dataset" in
    locomo)
      args+=(--dataset-path "$ROOT/datasets/amem_original/locomo/locomo10.official.json"
             --sample-manifest "$ROOT/data/processed/locomo_dev_selection_20260715/split_manifest.json") ;;
    dialsim)
      args+=(--dataset-path "$ROOT/datasets/amem_original/dialsim" --sample-n 1000) ;;
    rhelm)
      args+=(--dataset-path "$ROOT/datasets/amem_original/rhelm/data") ;;
    longmemeval)
      args+=(--dataset-path "$ROOT/datasets/amem_original/longmemeval/longmemeval_s_cleaned.json") ;;
  esac
  DEEPSEEK_BASE_URL="$endpoint" "$LETTA_PY" "$ROOT/scripts/run_official_letta_timeline.py" "${args[@]}" \
    >"$LOG_ROOT/letta_$dataset.log" 2>&1
  rc=$?
  end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\n' "letta_$dataset" "$endpoint" "$start" "$end" "$rc" >> "$SUMMARY"
  echo "[$end] END letta_$dataset endpoint=$endpoint rc=$rc" >> "$LOG_ROOT/queue.log"
  return 0
}

(
  run_dataset locomo "$GPU1_ENDPOINT"
  run_dataset dialsim "$GPU1_ENDPOINT"
) &
gpu1_queue_pid=$!

(
  run_dataset rhelm "$GPU0_ENDPOINT"
  run_dataset longmemeval "$GPU0_ENDPOINT"
) &
gpu0_queue_pid=$!

wait "$gpu0_queue_pid"
wait "$gpu1_queue_pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/QUEUE_COMPLETE"
