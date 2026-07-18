#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/syh/A-mem-aaai}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/nonmedical_timeline_formal_20260719_recovery_v7}"
LOG_ROOT="$RUN_ROOT/logs"
SUMMARY="$RUN_ROOT/queue_status.tsv"
LETTA_PY="/home/ymu/miniconda3/envs/letta-official/bin/python"
GPU0_ENDPOINT="http://127.0.0.1:8002/v1"
GPU1_ENDPOINT="http://127.0.0.1:8001/v1"
SEED=20260716

DIALSIM_RESUME="$ROOT/runs/nonmedical_timeline_formal_20260718_recovery_v6/letta/letta_dialsim_20260718_120056_629840_pid1255217"
RHELM_RESUME="$ROOT/runs/nonmedical_timeline_formal_20260717_recovery_v5/letta/letta_rhelm_20260718_024615_524573_pid1077523"
LONGMEM_RESUME="$ROOT/runs/nonmedical_timeline_formal_20260717_recovery_v5/letta/letta_longmemeval_20260718_044426_023328_pid1115417"

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
  local resume="$3"
  local start end rc
  local args
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$start] START letta_$dataset endpoint=$endpoint" >> "$LOG_ROOT/queue.log"
  args=(--dataset "$dataset" --workers 8 --endpoint "$endpoint"
        --output-root "$RUN_ROOT/letta" --require-api --judge-answers --qa-max-steps 3
        --random-seed "$SEED" --resume-predictions "$resume"
        --resume-allowed-endpoint "$GPU1_ENDPOINT")
  case "$dataset" in
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
  return "$rc"
}

run_dataset dialsim "$GPU1_ENDPOINT" "$DIALSIM_RESUME" &
dialsim_pid=$!
run_dataset rhelm "$GPU0_ENDPOINT" "$RHELM_RESUME" &
rhelm_pid=$!

longmem_endpoint=""
while [ -z "$longmem_endpoint" ]; do
  if ! kill -0 "$dialsim_pid" 2>/dev/null; then
    wait "$dialsim_pid" || true
    longmem_endpoint="$GPU1_ENDPOINT"
  elif ! kill -0 "$rhelm_pid" 2>/dev/null; then
    wait "$rhelm_pid" || true
    longmem_endpoint="$GPU0_ENDPOINT"
  else
    sleep 10
  fi
done

run_dataset longmemeval "$longmem_endpoint" "$LONGMEM_RESUME" &
longmem_pid=$!

wait "$dialsim_pid" 2>/dev/null || true
wait "$rhelm_pid" 2>/dev/null || true
wait "$longmem_pid" || true
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/QUEUE_COMPLETE"
