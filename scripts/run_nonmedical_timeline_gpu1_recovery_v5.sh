#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/syh/A-mem-aaai}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/nonmedical_timeline_formal_20260717_recovery_v5}"
LOG_ROOT="$RUN_ROOT/logs"
SUMMARY="$RUN_ROOT/queue_status.tsv"
CORE_PY="/root/miniconda3/envs/qwen3/bin/python"
LETTA_PY="/home/ymu/miniconda3/envs/letta-official/bin/python"
ENDPOINT="http://127.0.0.1:8001/v1"
SEED=20260716

export PYTHONPATH="$ROOT"
export DEEPSEEK_BASE_URL="$ENDPOINT"
export DEEPSEEK_API_KEY="EMPTY"
export DEEPSEEK_MODEL="qwen3-vl-8b"
export DEEPSEEK_TIMEOUT="240"
export DEEPSEEK_MAX_TOKENS="700"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOG_ROOT"
printf 'step\tstart_utc\tend_utc\texit_code\n' > "$SUMMARY"

run_step() {
  local name="$1"
  shift
  local start end rc
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$start] START $name" | tee -a "$LOG_ROOT/queue.log"
  "$@" >"$LOG_ROOT/$name.log" 2>&1
  rc=$?
  end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\n' "$name" "$start" "$end" "$rc" >> "$SUMMARY"
  echo "[$end] END $name rc=$rc" | tee -a "$LOG_ROOT/queue.log"
  return 0
}

core_methods="direct,static_rag,amem,ddo,gmemory,meminsight,memoryos,medimem"
run_step core_rhelm \
  "$CORE_PY" -m mem_ehr_agent.cli benchmark run \
  --dataset rhelm --methods "$core_methods" \
  --dataset-path "$ROOT/datasets/amem_original/rhelm/data" \
  --max-workers 8 --output-root "$RUN_ROOT/core" --require-api \
  --locomo-top-k 32 --locomo-coarse-k 32 --judge-answers

for dataset in dialsim rhelm longmemeval; do
  workers=4
  if [ "$dataset" = rhelm ]; then
    workers=1
  fi
  args=(--dataset "$dataset" --workers "$workers" --endpoint "$ENDPOINT"
        --output-root "$RUN_ROOT/letta" --require-api --judge-answers --qa-max-steps 3
        --random-seed "$SEED")
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
  run_step "letta_$dataset" "$LETTA_PY" "$ROOT/scripts/run_official_letta_timeline.py" "${args[@]}"
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/QUEUE_COMPLETE"
