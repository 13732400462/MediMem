#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/home/syh/A-mem}"
PY="${PY:-/root/miniconda3/envs/llmserve/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/syh/mem_ehr_hf_cache}"
DATA_ROOT="${DATA_ROOT:-data/processed/fast_formal_8_sources_$(date +%Y%m%d)}"
RUN_ROOT="${RUN_ROOT:-runs/fast_formal_8_sources_$(date +%Y%m%d_%H%M%S)}"
PER_SOURCE_N="${PER_SOURCE_N:-100}"
LOCOMO_SAMPLE_N="${LOCOMO_SAMPLE_N:-100}"
RANDOM_SEED="${RANDOM_SEED:-20260606}"
MAX_WORKERS_PER_PORT="${MAX_WORKERS_PER_PORT:-32}"
COUNTERFACTUAL_SAMPLE_RATE="${COUNTERFACTUAL_SAMPLE_RATE:-0.20}"
COUNTERFACTUAL_RISK_THRESHOLD="${COUNTERFACTUAL_RISK_THRESHOLD:-0.55}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-30b-fp8}"
BASE_URL_A="${BASE_URL_A:-http://127.0.0.1:8000/v1}"
BASE_URL_B="${BASE_URL_B:-http://127.0.0.1:8001/v1}"
LOCOMO_DATASET_PATH="${LOCOMO_DATASET_PATH:-datasets/amem_original/locomo/locomo10.official.json}"
LOCOMO_METHODS="${LOCOMO_METHODS:-direct,amem,medimem}"

QUEUE_A_SOURCES=(medmcqa chatdoctor_healthcaremagic medical_meadow_wikidoc locomo)
QUEUE_B_SOURCES=(medqa pmoa_tts pmc_patients)

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=.
export MEM_EHR_DATA_CACHE_DIR="$CACHE_DIR"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-local-qwen3}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-$MODEL_NAME}"
export DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-600}"

mkdir -p "$RUN_ROOT" "$DATA_ROOT" "$CACHE_DIR"
STATUS_JSONL="$RUN_ROOT/source_status.jsonl"
BLOCKED_JSONL="$RUN_ROOT/blocked_sources.jsonl"
: > "$STATUS_JSONL"
: > "$BLOCKED_JSONL"
: > "$RUN_ROOT/run.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUN_ROOT/run.log"
}

append_json() {
  local path="$1"
  local payload="$2"
  "$PY" - "$path" "$payload" <<'PY'
import json
import sys
path, payload = sys.argv[1], sys.argv[2]
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(json.loads(payload), ensure_ascii=False, sort_keys=True) + "\n")
PY
}

json_quote() {
  "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

run_medical_source() {
  local source="$1"
  local base_url="$2"
  local source_dir="$RUN_ROOT/$source"
  local data_dir="$DATA_ROOT/$source"
  local dataset_path="$data_dir/${source}_${PER_SOURCE_N}.jsonl"
  mkdir -p "$source_dir" "$data_dir"
  export DEEPSEEK_BASE_URL="$base_url"

  log "START source=$source build n=$PER_SOURCE_N base_url=$base_url"
  if ! "$PY" -m medimem data build-medical-pool \
      --per-source-n "$PER_SOURCE_N" \
      --sources "$source" \
      --output-dir "$data_dir" \
      --require-real-data \
      --cache-dir "$CACHE_DIR" \
      --random-seed "$RANDOM_SEED" \
      > "$source_dir/build.log" 2>&1; then
    local err
    err="$(tail -40 "$source_dir/build.log" | tr '\n' ' ')"
    log "BLOCKED source=$source stage=build"
    append_json "$BLOCKED_JSONL" "{\"source\":\"$source\",\"stage\":\"build\",\"error\":$(printf '%s' "$err" | json_quote)}"
    return 1
  fi

  log "START source=$source validate"
  if ! "$PY" -m medimem data validate --dataset "$dataset_path" > "$source_dir/validate.log" 2>&1; then
    local err
    err="$(tail -40 "$source_dir/validate.log" | tr '\n' ' ')"
    log "BLOCKED source=$source stage=validate"
    append_json "$BLOCKED_JSONL" "{\"source\":\"$source\",\"stage\":\"validate\",\"error\":$(printf '%s' "$err" | json_quote)}"
    return 1
  fi

  log "START source=$source experiment max_workers=$MAX_WORKERS_PER_PORT"
  if ! "$PY" -m medimem experiment-suite \
      --dataset "$dataset_path" \
      --require-api \
      --max-workers "$MAX_WORKERS_PER_PORT" \
      --suite-profile fast-formal \
      --baseline-set required \
      --ablation-groups full,no_memory_cleaning,no_evidence_note_injection \
      --counterfactual-policy risk_sample \
      --counterfactual-sample-rate "$COUNTERFACTUAL_SAMPLE_RATE" \
      --counterfactual-risk-threshold "$COUNTERFACTUAL_RISK_THRESHOLD" \
      > "$source_dir/experiment.log" 2>&1; then
    local err
    err="$(tail -80 "$source_dir/experiment.log" | tr '\n' ' ')"
    log "BLOCKED source=$source stage=experiment"
    append_json "$BLOCKED_JSONL" "{\"source\":\"$source\",\"stage\":\"experiment\",\"error\":$(printf '%s' "$err" | json_quote)}"
    return 1
  fi

  local run_dir
  run_dir="$(grep -E '^run_dir=' "$source_dir/experiment.log" | tail -1 | cut -d= -f2-)"
  if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    log "BLOCKED source=$source stage=run_dir"
    append_json "$BLOCKED_JSONL" "{\"source\":\"$source\",\"stage\":\"run_dir\",\"error\":\"experiment completed but run_dir was not found\"}"
    return 1
  fi

  local copied_run_dir="$source_dir/$(basename "$run_dir")"
  cp -a "$run_dir" "$copied_run_dir"
  append_json "$STATUS_JSONL" "{\"source\":\"$source\",\"kind\":\"medical\",\"dataset_path\":\"$dataset_path\",\"run_dir\":\"$run_dir\",\"copied_run_dir\":\"$copied_run_dir\",\"base_url\":\"$base_url\",\"completed_at\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}"
  log "DONE source=$source run_dir=$run_dir"
}

run_locomo() {
  local base_url="$1"
  local source="locomo"
  local source_dir="$RUN_ROOT/$source"
  mkdir -p "$source_dir"
  export DEEPSEEK_BASE_URL="$base_url"

  log "START source=locomo benchmark sample_n=$LOCOMO_SAMPLE_N base_url=$base_url"
  if ! "$PY" -m medimem benchmark run \
      --dataset locomo \
      --methods "$LOCOMO_METHODS" \
      --dataset-path "$LOCOMO_DATASET_PATH" \
      --sample-n "$LOCOMO_SAMPLE_N" \
      --random-seed 20260529 \
      --max-workers "$MAX_WORKERS_PER_PORT" \
      --require-api \
      --output-root "$source_dir" \
      > "$source_dir/benchmark.log" 2>&1; then
    local err
    err="$(tail -80 "$source_dir/benchmark.log" | tr '\n' ' ')"
    log "BLOCKED source=locomo stage=benchmark"
    append_json "$BLOCKED_JSONL" "{\"source\":\"locomo\",\"stage\":\"benchmark\",\"error\":$(printf '%s' "$err" | json_quote)}"
    return 1
  fi

  local run_dir
  run_dir="$(grep -E '^run_dir=' "$source_dir/benchmark.log" | tail -1 | cut -d= -f2-)"
  append_json "$STATUS_JSONL" "{\"source\":\"locomo\",\"kind\":\"benchmark\",\"dataset_path\":\"$LOCOMO_DATASET_PATH\",\"run_dir\":\"$run_dir\",\"base_url\":\"$base_url\",\"completed_at\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}"
  log "DONE source=locomo run_dir=$run_dir"
}

run_queue() {
  local queue_name="$1"
  local base_url="$2"
  shift 2
  local completed=0
  for source in "$@"; do
    if [ "$source" = "locomo" ]; then
      if run_locomo "$base_url"; then
        completed=$((completed + 1))
      fi
    else
      if run_medical_source "$source" "$base_url"; then
        completed=$((completed + 1))
      fi
    fi
    log "progress queue=$queue_name completed=$completed target=$#"
  done
}

summarize_results() {
  "$PY" - "$RUN_ROOT" "$PER_SOURCE_N" "$LOCOMO_SAMPLE_N" "$RANDOM_SEED" "$MAX_WORKERS_PER_PORT" "$MODEL_NAME" "$BASE_URL_A" "$BASE_URL_B" <<'PY'
import csv
import json
import pathlib
import time

import sys

run_root = pathlib.Path(sys.argv[1])
per_source_n = int(sys.argv[2])
locomo_sample_n = int(sys.argv[3])
random_seed = int(sys.argv[4])
max_workers_per_port = int(sys.argv[5])
model_name = sys.argv[6]
base_url_a = sys.argv[7]
base_url_b = sys.argv[8]
status_path = run_root / "source_status.jsonl"
blocked_path = run_root / "blocked_sources.jsonl"
completed = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()] if status_path.exists() else []
blocked = [json.loads(line) for line in blocked_path.read_text(encoding="utf-8").splitlines() if line.strip()] if blocked_path.exists() else []
rows = []
for item in completed:
    run_dir = pathlib.Path(item.get("copied_run_dir") or item.get("run_dir") or "")
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        metrics_path = run_dir / "locomo_metrics.csv"
    if not metrics_path.exists():
        continue
    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["dataset"] = item.get("source")
            row["kind"] = item.get("kind")
            row["run_dir"] = str(run_dir)
            rows.append(row)
if rows:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (run_root / "fast_formal_8_sources_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
(run_root / "fast_formal_8_sources_manifest.json").write_text(json.dumps({
    "track": "fast_formal_8_sources",
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "per_source_n": per_source_n,
    "locomo_sample_n": locomo_sample_n,
    "random_seed": random_seed,
    "max_workers_per_port": max_workers_per_port,
    "model_name": model_name,
    "base_url_a": base_url_a,
    "base_url_b": base_url_b,
    "completed_sources": completed,
    "blocked_sources": blocked,
    "comparison_csv": str(run_root / "fast_formal_8_sources_comparison.csv"),
}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

log "fast_formal_8_sources started per_source_n=$PER_SOURCE_N locomo_sample_n=$LOCOMO_SAMPLE_N random_seed=$RANDOM_SEED workers_per_port=$MAX_WORKERS_PER_PORT model=$MODEL_NAME"
run_queue A "$BASE_URL_A" "${QUEUE_A_SOURCES[@]}" &
pid_a=$!
run_queue B "$BASE_URL_B" "${QUEUE_B_SOURCES[@]}" &
pid_b=$!
wait "$pid_a"
status_a=$?
wait "$pid_b"
status_b=$?
summarize_results
if [ "$status_a" -ne 0 ] || [ "$status_b" -ne 0 ]; then
  log "FINISHED_WITH_BLOCKERS queue_a=$status_a queue_b=$status_b"
  exit 1
fi
log "FINISHED run_root=$RUN_ROOT"

