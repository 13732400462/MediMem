#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/syh/A-mem}"
PY="${PY:-/root/miniconda3/envs/llmserve/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/syh/mem_ehr_hf_cache}"
DATA_ROOT="${DATA_ROOT:-data/processed/fast_formal_medical_pooled_$(date +%Y%m%d)}"
RUN_ROOT="${RUN_ROOT:-runs/fast_formal_medical_pooled_$(date +%Y%m%d_%H%M%S)}"
MEDICAL_SOURCES="${MEDICAL_SOURCES:-medmcqa,medqa,chatdoctor_healthcaremagic,medical_meadow_wikidoc,medical_dialogue_to_soap,pmoa_tts,pmc_patients}"
PER_SOURCE_N="${PER_SOURCE_N:-200}"
LOCOMO_SAMPLE_N="${LOCOMO_SAMPLE_N:-200}"
RANDOM_SEED="${RANDOM_SEED:-20260606}"
GPU_ID="${GPU_ID:-0}"
MAX_WORKERS="${MAX_WORKERS:-96}"
LOCOMO_MAX_WORKERS="${LOCOMO_MAX_WORKERS:-16}"
COUNTERFACTUAL_SAMPLE_RATE="${COUNTERFACTUAL_SAMPLE_RATE:-0.20}"
COUNTERFACTUAL_RISK_THRESHOLD="${COUNTERFACTUAL_RISK_THRESHOLD:-0.55}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-4b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
LOCOMO_BASE_URL="${LOCOMO_BASE_URL:-$BASE_URL}"
LOCOMO_DATASET_PATH="${LOCOMO_DATASET_PATH:-datasets/amem_original/locomo/locomo10.official.json}"
LOCOMO_METHODS="${LOCOMO_METHODS:-direct,amem,medimem}"
RUN_LOCOMO="${RUN_LOCOMO:-1}"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=.
export MEM_EHR_DATA_CACHE_DIR="$CACHE_DIR"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-local-qwen3}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-$MODEL_NAME}"
export DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-600}"
export FAST_FORMAL_EARLY_STOP_ON_WIN=0
export MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER="${MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER:-5}"
export MEDICAL_STRICT_NO_LEAK_FILTER="${MEDICAL_STRICT_NO_LEAK_FILTER:-1}"

mkdir -p "$RUN_ROOT" "$DATA_ROOT" "$CACHE_DIR"
STATUS_JSONL="$RUN_ROOT/status.jsonl"
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

run_medical_pooled() {
  local source_dir="$RUN_ROOT/medical_pooled"
  mkdir -p "$source_dir"
  export DEEPSEEK_BASE_URL="$BASE_URL"

  log "START medical_pooled build sources=$MEDICAL_SOURCES per_source_n=$PER_SOURCE_N"
  if ! "$PY" -m mem_ehr_agent data build-medical-pool \
      --per-source-n "$PER_SOURCE_N" \
      --sources "$MEDICAL_SOURCES" \
      --output-dir "$DATA_ROOT/medical_pooled" \
      --require-real-data \
      --cache-dir "$CACHE_DIR" \
      --random-seed "$RANDOM_SEED" \
      > "$source_dir/build.log" 2>&1; then
    local err
    err="$(tail -80 "$source_dir/build.log" | tr '\n' ' ')"
    log "BLOCKED medical_pooled stage=build"
    append_json "$BLOCKED_JSONL" "{\"source\":\"medical_pooled\",\"stage\":\"build\",\"error\":$(printf '%s' "$err" | json_quote)}"
    return 1
  fi

  local dataset_path
  dataset_path="$("$PY" - "$DATA_ROOT/medical_pooled/manifest.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["pooled_path"])
PY
)"

  log "START medical_pooled validate dataset=$dataset_path"
  if ! "$PY" -m mem_ehr_agent data validate --dataset "$dataset_path" > "$source_dir/validate.log" 2>&1; then
    local err
    err="$(tail -80 "$source_dir/validate.log" | tr '\n' ' ')"
    log "BLOCKED medical_pooled stage=validate"
    append_json "$BLOCKED_JSONL" "{\"source\":\"medical_pooled\",\"stage\":\"validate\",\"error\":$(printf '%s' "$err" | json_quote)}"
    return 1
  fi

  log "START medical_pooled experiment max_workers=$MAX_WORKERS base_url=$BASE_URL"
  if ! "$PY" -u -m mem_ehr_agent experiment-suite \
      --dataset "$dataset_path" \
      --require-api \
      --max-workers "$MAX_WORKERS" \
      --suite-profile fast-formal \
      --baseline-set required \
      --ablation-groups full,no_memory_cleaning,no_evidence_note_injection \
      --counterfactual-policy risk_sample \
      --counterfactual-sample-rate "$COUNTERFACTUAL_SAMPLE_RATE" \
      --counterfactual-risk-threshold "$COUNTERFACTUAL_RISK_THRESHOLD" \
      --defer-reports \
      2>&1 | tee "$source_dir/experiment.log"; then
    local err
    err="$(tail -120 "$source_dir/experiment.log" | tr '\n' ' ')"
    log "BLOCKED medical_pooled stage=experiment"
    append_json "$BLOCKED_JSONL" "{\"source\":\"medical_pooled\",\"stage\":\"experiment\",\"error\":$(printf '%s' "$err" | json_quote)}"
    return 1
  fi

  local run_dir
  run_dir="$(grep -E '^run_dir=' "$source_dir/experiment.log" | tail -1 | cut -d= -f2-)"
  if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    log "BLOCKED medical_pooled stage=run_dir"
    append_json "$BLOCKED_JSONL" "{\"source\":\"medical_pooled\",\"stage\":\"run_dir\",\"error\":\"experiment completed but run_dir was not found\"}"
    return 1
  fi
  append_json "$STATUS_JSONL" "{\"source\":\"medical_pooled\",\"kind\":\"medical_pooled\",\"dataset_path\":\"$dataset_path\",\"run_dir\":\"$run_dir\",\"base_url\":\"$BASE_URL\",\"completed_at\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}"
  log "DONE medical_pooled run_dir=$run_dir"
}

run_locomo() {
  local source_dir="$RUN_ROOT/locomo"
  mkdir -p "$source_dir"
  export DEEPSEEK_BASE_URL="$LOCOMO_BASE_URL"

  log "START locomo benchmark sample_n=$LOCOMO_SAMPLE_N max_workers=$LOCOMO_MAX_WORKERS"
  if ! "$PY" -u -m mem_ehr_agent benchmark run \
      --dataset locomo \
      --methods "$LOCOMO_METHODS" \
      --dataset-path "$LOCOMO_DATASET_PATH" \
      --sample-n "$LOCOMO_SAMPLE_N" \
      --random-seed "$RANDOM_SEED" \
      --max-workers "$LOCOMO_MAX_WORKERS" \
      --require-api \
      --output-root "$source_dir" \
      2>&1 | tee "$source_dir/benchmark.log"; then
    local err
    err="$(tail -120 "$source_dir/benchmark.log" | tr '\n' ' ')"
    log "BLOCKED locomo stage=benchmark"
    append_json "$BLOCKED_JSONL" "{\"source\":\"locomo\",\"stage\":\"benchmark\",\"error\":$(printf '%s' "$err" | json_quote)}"
    return 1
  fi

  local run_dir
  run_dir="$(grep -E '^run_dir=' "$source_dir/benchmark.log" | tail -1 | cut -d= -f2-)"
  append_json "$STATUS_JSONL" "{\"source\":\"locomo\",\"kind\":\"benchmark\",\"dataset_path\":\"$LOCOMO_DATASET_PATH\",\"run_dir\":\"$run_dir\",\"base_url\":\"$LOCOMO_BASE_URL\",\"completed_at\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}"
  log "DONE locomo run_dir=$run_dir"
}

summarize_results() {
  "$PY" - "$RUN_ROOT" "$DATA_ROOT/medical_pooled/manifest.json" "$PER_SOURCE_N" "$LOCOMO_SAMPLE_N" "$RANDOM_SEED" "$GPU_ID" "$MAX_WORKERS" "$LOCOMO_MAX_WORKERS" "$MODEL_NAME" "$BASE_URL" "$LOCOMO_BASE_URL" <<'PY'
import csv
import json
import pathlib
import sys
import time

run_root = pathlib.Path(sys.argv[1])
medical_manifest_path = pathlib.Path(sys.argv[2])
per_source_n = int(sys.argv[3])
locomo_sample_n = int(sys.argv[4])
random_seed = int(sys.argv[5])
gpu_id = sys.argv[6]
max_workers = int(sys.argv[7])
locomo_max_workers = int(sys.argv[8])
model_name = sys.argv[9]
base_url = sys.argv[10]
locomo_base_url = sys.argv[11]
status_path = run_root / "status.jsonl"
blocked_path = run_root / "blocked_sources.jsonl"
completed = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()] if status_path.exists() else []
blocked = [json.loads(line) for line in blocked_path.read_text(encoding="utf-8").splitlines() if line.strip()] if blocked_path.exists() else []
rows = []
leakage = []
for item in completed:
    run_dir = pathlib.Path(item.get("run_dir") or "")
    metrics_path = run_dir / "source_metrics.csv" if item.get("kind") == "medical_pooled" else run_dir / "locomo_metrics.csv"
    if not metrics_path.exists():
        metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row.setdefault("dataset", item.get("source"))
                row.setdefault("source", item.get("source"))
                row["kind"] = item.get("kind")
                row["run_dir"] = str(run_dir)
                rows.append(row)
    audit_path = run_dir / "leakage_audit.jsonl"
    if audit_path.exists():
        audits = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if audits:
            leakage.append({"source": item.get("source"), "run_dir": str(run_dir), **audits[-1]})
critical = sum(int(item.get("critical_leakage_count", 0) or 0) for item in leakage)
needs_review = sum(int(item.get("needs_review_count", 0) or 0) for item in leakage)
if rows:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (run_root / "fast_formal_medical_pooled_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
medical_manifest = json.loads(medical_manifest_path.read_text(encoding="utf-8")) if medical_manifest_path.exists() else {}
(run_root / "fast_formal_medical_pooled_manifest.json").write_text(json.dumps({
    "track": "fast_formal_medical_pooled",
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "per_source_n": per_source_n,
    "locomo_sample_n": locomo_sample_n,
    "random_seed": random_seed,
    "gpu_id": gpu_id,
    "max_workers": max_workers,
    "locomo_max_workers": locomo_max_workers,
    "model_name": model_name,
    "base_url": base_url,
    "locomo_base_url": locomo_base_url,
    "medical_data_manifest": medical_manifest,
    "completed": completed,
    "blocked_sources": blocked,
    "leakage_audit": leakage,
    "critical_leakage_count": critical,
    "needs_review_count": needs_review,
    "comparison_csv": str(run_root / "fast_formal_medical_pooled_comparison.csv"),
}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if critical:
    raise SystemExit(f"critical leakage count is non-zero: {critical}")
if needs_review:
    raise SystemExit(f"needs-review leakage count is non-zero: {needs_review}")
PY
}

log "fast_formal_medical_pooled started sources=$MEDICAL_SOURCES per_source_n=$PER_SOURCE_N locomo_sample_n=$LOCOMO_SAMPLE_N seed=$RANDOM_SEED gpu=$GPU_ID workers=$MAX_WORKERS model=$MODEL_NAME base_url=$BASE_URL"
run_medical_pooled
medical_status=$?
locomo_status=0
if [ "$RUN_LOCOMO" = "1" ] || [ "$RUN_LOCOMO" = "true" ]; then
  run_locomo
  locomo_status=$?
fi
summarize_results
if [ "$medical_status" -ne 0 ] || [ "$locomo_status" -ne 0 ]; then
  log "FINISHED_WITH_BLOCKERS medical=$medical_status locomo=$locomo_status"
  exit 1
fi
log "FINISHED run_root=$RUN_ROOT"
