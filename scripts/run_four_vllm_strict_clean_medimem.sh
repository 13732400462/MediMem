#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/syh/A-mem}"
PY="${PY:-/root/miniconda3/envs/llmserve/bin/python}"
VLLM_PY="${VLLM_PY:-/root/miniconda3/envs/llmserve/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/syh/mem_ehr_hf_cache}"
RUN_ROOT="${RUN_ROOT:-runs/four_vllm_strict_clean_medimem_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-data/processed/four_vllm_strict_clean_medimem_$(date +%Y%m%d_%H%M%S)}"
MEDICAL_SOURCES="${MEDICAL_SOURCES:-medmcqa,medqa,chatdoctor_healthcaremagic,medical_meadow_wikidoc,pmoa_tts,pmc_patients}"
PER_SOURCE_N="${PER_SOURCE_N:-1000}"
LOCOMO_SAMPLE_N="${LOCOMO_SAMPLE_N:-1000}"
RANDOM_SEED="${RANDOM_SEED:-20260606}"
MAX_WORKERS="${MAX_WORKERS:-128}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
PMOA_ABLATION_GROUPS="${PMOA_ABLATION_GROUPS:-full,no_memory_cleaning,no_evidence_note_injection,with_polluted_memory,no_temporal_signal}"
LOCOMO_DATASET_PATH="${LOCOMO_DATASET_PATH:-datasets/amem_original/locomo/locomo10.official.json}"
LOCOMO_METHODS="${LOCOMO_METHODS:-direct,amem,medimem}"

MODEL_0_8B_PATH="${MODEL_0_8B_PATH:-/root/models/qwen_vl_size_sweep/0_8b}"
MODEL_2B_PATH="${MODEL_2B_PATH:-/root/models/qwen_vl_size_sweep/2b}"
MODEL_4B_PATH="${MODEL_4B_PATH:-/root/models/qwen_vl_size_sweep/4b}"
MODEL_8B_PATH="${MODEL_8B_PATH:-/root/models/qwen_vl_size_sweep/8b}"

cd "$PROJECT_ROOT"
export PYTHONPATH=.
export MEM_EHR_DATA_CACHE_DIR="$CACHE_DIR"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-local-qwen3}"
export DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-600}"
export DEEPSEEK_MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-1800}"
export BENCHMARK_MAX_TOKENS="${BENCHMARK_MAX_TOKENS:-256}"
export MEDICAL_PREDICTION_MAX_TOKENS="${MEDICAL_PREDICTION_MAX_TOKENS:-128}"
export MEDICAL_JSON_PARSE_RETRIES="${MEDICAL_JSON_PARSE_RETRIES:-2}"
export FAST_FORMAL_EARLY_STOP_ON_WIN=0
export MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER="${MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER:-5}"
export MEDICAL_STRICT_NO_LEAK_FILTER=1

mkdir -p "$RUN_ROOT" "$DATA_ROOT" "$CACHE_DIR"
STATUS_JSONL="$RUN_ROOT/status.jsonl"
BLOCKED_JSONL="$RUN_ROOT/blocked.jsonl"
SUMMARY_JSON="$RUN_ROOT/summary.json"
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
import json, sys
path, payload = sys.argv[1], sys.argv[2]
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(json.loads(payload), ensure_ascii=False, sort_keys=True) + "\n")
PY
}

run_logged() {
  local label="$1"
  local logfile="$2"
  shift 2
  log "RUN label=$label log=$logfile"
  "$@" > "$logfile" 2>&1
}

wait_for_model() {
  local base_url="$1"
  local served_name="$2"
  for _ in $(seq 1 180); do
    if "$PY" - "$base_url" "$served_name" <<'PY'
import json, sys, urllib.request
base_url, served = sys.argv[1].rstrip("/"), sys.argv[2]
try:
    with urllib.request.urlopen(base_url + "/models", timeout=3) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    ids = [item.get("id") for item in payload.get("data", [])]
    raise SystemExit(0 if served in ids or ids else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 5
  done
  return 1
}

start_vllm() {
  local gpu="$1" port="$2" model_path="$3" served_name="$4" model_root="$5"
  mkdir -p "$model_root"
  log "START vLLM gpu=$gpu port=$port model=$served_name path=$model_path"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" VLLM_USE_V1="${VLLM_USE_V1:-0}" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 \
      --port "$port" \
      --model "$model_path" \
      --served-model-name "$served_name" \
      --trust-remote-code \
      --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --enforce-eager \
      > "$model_root/vllm.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$model_root/vllm.pid"
  local base_url="http://127.0.0.1:${port}/v1"
  wait_for_model "$base_url" "$served_name"
  append_json "$STATUS_JSONL" "{\"stage\":\"vllm\",\"status\":\"ready\",\"model\":\"$served_name\",\"gpu\":$gpu,\"port\":$port,\"pid\":$pid,\"base_url\":\"$base_url\"}"
}

stop_vllm() {
  local model_root="$1"
  if [ -f "$model_root/vllm.pid" ]; then
    local pid
    pid="$(cat "$model_root/vllm.pid")"
    if kill -0 "$pid" 2>/dev/null; then
      log "STOP vLLM pid=$pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
}

build_shared_pool() {
  local data_dir="$DATA_ROOT/shared_medical_pool"
  mkdir -p "$data_dir"
  log "BUILD shared pool sources=$MEDICAL_SOURCES per_source_n=$PER_SOURCE_N seed=$RANDOM_SEED"
  run_logged "build-shared-pool" "$RUN_ROOT/build_shared_pool.log" \
    "$PY" -m mem_ehr_agent data build-medical-pool \
      --per-source-n "$PER_SOURCE_N" \
      --sources "$MEDICAL_SOURCES" \
      --output-dir "$data_dir" \
      --require-real-data \
      --cache-dir "$CACHE_DIR" \
      --random-seed "$RANDOM_SEED"
  SHARED_POOL_PATH="$("$PY" - "$data_dir/manifest.json" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["pooled_path"])
PY
)"
  SHARED_PMOA_PATH="$data_dir/pmoa_tts_${PER_SOURCE_N}.jsonl"
  if [ ! -s "$SHARED_PMOA_PATH" ]; then
    echo "missing shared PMOA slice: $SHARED_PMOA_PATH" >&2
    exit 1
  fi
  run_logged "validate-shared-pool" "$RUN_ROOT/validate_shared_pool.log" \
    "$PY" -m mem_ehr_agent data validate --dataset "$SHARED_POOL_PATH"
  run_logged "validate-shared-pmoa" "$RUN_ROOT/validate_shared_pmoa.log" \
    "$PY" -m mem_ehr_agent data validate --dataset "$SHARED_PMOA_PATH"
  append_json "$STATUS_JSONL" "{\"stage\":\"shared_data\",\"status\":\"ready\",\"pool\":\"$SHARED_POOL_PATH\",\"pmoa\":\"$SHARED_PMOA_PATH\",\"seed\":$RANDOM_SEED}"
}

run_pool() {
  local served="$1" base_url="$2" model_root="$3"
  local out="$model_root/pool"
  mkdir -p "$out"
  export DEEPSEEK_BASE_URL="$base_url" DEEPSEEK_MODEL="$served"
  log "START pool model=$served workers=$MAX_WORKERS dataset=$SHARED_POOL_PATH"
  run_logged "pool:$served" "$out/experiment.log" \
    "$PY" -u -m mem_ehr_agent experiment-suite \
      --dataset "$SHARED_POOL_PATH" \
      --require-api \
      --max-workers "$MAX_WORKERS" \
      --suite-profile fast-formal \
      --baseline-set required \
      --ablation-groups full \
      --defer-reports
  append_json "$STATUS_JSONL" "{\"stage\":\"pool\",\"status\":\"started_or_finished\",\"model\":\"$served\",\"log\":\"$out/experiment.log\"}"
}

run_locomo() {
  local served="$1" base_url="$2" model_root="$3"
  local out="$model_root/locomo"
  mkdir -p "$out"
  export DEEPSEEK_BASE_URL="$base_url" DEEPSEEK_MODEL="$served"
  log "START locomo model=$served workers=$MAX_WORKERS sample_n=$LOCOMO_SAMPLE_N seed=$RANDOM_SEED"
  run_logged "locomo:$served" "$out/benchmark.log" \
    "$PY" -u -m mem_ehr_agent benchmark run \
      --dataset locomo \
      --methods "$LOCOMO_METHODS" \
      --dataset-path "$LOCOMO_DATASET_PATH" \
      --sample-n "$LOCOMO_SAMPLE_N" \
      --random-seed "$RANDOM_SEED" \
      --max-workers "$MAX_WORKERS" \
      --require-api \
      --output-root "$out"
  append_json "$STATUS_JSONL" "{\"stage\":\"locomo\",\"status\":\"finished\",\"model\":\"$served\",\"log\":\"$out/benchmark.log\"}"
}

run_pmoa() {
  local served="$1" base_url="$2" model_root="$3"
  local out="$model_root/pmoa_ablation"
  mkdir -p "$out"
  export DEEPSEEK_BASE_URL="$base_url" DEEPSEEK_MODEL="$served"
  log "START pmoa model=$served workers=$MAX_WORKERS dataset=$SHARED_PMOA_PATH groups=$PMOA_ABLATION_GROUPS"
  run_logged "pmoa:$served" "$out/experiment.log" \
    "$PY" -u -m mem_ehr_agent experiment-suite \
      --dataset "$SHARED_PMOA_PATH" \
      --require-api \
      --max-workers "$MAX_WORKERS" \
      --suite-profile fast-formal \
      --baseline-set none \
      --ablation-groups "$PMOA_ABLATION_GROUPS" \
      --defer-reports
  append_json "$STATUS_JSONL" "{\"stage\":\"pmoa_ablation\",\"status\":\"finished\",\"model\":\"$served\",\"log\":\"$out/experiment.log\"}"
}

run_model_sequence() {
  local gpu="$1" port="$2"
  shift 2
  while [ "$#" -gt 0 ]; do
    local model_path="$1" served="$2"
    shift 2
    local model_root="$RUN_ROOT/$served"
    local base_url="http://127.0.0.1:${port}/v1"
    local status=0
    if ! start_vllm "$gpu" "$port" "$model_path" "$served" "$model_root"; then
      append_json "$BLOCKED_JSONL" "{\"model\":\"$served\",\"stage\":\"vllm\",\"status\":\"failed\"}"
      return 1
    fi
    run_pool "$served" "$base_url" "$model_root" || status=$?
    if [ "$status" -eq 0 ]; then run_locomo "$served" "$base_url" "$model_root" || status=$?; fi
    if [ "$status" -eq 0 ]; then run_pmoa "$served" "$base_url" "$model_root" || status=$?; fi
    stop_vllm "$model_root" || true
    if [ "$status" -ne 0 ]; then
      append_json "$BLOCKED_JSONL" "{\"model\":\"$served\",\"stage\":\"sequence\",\"status\":\"failed\",\"code\":$status}"
      return "$status"
    fi
    append_json "$STATUS_JSONL" "{\"stage\":\"model_sequence\",\"status\":\"finished\",\"model\":\"$served\"}"
  done
}

summarize() {
  "$PY" - "$RUN_ROOT" "$STATUS_JSONL" "$BLOCKED_JSONL" "$SUMMARY_JSON" <<'PY'
import json, pathlib, sys
run_root = pathlib.Path(sys.argv[1])
status = pathlib.Path(sys.argv[2])
blocked = pathlib.Path(sys.argv[3])
summary = pathlib.Path(sys.argv[4])
rows = [json.loads(x) for x in status.read_text(encoding="utf-8").splitlines() if x.strip()]
blocks = [json.loads(x) for x in blocked.read_text(encoding="utf-8").splitlines() if x.strip()]
summary.write_text(json.dumps({"run_root": str(run_root), "status_rows": rows, "blocked": blocks, "passed": not blocks}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

log "four-vllm strict clean MediMem started run_root=$RUN_ROOT data_root=$DATA_ROOT"
build_shared_pool
left_status=0
right_status=0
run_model_sequence 0 8000 "$MODEL_0_8B_PATH" qwen3-vl-0_8b "$MODEL_4B_PATH" qwen3-vl-4b &
left_pid=$!
run_model_sequence 1 8001 "$MODEL_2B_PATH" qwen3-vl-2b "$MODEL_8B_PATH" qwen3-vl-8b &
right_pid=$!
wait "$left_pid" || left_status=$?
wait "$right_pid" || right_status=$?
summarize
log "FINISHED summary=$SUMMARY_JSON left_status=$left_status right_status=$right_status"
if [ "$left_status" -ne 0 ] || [ "$right_status" -ne 0 ]; then
  exit 1
fi
