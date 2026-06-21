#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/syh/A-mem}"
PY="${PY:-/root/miniconda3/envs/llmserve/bin/python}"
VLLM_PY="${VLLM_PY:-/root/miniconda3/envs/llmserve/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/syh/mem_ehr_hf_cache}"
RUN_ROOT="${RUN_ROOT:-runs/strict_size_sweep_dual_pipeline_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-data/processed/strict_size_sweep_dual_pipeline_$(date +%Y%m%d_%H%M%S)}"
MEDICAL_SOURCES="${MEDICAL_SOURCES:-medmcqa,medqa,chatdoctor_healthcaremagic,medical_meadow_wikidoc,pmoa_tts,pmc_patients}"
PER_SOURCE_N="${PER_SOURCE_N:-1000}"
LOCOMO_SAMPLE_N="${LOCOMO_SAMPLE_N:-1000}"
RANDOM_SEED="${RANDOM_SEED:-20260606}"
PIPELINE_WORKERS="${PIPELINE_WORKERS:-48}"
EXCLUSIVE_PIPELINE_WORKERS="${EXCLUSIVE_PIPELINE_WORKERS:-128}"
PIPELINE_SCHEDULE="${PIPELINE_SCHEDULE:-pool_then_locomo}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-600}"
DEEPSEEK_MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-1800}"
BENCHMARK_MAX_TOKENS="${BENCHMARK_MAX_TOKENS:-256}"
MEDICAL_PREDICTION_MAX_TOKENS="${MEDICAL_PREDICTION_MAX_TOKENS:-128}"
MEDICAL_JSON_PARSE_RETRIES="${MEDICAL_JSON_PARSE_RETRIES:-2}"
LOCOMO_DATASET_PATH="${LOCOMO_DATASET_PATH:-datasets/amem_original/locomo/locomo10.official.json}"
LOCOMO_METHODS="${LOCOMO_METHODS:-direct,amem,medimem}"
MEDICAL_MAIN_GROUPS="${MEDICAL_MAIN_GROUPS:-full}"
PMOA_ABLATION_GROUPS="${PMOA_ABLATION_GROUPS:-full,no_memory_cleaning,no_evidence_note_injection,ablate_with_polluted_memory}"

cd "$PROJECT_ROOT"
export PYTHONPATH=.
export MEM_EHR_DATA_CACHE_DIR="$CACHE_DIR"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-local-qwen3}"
export DEEPSEEK_TIMEOUT
export DEEPSEEK_MAX_TOKENS
export BENCHMARK_MAX_TOKENS
export MEDICAL_PREDICTION_MAX_TOKENS
export MEDICAL_JSON_PARSE_RETRIES
export FAST_FORMAL_EARLY_STOP_ON_WIN=0
export MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER="${MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER:-5}"
export MEDICAL_STRICT_NO_LEAK_FILTER=1

mkdir -p "$RUN_ROOT" "$DATA_ROOT" "$CACHE_DIR"
STATUS_JSONL="$RUN_ROOT/status.jsonl"
BLOCKED_JSONL="$RUN_ROOT/blocked_sources.jsonl"
SUMMARY_JSON="$RUN_ROOT/strict_size_sweep_summary.json"
RUN_CHILD_PIDS="$RUN_ROOT/child_pids.tsv"
RUN_EXIT_JSON="$RUN_ROOT/launcher_exit.json"
: > "$STATUS_JSONL"
: > "$BLOCKED_JSONL"
: > "$RUN_CHILD_PIDS"
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

record_process_snapshot() {
  local path="$RUN_ROOT/process_snapshot_$(date +%Y%m%d_%H%M%S).txt"
  {
    date '+%Y-%m-%d %H:%M:%S'
    ps -eo pid,ppid,stat,pcpu,pmem,etime,args --sort=-pcpu | head -120
    printf '\n[registered children]\n'
    cat "$RUN_CHILD_PIDS" 2>/dev/null || true
    printf '\n[gpu]\n'
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
  } > "$path" 2>&1 || true
  echo "$path"
}

terminate_registered_children() {
  if [ ! -s "$RUN_CHILD_PIDS" ]; then
    return 0
  fi
  awk -F '\t' '{print $1}' "$RUN_CHILD_PIDS" | while read -r pid; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 3
  awk -F '\t' '{print $1}' "$RUN_CHILD_PIDS" | while read -r pid; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

on_signal() {
  local sig="$1"
  local snapshot
  snapshot="$(record_process_snapshot)"
  log "TERMINATED signal=$sig snapshot=$snapshot"
  append_json "$BLOCKED_JSONL" "{\"stage\":\"launcher\",\"status\":\"terminated\",\"signal\":\"$sig\",\"snapshot\":\"$snapshot\"}"
  terminate_registered_children
  exit 128
}

on_exit() {
  local status="$?"
  if [ "$status" -ne 0 ]; then
    local snapshot
    snapshot="$(record_process_snapshot)"
    append_json "$BLOCKED_JSONL" "{\"stage\":\"launcher\",\"status\":\"exited_nonzero\",\"exit_code\":$status,\"snapshot\":\"$snapshot\"}"
  fi
  "$PY" - "$RUN_EXIT_JSON" "$status" <<'PY' || true
import json
import sys
from datetime import datetime
path, status = sys.argv[1], int(sys.argv[2])
with open(path, "w", encoding="utf-8") as f:
    json.dump({"exit_code": status, "finished_at": datetime.now().isoformat(timespec="seconds")}, f, ensure_ascii=False, sort_keys=True)
    f.write("\n")
PY
}

trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
trap on_exit EXIT

run_logged() {
  local label="$1"
  local logfile="$2"
  shift 2
  "$@" > "$logfile" 2>&1 &
  local pid=$!
  printf '%s\t%s\t%s\n' "$pid" "$label" "$logfile" >> "$RUN_CHILD_PIDS"
  printf '%s\n' "$pid" > "${logfile}.pid"
  log "RUN command label=$label pid=$pid log=$logfile"
  local rc=0
  wait "$pid" || rc=$?
  printf '%s\n' "$rc" > "${logfile}.exit"
  log "DONE command label=$label pid=$pid rc=$rc log=$logfile"
  return "$rc"
}

json_quote() {
  "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

wait_for_model() {
  local base_url="$1"
  local name="$2"
  for _ in $(seq 1 180); do
    if "$PY" - "$base_url" "$name" <<'PY'
import json
import sys
import urllib.request
base_url, name = sys.argv[1].rstrip("/"), sys.argv[2]
try:
    with urllib.request.urlopen(base_url + "/models", timeout=3) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    ids = [item.get("id") for item in payload.get("data", [])]
    raise SystemExit(0 if name in ids or ids else 1)
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
  local gpu="$1"
  local port="$2"
  local model_path="$3"
  local served_name="$4"
  local model_root="$5"
  mkdir -p "$model_root"
  log "START vLLM gpu=$gpu port=$port model=$served_name path=$model_path"
  printf '%s\n' "$model_path" > "$model_root/vllm_model_path.txt"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" VLLM_USE_V1="${VLLM_USE_V1:-0}" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 \
      --port "$port" \
      --model "$model_path" \
      --served-model-name "$served_name" \
      --trust-remote-code \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --enforce-eager \
      --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
      --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
      > "$model_root/vllm.log" 2>&1 &
  local pid=$!
  printf '%s\t%s\t%s\n' "$pid" "vllm:$served_name" "$model_root/vllm.log" >> "$RUN_CHILD_PIDS"
  echo "$pid" > "$model_root/vllm.pid"
  local base_url="http://127.0.0.1:${port}/v1"
  if ! wait_for_model "$base_url" "$served_name"; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"stage\":\"vllm\",\"error\":\"model healthcheck failed\",\"log\":\"$model_root/vllm.log\"}"
    return 1
  fi
  append_json "$STATUS_JSONL" "{\"model\":\"$served_name\",\"stage\":\"vllm\",\"status\":\"ready\",\"gpu\":\"$gpu\",\"port\":\"$port\",\"pid\":$pid,\"base_url\":\"$base_url\"}"
}

stop_vllm() {
  local model_root="$1"
  if [ -f "$model_root/vllm.pid" ]; then
    local pid
    pid="$(cat "$model_root/vllm.pid")"
    local model_path=""
    if [ -f "$model_root/vllm_model_path.txt" ]; then
      model_path="$(cat "$model_root/vllm_model_path.txt")"
    fi
    if kill -0 "$pid" 2>/dev/null; then
      ps -p "$pid" -o pid,ppid,etime,args | tee -a "$RUN_ROOT/run.log" || true
      if [ -n "$model_path" ] && ! ps -p "$pid" -o args= | grep -Fq -- "$model_path"; then
        append_json "$BLOCKED_JSONL" "{\"stage\":\"vllm_stop\",\"pid\":$pid,\"error\":\"pid args did not match model path\",\"model_path\":\"$model_path\"}"
        return 1
      fi
      log "STOP vLLM pid=$pid model_path=$model_path"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
}

terminate_tree() {
  local pid="$1"
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  sleep 3
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

run_medical_pipeline() {
  local served_name="$1"
  local base_url="$2"
  local model_root="$3"
  local workers="$4"
  local phase="$5"
  local data_dir="$DATA_ROOT/$served_name/medical_pooled"
  local run_dir="$model_root/medical_pooled_${phase}"
  mkdir -p "$data_dir" "$run_dir"
  export DEEPSEEK_BASE_URL="$base_url"
  export DEEPSEEK_MODEL="$served_name"

  log "START medical model=$served_name phase=$phase workers=$workers per_source_n=$PER_SOURCE_N"
  if ! run_logged "build-medical:$served_name:$phase" "$run_dir/build.log" \
  "$PY" -m mem_ehr_agent data build-medical-pool \
    --per-source-n "$PER_SOURCE_N" \
    --sources "$MEDICAL_SOURCES" \
    --output-dir "$data_dir" \
    --require-real-data \
    --cache-dir "$CACHE_DIR" \
    --random-seed "$RANDOM_SEED"; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"medical_pooled\",\"phase\":\"$phase\",\"status\":\"failed\",\"stage\":\"build\",\"workers\":$workers,\"log\":\"$run_dir/build.log\"}"
    return 1
  fi

  local dataset_path
  dataset_path="$("$PY" - "$data_dir/manifest.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["pooled_path"])
PY
)"
  if ! run_logged "validate-medical:$served_name:$phase" "$run_dir/validate.log" \
    "$PY" -m mem_ehr_agent data validate --dataset "$dataset_path"; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"medical_pooled\",\"phase\":\"$phase\",\"status\":\"failed\",\"stage\":\"validate\",\"workers\":$workers,\"log\":\"$run_dir/validate.log\"}"
    return 1
  fi
  if ! run_logged "experiment-medical:$served_name:$phase" "$run_dir/experiment.log" \
    "$PY" -u -m mem_ehr_agent experiment-suite \
    --dataset "$dataset_path" \
    --require-api \
    --max-workers "$workers" \
    --suite-profile fast-formal \
    --baseline-set required \
    --ablation-groups "$MEDICAL_MAIN_GROUPS" \
    --counterfactual-policy risk_sample \
    --counterfactual-sample-rate "${COUNTERFACTUAL_SAMPLE_RATE:-0.20}" \
    --counterfactual-risk-threshold "${COUNTERFACTUAL_RISK_THRESHOLD:-0.55}" \
    --defer-reports; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"medical_pooled\",\"phase\":\"$phase\",\"status\":\"failed\",\"workers\":$workers,\"log\":\"$run_dir/experiment.log\"}"
    return 1
  fi
  local experiment_run_dir
  experiment_run_dir="$(grep -E '^run_dir=' "$run_dir/experiment.log" | tail -1 | cut -d= -f2-)"
  if [ -z "$experiment_run_dir" ] || [ ! -d "$experiment_run_dir" ]; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"medical_pooled\",\"phase\":\"$phase\",\"status\":\"failed\",\"workers\":$workers,\"error\":\"run_dir_missing_after_success\",\"log\":\"$run_dir/experiment.log\"}"
    return 1
  fi
  append_json "$STATUS_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"medical_pooled\",\"phase\":\"$phase\",\"status\":\"finished\",\"dataset_path\":\"$dataset_path\",\"run_dir\":\"$experiment_run_dir\",\"base_url\":\"$base_url\",\"workers\":$workers}"
}

run_pmoa_ablation_pipeline() {
  local served_name="$1"
  local base_url="$2"
  local model_root="$3"
  local workers="$4"
  local phase="$5"
  local data_dir="$DATA_ROOT/$served_name/pmoa_ablation"
  local run_dir="$model_root/pmoa_ablation_${phase}"
  mkdir -p "$data_dir" "$run_dir"
  export DEEPSEEK_BASE_URL="$base_url"
  export DEEPSEEK_MODEL="$served_name"

  log "START pmoa ablation model=$served_name phase=$phase workers=$workers per_source_n=$PER_SOURCE_N groups=$PMOA_ABLATION_GROUPS"
  if ! run_logged "build-pmoa:$served_name:$phase" "$run_dir/build.log" \
  "$PY" -m mem_ehr_agent data build-medical-pool \
    --per-source-n "$PER_SOURCE_N" \
    --sources pmoa_tts \
    --output-dir "$data_dir" \
    --require-real-data \
    --cache-dir "$CACHE_DIR" \
    --random-seed "$RANDOM_SEED"; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"pmoa_ablation\",\"phase\":\"$phase\",\"status\":\"failed\",\"stage\":\"build\",\"workers\":$workers,\"log\":\"$run_dir/build.log\"}"
    return 1
  fi

  local dataset_path
  dataset_path="$("$PY" - "$data_dir/manifest.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["pooled_path"])
PY
)"
  if ! run_logged "validate-pmoa:$served_name:$phase" "$run_dir/validate.log" \
    "$PY" -m mem_ehr_agent data validate --dataset "$dataset_path"; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"pmoa_ablation\",\"phase\":\"$phase\",\"status\":\"failed\",\"stage\":\"validate\",\"workers\":$workers,\"log\":\"$run_dir/validate.log\"}"
    return 1
  fi
  if ! run_logged "experiment-pmoa:$served_name:$phase" "$run_dir/experiment.log" \
    "$PY" -u -m mem_ehr_agent experiment-suite \
    --dataset "$dataset_path" \
    --require-api \
    --max-workers "$workers" \
    --suite-profile fast-formal \
    --baseline-set none \
    --ablation-groups "$PMOA_ABLATION_GROUPS" \
    --counterfactual-policy risk_sample \
    --counterfactual-sample-rate "${COUNTERFACTUAL_SAMPLE_RATE:-0.20}" \
    --counterfactual-risk-threshold "${COUNTERFACTUAL_RISK_THRESHOLD:-0.55}" \
    --defer-reports; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"pmoa_ablation\",\"phase\":\"$phase\",\"status\":\"failed\",\"workers\":$workers,\"log\":\"$run_dir/experiment.log\"}"
    return 1
  fi
  local experiment_run_dir
  experiment_run_dir="$(grep -E '^run_dir=' "$run_dir/experiment.log" | tail -1 | cut -d= -f2-)"
  if [ -z "$experiment_run_dir" ] || [ ! -d "$experiment_run_dir" ]; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"pmoa_ablation\",\"phase\":\"$phase\",\"status\":\"failed\",\"workers\":$workers,\"error\":\"run_dir_missing_after_success\",\"log\":\"$run_dir/experiment.log\"}"
    return 1
  fi
  append_json "$STATUS_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"pmoa_ablation\",\"phase\":\"$phase\",\"status\":\"finished\",\"dataset_path\":\"$dataset_path\",\"run_dir\":\"$experiment_run_dir\",\"base_url\":\"$base_url\",\"workers\":$workers}"
}

run_locomo_pipeline() {
  local served_name="$1"
  local base_url="$2"
  local model_root="$3"
  local workers="$4"
  local phase="$5"
  local run_dir="$model_root/locomo_${phase}"
  mkdir -p "$run_dir"
  export DEEPSEEK_BASE_URL="$base_url"
  export DEEPSEEK_MODEL="$served_name"

  log "START locomo model=$served_name phase=$phase workers=$workers sample_n=$LOCOMO_SAMPLE_N"
  if ! run_logged "benchmark-locomo:$served_name:$phase" "$run_dir/benchmark.log" \
    "$PY" -u -m mem_ehr_agent benchmark run \
    --dataset locomo \
    --methods "$LOCOMO_METHODS" \
    --dataset-path "$LOCOMO_DATASET_PATH" \
    --sample-n "$LOCOMO_SAMPLE_N" \
    --random-seed "$RANDOM_SEED" \
    --max-workers "$workers" \
    --require-api \
    --output-root "$run_dir"; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"locomo\",\"phase\":\"$phase\",\"status\":\"failed\",\"workers\":$workers,\"log\":\"$run_dir/benchmark.log\"}"
    return 1
  fi
  local benchmark_run_dir
  benchmark_run_dir="$(grep -E '^run_dir=' "$run_dir/benchmark.log" | tail -1 | cut -d= -f2-)"
  if [ -z "$benchmark_run_dir" ] || [ ! -d "$benchmark_run_dir" ]; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"locomo\",\"phase\":\"$phase\",\"status\":\"failed\",\"workers\":$workers,\"error\":\"run_dir_missing_after_success\",\"log\":\"$run_dir/benchmark.log\"}"
    return 1
  fi
  append_json "$STATUS_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"locomo\",\"phase\":\"$phase\",\"status\":\"finished\",\"dataset_path\":\"$LOCOMO_DATASET_PATH\",\"run_dir\":\"$benchmark_run_dir\",\"base_url\":\"$base_url\",\"workers\":$workers}"
}

run_model() {
  local gpu="$1"
  local port="$2"
  local model_path="$3"
  local served_name="$4"
  local model_root="$RUN_ROOT/$served_name"
  local base_url="http://127.0.0.1:${port}/v1"
  mkdir -p "$model_root"
  start_vllm "$gpu" "$port" "$model_path" "$served_name" "$model_root"
  local medical_status="$model_root/medical_shared.exit"
  local locomo_status="$model_root/locomo_shared.exit"
  rm -f "$medical_status" "$locomo_status"
  (run_medical_pipeline "$served_name" "$base_url" "$model_root" "$PIPELINE_WORKERS" "shared"; echo $? > "$medical_status") &
  local medical_pid=$!
  (run_locomo_pipeline "$served_name" "$base_url" "$model_root" "$PIPELINE_WORKERS" "shared"; echo $? > "$locomo_status") &
  local locomo_pid=$!
  local status=0
  local first=""
  while true; do
    if [ -s "$medical_status" ]; then
      first="medical"
      break
    fi
    if [ -s "$locomo_status" ]; then
      first="locomo"
      break
    fi
    sleep 5
  done
  if [ "$first" = "medical" ]; then
    status="$(cat "$medical_status")"
    if [ "$status" -ne 0 ]; then
      terminate_tree "$locomo_pid"
      wait "$locomo_pid" 2>/dev/null || true
    elif [ ! -s "$locomo_status" ]; then
      log "PROMOTE locomo model=$served_name from workers=$PIPELINE_WORKERS to workers=$EXCLUSIVE_PIPELINE_WORKERS after medical finished"
      append_json "$STATUS_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"locomo\",\"phase\":\"shared\",\"status\":\"superseded\",\"workers\":$PIPELINE_WORKERS,\"reason\":\"medical_finished_first_promote_to_exclusive\"}"
      terminate_tree "$locomo_pid"
      wait "$locomo_pid" 2>/dev/null || true
      run_locomo_pipeline "$served_name" "$base_url" "$model_root" "$EXCLUSIVE_PIPELINE_WORKERS" "exclusive" || status=$?
    else
      wait "$locomo_pid" || status=$?
    fi
  else
    status="$(cat "$locomo_status")"
    if [ "$status" -ne 0 ]; then
      terminate_tree "$medical_pid"
      wait "$medical_pid" 2>/dev/null || true
    elif [ ! -s "$medical_status" ]; then
      log "PROMOTE medical model=$served_name from workers=$PIPELINE_WORKERS to workers=$EXCLUSIVE_PIPELINE_WORKERS after locomo finished"
      append_json "$STATUS_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"medical_pooled\",\"phase\":\"shared\",\"status\":\"superseded\",\"workers\":$PIPELINE_WORKERS,\"reason\":\"locomo_finished_first_promote_to_exclusive\"}"
      terminate_tree "$medical_pid"
      wait "$medical_pid" 2>/dev/null || true
      run_medical_pipeline "$served_name" "$base_url" "$model_root" "$EXCLUSIVE_PIPELINE_WORKERS" "exclusive" || status=$?
    else
      wait "$medical_pid" || status=$?
    fi
  fi
  stop_vllm "$model_root"
  if [ "$status" -ne 0 ]; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"stage\":\"pipeline\",\"error\":\"one or more pipelines failed\"}"
    return "$status"
  fi
}

run_wave() {
  local left_model_path="$1"
  local left_name="$2"
  local right_model_path="$3"
  local right_name="$4"
  log "START wave left=$left_name right=$right_name"
  run_model 0 8000 "$left_model_path" "$left_name" &
  local left_pid=$!
  run_model 1 8001 "$right_model_path" "$right_name" &
  local right_pid=$!
  local status=0
  wait "$left_pid" || status=$?
  wait "$right_pid" || status=$?
  log "DONE wave left=$left_name right=$right_name status=$status"
  return "$status"
}

run_wave_pool_then_locomo() {
  local left_model_path="$1"
  local left_name="$2"
  local right_model_path="$3"
  local right_name="$4"
  log "START wave pool-then-locomo left=$left_name right=$right_name workers=$EXCLUSIVE_PIPELINE_WORKERS"
  run_model_pool_locomo_pmoa 0 8000 "$left_model_path" "$left_name" &
  local left_pid=$!
  run_model_pool_locomo_pmoa 1 8001 "$right_model_path" "$right_name" &
  local right_pid=$!
  local status=0
  wait "$left_pid" || status=$?
  wait "$right_pid" || status=$?
  log "DONE wave pool-then-locomo left=$left_name right=$right_name status=$status"
  return "$status"
}

run_model_pool_locomo_pmoa() {
  local gpu="$1"
  local port="$2"
  local model_path="$3"
  local served_name="$4"
  local model_root="$RUN_ROOT/$served_name"
  local base_url="http://127.0.0.1:${port}/v1"
  local status=0
  mkdir -p "$model_root"
  if ! start_vllm "$gpu" "$port" "$model_path" "$served_name" "$model_root"; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"stage\":\"vllm\",\"error\":\"startup failed\"}"
    return 1
  fi
  if ! run_medical_pipeline "$served_name" "$base_url" "$model_root" "$EXCLUSIVE_PIPELINE_WORKERS" "pool_first"; then
    status=1
  elif ! run_locomo_pipeline "$served_name" "$base_url" "$model_root" "$EXCLUSIVE_PIPELINE_WORKERS" "locomo_after_pool"; then
    status=1
  elif ! run_pmoa_ablation_pipeline "$served_name" "$base_url" "$model_root" "$EXCLUSIVE_PIPELINE_WORKERS" "pmoa_after_locomo"; then
    status=1
  fi
  stop_vllm "$model_root" || status=1
  if [ "$status" -ne 0 ]; then
    append_json "$BLOCKED_JSONL" "{\"model\":\"$served_name\",\"stage\":\"pipeline\",\"error\":\"model sequence stopped after failure\"}"
  fi
  return "$status"
}

summarize_and_gate() {
  "$PY" - "$RUN_ROOT" "$STATUS_JSONL" "$BLOCKED_JSONL" "$SUMMARY_JSON" <<'PY'
import csv
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
status_path = pathlib.Path(sys.argv[2])
blocked_path = pathlib.Path(sys.argv[3])
summary_path = pathlib.Path(sys.argv[4])
completed = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()] if status_path.exists() else []
blocked = [json.loads(line) for line in blocked_path.read_text(encoding="utf-8").splitlines() if line.strip()] if blocked_path.exists() else []
baseline_methods = {
    "direct_deepseek",
    "baseline_single_cot_agent",
    "baseline_amem_adapter",
    "baseline_ddo_adapter",
    "baseline_colacare_adapter",
}
rows = []
failures = []
for item in completed:
    if item.get("status") != "finished":
        continue
    if item.get("pipeline") not in {"medical_pooled", "locomo", "pmoa_ablation"}:
        continue
    model = item.get("model")
    run_dir = pathlib.Path(item.get("run_dir") or "")
    pipeline = item.get("pipeline")
    record = {"model": model, "pipeline": pipeline, "run_dir": str(run_dir), "passed": False}
    if not run_dir.exists():
        record["failure"] = "run_dir_missing"
        failures.append(record)
        rows.append(record)
        continue
    fallback_hits = 0
    for pred_path in (run_dir / "predictions").glob("*.jsonl"):
        fallback_hits += sum(1 for line in pred_path.read_text(encoding="utf-8").splitlines() if "fallback_reason" in line or "llm_error" in line)
    record["fallback_hits"] = fallback_hits
    if pipeline == "medical_pooled":
        gate_path = run_dir / "fast_formal_gate.json"
        source_metrics = run_dir / "source_metrics.csv"
        if not gate_path.exists() or not source_metrics.exists():
            record["failure"] = "medical_gate_or_source_metrics_missing"
            failures.append(record)
            rows.append(record)
            continue
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        with source_metrics.open(newline="", encoding="utf-8") as f:
            source_rows = list(csv.DictReader(f))
        overall = [row for row in source_rows if row.get("source") in {"overall", ""}]
        full = next((row for row in overall if row.get("method") == "full_medimem_merged"), None)
        baselines = [row for row in overall if row.get("method") in baseline_methods]
        full_obj = float((full or {}).get("primary_diag_objective") or 0)
        best = max((float(row.get("primary_diag_objective") or 0) for row in baselines), default=0)
        record.update({
            "full_medimem_primary_diag_objective": full_obj,
            "best_baseline_primary_diag_objective": best,
            "critical_leakage_count": int(gate.get("critical_leakage_count", 0) or 0),
            "needs_review_count": int(gate.get("needs_review_count", 0) or 0),
            "progress_failed": int(gate.get("progress_failed", 0) or 0),
            "blocked_sources": len(gate.get("blocked_sources") or []),
            "passed": bool(gate.get("passed")) and fallback_hits == 0 and full_obj > best,
        })
    elif pipeline == "locomo":
        manifest_path = run_dir / "benchmark_manifest.json"
        metrics_path = run_dir / "locomo_metrics_overall.csv"
        if not manifest_path.exists() or not metrics_path.exists():
            record["failure"] = "locomo_manifest_or_metrics_missing"
            failures.append(record)
            rows.append(record)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with metrics_path.open(newline="", encoding="utf-8") as f:
            metrics = list(csv.DictReader(f))
        medimem = max(
            [row for row in metrics if str(row.get("method", "")).startswith(("medimem_locomo_memory_pipeline", "ours_locomo_memory_pipeline"))],
            key=lambda row: float(row.get("qa_f1") or 0),
            default=None,
        )
        amem = next((row for row in metrics if row.get("method") in {"source_aligned_amem_adapter", "official_amem_locomo_wrapper"}), None)
        medimem_f1 = float((medimem or {}).get("qa_f1") or 0)
        amem_f1 = float((amem or {}).get("qa_f1") or 0)
        record.update({
            "medimem_qa_f1": medimem_f1,
            "amem_qa_f1": amem_f1,
            "validation_passed": bool((manifest.get("validation") or {}).get("passed")),
            "blocked_methods": len(manifest.get("blocked_methods") or []),
            "passed": bool((manifest.get("validation") or {}).get("passed")) and fallback_hits == 0 and medimem_f1 > amem_f1,
        })
    else:
        gate_path = run_dir / "fast_formal_gate.json"
        delta_path = run_dir / "medimem_ablation_delta.csv"
        source_metrics = run_dir / "source_metrics.csv"
        if not gate_path.exists() or not delta_path.exists() or not source_metrics.exists():
            record["failure"] = "pmoa_gate_or_ablation_outputs_missing"
            failures.append(record)
            rows.append(record)
            continue
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        with source_metrics.open(newline="", encoding="utf-8") as f:
            source_rows = list(csv.DictReader(f))
        observed_methods = {row.get("method") for row in source_rows if row.get("source") in {"overall", ""}}
        expected_methods = {
            "full_medimem_merged",
            "ablate_no_memory_cleaning_medimem_merged",
            "ablate_no_evidence_note_injection_medimem_merged",
            "ablate_with_polluted_memory_medimem_merged",
        }
        record.update({
            "critical_leakage_count": int(gate.get("critical_leakage_count", 0) or 0),
            "needs_review_count": int(gate.get("needs_review_count", 0) or 0),
            "progress_failed": int(gate.get("progress_failed", 0) or 0),
            "blocked_sources": len(gate.get("blocked_sources") or []),
            "expected_pmoa_methods_present": expected_methods.issubset(observed_methods),
            "passed": bool(gate.get("passed")) and fallback_hits == 0 and expected_methods.issubset(observed_methods),
        })
    if not record["passed"]:
        failures.append(record)
    rows.append(record)
summary = {
    "run_root": str(run_root),
    "blocked": blocked,
    "rows": rows,
    "passed": not blocked and bool(rows) and not failures,
    "failures": failures,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
csv_path = summary_path.with_suffix(".csv")
fieldnames = sorted({key for row in rows for key in row})
if fieldnames:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
if not summary["passed"]:
    raise SystemExit("strict size sweep gate failed; see " + str(summary_path))
PY
}

log "strict size sweep started run_root=$RUN_ROOT data_root=$DATA_ROOT workers_per_pipeline=$PIPELINE_WORKERS exclusive_workers=$EXCLUSIVE_PIPELINE_WORKERS schedule=$PIPELINE_SCHEDULE"
overall_status=0
if [ "$PIPELINE_SCHEDULE" = "pool_then_locomo" ]; then
  run_wave_pool_then_locomo "/root/models/qwen_vl_size_sweep/2b" "qwen3-vl-2b" "/root/models/qwen_vl_size_sweep/0_8b" "qwen3-vl-0_8b" || overall_status=$?
  run_wave_pool_then_locomo "/root/models/qwen_vl_size_sweep/4b" "qwen3-vl-4b" "/root/models/qwen_vl_size_sweep/8b" "qwen3-vl-8b" || overall_status=$?
else
  run_wave "/root/models/qwen_vl_size_sweep/2b" "qwen3-vl-2b" "/root/models/qwen_vl_size_sweep/0_8b" "qwen3-vl-0_8b" || overall_status=$?
  run_wave "/root/models/qwen_vl_size_sweep/4b" "qwen3-vl-4b" "/root/models/qwen_vl_size_sweep/8b" "qwen3-vl-8b" || overall_status=$?
fi
summarize_and_gate
log "FINISHED strict size sweep summary=$SUMMARY_JSON"
exit "$overall_status"
