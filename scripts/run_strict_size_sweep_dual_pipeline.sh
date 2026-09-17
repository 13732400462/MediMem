#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/syh/A-mem}"
PY="${PY:-/root/miniconda3/envs/llmserve/bin/python}"
VLLM_PY="${VLLM_PY:-/root/miniconda3/envs/llmserve/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/syh/mem_ehr_hf_cache}"
RUN_ROOT="${RUN_ROOT:-runs/strict_size_sweep_dual_pipeline_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-data/processed/strict_size_sweep_dual_pipeline_$(date +%Y%m%d_%H%M%S)}"
MEDICAL_SOURCES="${MEDICAL_SOURCES:-medmcqa,medqa,chatdoctor_healthcaremagic,medical_meadow_wikidoc,medical_dialogue_to_soap,pmoa_tts,pmc_patients}"
PER_SOURCE_N="${PER_SOURCE_N:-1000}"
LOCOMO_SAMPLE_N="${LOCOMO_SAMPLE_N:-1000}"
RANDOM_SEED="${RANDOM_SEED:-20260606}"
PIPELINE_WORKERS="${PIPELINE_WORKERS:-48}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-96}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-600}"
DEEPSEEK_MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-900}"
LOCOMO_DATASET_PATH="${LOCOMO_DATASET_PATH:-datasets/amem_original/locomo/locomo10.official.json}"
LOCOMO_METHODS="${LOCOMO_METHODS:-direct,amem,medimem}"
MEDICAL_ABLATION_GROUPS="${MEDICAL_ABLATION_GROUPS:-full,no_memory_cleaning,no_evidence_note_injection,with_polluted_memory}"

cd "$PROJECT_ROOT"
export PYTHONPATH=.
export MEM_EHR_DATA_CACHE_DIR="$CACHE_DIR"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-local-qwen3}"
export DEEPSEEK_TIMEOUT
export DEEPSEEK_MAX_TOKENS
export FAST_FORMAL_EARLY_STOP_ON_WIN=0
export MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER="${MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER:-5}"
export MEDICAL_STRICT_NO_LEAK_FILTER=1

mkdir -p "$RUN_ROOT" "$DATA_ROOT" "$CACHE_DIR"
STATUS_JSONL="$RUN_ROOT/status.jsonl"
BLOCKED_JSONL="$RUN_ROOT/blocked_sources.jsonl"
SUMMARY_JSON="$RUN_ROOT/strict_size_sweep_summary.json"
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
      --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
      --guided-decoding-backend auto \
      > "$model_root/vllm.log" 2>&1 &
  local pid=$!
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
    if kill -0 "$pid" 2>/dev/null; then
      log "STOP vLLM pid=$pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
}

run_medical_pipeline() {
  local served_name="$1"
  local base_url="$2"
  local model_root="$3"
  local data_dir="$DATA_ROOT/$served_name/medical_pooled"
  local run_dir="$model_root/medical_pooled"
  mkdir -p "$data_dir" "$run_dir"
  export DEEPSEEK_BASE_URL="$base_url"
  export DEEPSEEK_MODEL="$served_name"

  log "START medical model=$served_name workers=$PIPELINE_WORKERS per_source_n=$PER_SOURCE_N"
  "$PY" -m medimem data build-medical-pool \
    --per-source-n "$PER_SOURCE_N" \
    --sources "$MEDICAL_SOURCES" \
    --output-dir "$data_dir" \
    --require-real-data \
    --cache-dir "$CACHE_DIR" \
    --random-seed "$RANDOM_SEED" \
    > "$run_dir/build.log" 2>&1

  local dataset_path
  dataset_path="$("$PY" - "$data_dir/manifest.json" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["pooled_path"])
PY
)"
  "$PY" -m medimem data validate --dataset "$dataset_path" > "$run_dir/validate.log" 2>&1
  "$PY" -u -m medimem experiment-suite \
    --dataset "$dataset_path" \
    --require-api \
    --max-workers "$PIPELINE_WORKERS" \
    --suite-profile fast-formal \
    --baseline-set required \
    --ablation-groups "$MEDICAL_ABLATION_GROUPS" \
    --counterfactual-policy risk_sample \
    --counterfactual-sample-rate "${COUNTERFACTUAL_SAMPLE_RATE:-0.20}" \
    --counterfactual-risk-threshold "${COUNTERFACTUAL_RISK_THRESHOLD:-0.55}" \
    --defer-reports \
    2>&1 | tee "$run_dir/experiment.log"
  local experiment_run_dir
  experiment_run_dir="$(grep -E '^run_dir=' "$run_dir/experiment.log" | tail -1 | cut -d= -f2-)"
  append_json "$STATUS_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"medical_pooled\",\"status\":\"finished\",\"dataset_path\":\"$dataset_path\",\"run_dir\":\"$experiment_run_dir\",\"base_url\":\"$base_url\",\"workers\":$PIPELINE_WORKERS}"
}

run_locomo_pipeline() {
  local served_name="$1"
  local base_url="$2"
  local model_root="$3"
  local run_dir="$model_root/locomo"
  mkdir -p "$run_dir"
  export DEEPSEEK_BASE_URL="$base_url"
  export DEEPSEEK_MODEL="$served_name"

  log "START locomo model=$served_name workers=$PIPELINE_WORKERS sample_n=$LOCOMO_SAMPLE_N"
  "$PY" -u -m medimem benchmark run \
    --dataset locomo \
    --methods "$LOCOMO_METHODS" \
    --dataset-path "$LOCOMO_DATASET_PATH" \
    --sample-n "$LOCOMO_SAMPLE_N" \
    --random-seed "$RANDOM_SEED" \
    --max-workers "$PIPELINE_WORKERS" \
    --require-api \
    --output-root "$run_dir" \
    2>&1 | tee "$run_dir/benchmark.log"
  local benchmark_run_dir
  benchmark_run_dir="$(grep -E '^run_dir=' "$run_dir/benchmark.log" | tail -1 | cut -d= -f2-)"
  append_json "$STATUS_JSONL" "{\"model\":\"$served_name\",\"pipeline\":\"locomo\",\"status\":\"finished\",\"dataset_path\":\"$LOCOMO_DATASET_PATH\",\"run_dir\":\"$benchmark_run_dir\",\"base_url\":\"$base_url\",\"workers\":$PIPELINE_WORKERS}"
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
  run_medical_pipeline "$served_name" "$base_url" "$model_root" &
  local medical_pid=$!
  run_locomo_pipeline "$served_name" "$base_url" "$model_root" &
  local locomo_pid=$!
  local status=0
  wait "$medical_pid" || status=$?
  wait "$locomo_pid" || status=$?
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
    if item.get("pipeline") not in {"medical_pooled", "locomo"}:
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
    else:
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
if not summary["passed"]:
    raise SystemExit("strict size sweep gate failed; see " + str(summary_path))
PY
}

log "strict size sweep started run_root=$RUN_ROOT data_root=$DATA_ROOT workers_per_pipeline=$PIPELINE_WORKERS"
run_wave "/root/models/qwen_vl_size_sweep/0_8b" "qwen3-vl-0_8b" "/root/models/qwen_vl_size_sweep/2b" "qwen3-vl-2b"
run_wave "/root/models/qwen_vl_size_sweep/4b" "qwen3-vl-4b" "/root/models/qwen_vl_size_sweep/8b" "qwen3-vl-8b"
summarize_and_gate
log "FINISHED strict size sweep summary=$SUMMARY_JSON"

