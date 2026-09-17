#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/syh/A-mem}"
PY="${PY:-/root/miniconda3/envs/llmserve/bin/python}"
VLLM_PY="${VLLM_PY:-/root/miniconda3/envs/llmserve/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/syh/mem_ehr_hf_cache}"
RUN_ROOT="${RUN_ROOT:-runs/gpu0_gap_fill_temporal_ablation_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-data/processed/gpu0_gap_fill_temporal_ablation_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
PER_SOURCE_N="${PER_SOURCE_N:-1000}"
LOCOMO_SAMPLE_N="${LOCOMO_SAMPLE_N:-1000}"
RANDOM_SEED="${RANDOM_SEED:-20260606}"
PRIMARY_WORKERS="${PRIMARY_WORKERS:-128}"
FALLBACK_WORKERS="${FALLBACK_WORKERS:-96}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-600}"
DEEPSEEK_MAX_TOKENS="${DEEPSEEK_MAX_TOKENS:-1800}"
BENCHMARK_MAX_TOKENS="${BENCHMARK_MAX_TOKENS:-256}"
MEDICAL_PREDICTION_MAX_TOKENS="${MEDICAL_PREDICTION_MAX_TOKENS:-128}"
MEDICAL_JSON_PARSE_RETRIES="${MEDICAL_JSON_PARSE_RETRIES:-2}"
LOCOMO_DATASET_PATH="${LOCOMO_DATASET_PATH:-datasets/amem_original/locomo/locomo10.official.json}"
LOCOMO_METHODS="${LOCOMO_METHODS:-direct,amem,medimem}"
PMOA_ABLATION_GROUPS="${PMOA_ABLATION_GROUPS:-full,no_temporal_signal}"

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
BLOCKED_JSONL="$RUN_ROOT/blocked.jsonl"
SUMMARY_JSON="$RUN_ROOT/gap_fill_summary.json"
if [ "${RESUME:-0}" = "1" ]; then
  touch "$STATUS_JSONL" "$BLOCKED_JSONL" "$RUN_ROOT/run.log"
else
  : > "$STATUS_JSONL"
  : > "$BLOCKED_JSONL"
  : > "$RUN_ROOT/run.log"
fi

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

status_finished() {
  local stage="$1"
  local model="$2"
  "$PY" - "$STATUS_JSONL" "$stage" "$model" <<'PY'
import json
import sys
path, stage, model = sys.argv[1:4]
try:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("stage") == stage and item.get("model") == model and item.get("status") == "finished":
                raise SystemExit(0)
except FileNotFoundError:
    pass
raise SystemExit(1)
PY
}

gpu_compute_pids() {
  nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
}

assert_gpu0_available() {
  log "GPU status before run"
  nvidia-smi | tee -a "$RUN_ROOT/run.log"
  gpu_compute_pids > "$RUN_ROOT/gpu_compute_apps_before.csv"
  if [ -s "$RUN_ROOT/gpu_compute_apps_before.csv" ]; then
    log "GPU compute processes detected before launch; refusing to touch existing processes"
    cat "$RUN_ROOT/gpu_compute_apps_before.csv" | tee -a "$RUN_ROOT/run.log"
    append_json "$BLOCKED_JSONL" '{"stage":"preflight","error":"gpu_compute_processes_exist_before_launch"}'
    exit 1
  fi
}

wait_for_model() {
  local base_url="$1"
  local served_name="$2"
  for _ in $(seq 1 180); do
    if "$PY" - "$base_url" "$served_name" <<'PY'
import json
import sys
import urllib.request
base_url, served_name = sys.argv[1].rstrip("/"), sys.argv[2]
try:
    with urllib.request.urlopen(base_url + "/models", timeout=3) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    ids = [item.get("id") for item in payload.get("data", [])]
    raise SystemExit(0 if served_name in ids else 1)
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

start_vllm_once() {
  local model_path="$1"
  local served_name="$2"
  local workers="$3"
  local model_root="$4"
  mkdir -p "$model_root"
  log "START vLLM gpu=$GPU_ID port=$PORT model=$served_name workers=$workers"
  CUDA_VISIBLE_DEVICES="$GPU_ID" VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V1="${VLLM_USE_V1:-0}" \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 \
      --port "$PORT" \
      --model "$model_path" \
      --served-model-name "$served_name" \
      --trust-remote-code \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --enforce-eager \
      --max-num-seqs "$workers" \
      > "$model_root/vllm_${workers}.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$model_root/vllm.pid"
  if wait_for_model "http://127.0.0.1:${PORT}/v1" "$served_name"; then
    append_json "$STATUS_JSONL" "{\"stage\":\"vllm\",\"status\":\"ready\",\"model\":\"$served_name\",\"pid\":$pid,\"workers\":$workers,\"base_url\":\"http://127.0.0.1:${PORT}/v1\"}"
    return 0
  fi
  log "vLLM healthcheck failed model=$served_name workers=$workers pid=$pid"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  return 1
}

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local model_root="$3"
  if start_vllm_once "$model_path" "$served_name" "$PRIMARY_WORKERS" "$model_root"; then
    echo "$PRIMARY_WORKERS" > "$model_root/effective_workers.txt"
    return 0
  fi
  log "Retry vLLM with fallback workers=$FALLBACK_WORKERS model=$served_name"
  if start_vllm_once "$model_path" "$served_name" "$FALLBACK_WORKERS" "$model_root"; then
    echo "$FALLBACK_WORKERS" > "$model_root/effective_workers.txt"
    append_json "$STATUS_JSONL" "{\"stage\":\"vllm\",\"status\":\"fallback_workers\",\"model\":\"$served_name\",\"workers\":$FALLBACK_WORKERS,\"reason\":\"primary_workers_failed\"}"
    return 0
  fi
  append_json "$BLOCKED_JSONL" "{\"stage\":\"vllm\",\"model\":\"$served_name\",\"error\":\"healthcheck_failed_primary_and_fallback\"}"
  return 1
}

stop_vllm() {
  local model_root="$1"
  if [ -f "$model_root/vllm.pid" ]; then
    local pid
    pid="$(cat "$model_root/vllm.pid")"
    if ps -p "$pid" -o args= | grep -q "vllm.entrypoints.openai.api_server"; then
      log "STOP own vLLM pid=$pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  fi
}

run_8b_locomo() {
  local served_name="qwen3-vl-8b"
  local model_root="$RUN_ROOT/$served_name"
  local workers
  workers="$(cat "$model_root/effective_workers.txt")"
  export DEEPSEEK_BASE_URL="http://127.0.0.1:${PORT}/v1"
  export DEEPSEEK_MODEL="$served_name"
  log "START 8B LoCoMo sample_n=$LOCOMO_SAMPLE_N workers=$workers"
  local out="$model_root/locomo_gap_fill"
  mkdir -p "$out"
  "$PY" -u -m medimem benchmark run \
    --dataset locomo \
    --methods "$LOCOMO_METHODS" \
    --dataset-path "$LOCOMO_DATASET_PATH" \
    --sample-n "$LOCOMO_SAMPLE_N" \
    --random-seed "$RANDOM_SEED" \
    --max-workers "$workers" \
    --require-api \
    --output-root "$out" \
    2>&1 | tee "$out/benchmark.log"
  local run_dir
  run_dir="$(grep -E '^run_dir=' "$out/benchmark.log" | tail -1 | cut -d= -f2-)"
  append_json "$STATUS_JSONL" "{\"stage\":\"locomo\",\"status\":\"finished\",\"model\":\"$served_name\",\"run_dir\":\"$run_dir\",\"workers\":$workers}"
}

run_pmoa_temporal_ablation() {
  local served_name="$1"
  local model_root="$RUN_ROOT/$served_name"
  local workers
  workers="$(cat "$model_root/effective_workers.txt")"
  local data_dir="$DATA_ROOT/$served_name/pmoa_tts"
  local out="$model_root/pmoa_temporal_ablation"
  mkdir -p "$data_dir" "$out"
  export DEEPSEEK_BASE_URL="http://127.0.0.1:${PORT}/v1"
  export DEEPSEEK_MODEL="$served_name"
  log "BUILD PMOA model=$served_name n=$PER_SOURCE_N"
  "$PY" -m medimem data build-medical-pool \
    --per-source-n "$PER_SOURCE_N" \
    --sources pmoa_tts \
    --output-dir "$data_dir" \
    --require-real-data \
    --cache-dir "$CACHE_DIR" \
    --random-seed "$RANDOM_SEED" \
    > "$out/build.log" 2>&1
  local dataset_path
  dataset_path="$("$PY" - "$data_dir/manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["pooled_path"])
PY
)"
  "$PY" -m medimem data validate --dataset "$dataset_path" > "$out/validate.log" 2>&1
  log "START PMOA temporal ablation model=$served_name workers=$workers groups=$PMOA_ABLATION_GROUPS"
  "$PY" -u -m medimem experiment-suite \
    --dataset "$dataset_path" \
    --require-api \
    --max-workers "$workers" \
    --suite-profile fast-formal \
    --baseline-set required \
    --ablation-groups "$PMOA_ABLATION_GROUPS" \
    --counterfactual-policy risk_sample \
    --counterfactual-sample-rate "${COUNTERFACTUAL_SAMPLE_RATE:-0.20}" \
    --counterfactual-risk-threshold "${COUNTERFACTUAL_RISK_THRESHOLD:-0.55}" \
    --defer-reports \
    2>&1 | tee "$out/experiment.log"
  local run_dir
  run_dir="$(grep -E '^run_dir=' "$out/experiment.log" | tail -1 | cut -d= -f2-)"
  append_json "$STATUS_JSONL" "{\"stage\":\"pmoa_temporal_ablation\",\"status\":\"finished\",\"model\":\"$served_name\",\"dataset_path\":\"$dataset_path\",\"run_dir\":\"$run_dir\",\"workers\":$workers}"
}

summarize() {
  "$PY" - "$RUN_ROOT" "$STATUS_JSONL" "$BLOCKED_JSONL" "$SUMMARY_JSON" <<'PY'
import csv
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
status_path = pathlib.Path(sys.argv[2])
blocked_path = pathlib.Path(sys.argv[3])
summary_path = pathlib.Path(sys.argv[4])
statuses = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()]
blocked = [json.loads(line) for line in blocked_path.read_text(encoding="utf-8").splitlines() if line.strip()]
rows = []
failures = []

def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

for item in statuses:
    if item.get("status") != "finished":
        continue
    stage = item.get("stage")
    model = item.get("model")
    run_dir = pathlib.Path(item.get("run_dir") or "")
    record = {"stage": stage, "model": model, "run_dir": str(run_dir), "passed": False}
    if not run_dir.exists():
        record["failure"] = "run_dir_missing"
        failures.append(record)
        rows.append(record)
        continue
    fallback_hits = 0
    pred_dir = run_dir / "predictions"
    if pred_dir.exists():
        for pred_path in pred_dir.glob("*.jsonl"):
            fallback_hits += sum(1 for line in pred_path.read_text(encoding="utf-8").splitlines() if "fallback_reason" in line or "llm_error" in line)
    record["fallback_hits"] = fallback_hits
    if stage == "locomo":
        manifest_path = run_dir / "benchmark_manifest.json"
        overall = read_csv(run_dir / "locomo_metrics_overall.csv")
        blocked_methods = [line for line in (run_dir / "blocked_methods.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (run_dir / "blocked_methods.jsonl").exists() else []
        medimem = next((row for row in overall if str(row.get("method", "")).startswith("medimem_locomo_memory_pipeline")), {})
        amem = next((row for row in overall if row.get("method") in {"official_amem_locomo_wrapper", "source_aligned_amem_adapter"}), {})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        record.update({
            "n": medimem.get("n"),
            "medimem_qa_f1": medimem.get("qa_f1"),
            "amem_qa_f1": amem.get("qa_f1"),
            "validation_passed": bool((manifest.get("validation") or {}).get("passed")),
            "blocked_methods": len(blocked_methods),
        })
        record["passed"] = record["validation_passed"] and record["blocked_methods"] == 0 and fallback_hits == 0
    elif stage == "pmoa_temporal_ablation":
        gate = json.loads((run_dir / "fast_formal_gate.json").read_text(encoding="utf-8")) if (run_dir / "fast_formal_gate.json").exists() else {}
        deltas = read_csv(run_dir / "medimem_ablation_delta.csv")
        overall = next((row for row in deltas if row.get("source") == "overall" and row.get("method") == "ablate_no_temporal_signal_medimem_merged"), {})
        full = next((row for row in deltas if row.get("source") == "overall" and row.get("method") == "full_medimem_merged"), {})
        record.update({
            "n": overall.get("n") or full.get("n"),
            "full_primary_diag_objective": full.get("primary_diag_objective"),
            "no_temporal_primary_diag_objective": overall.get("primary_diag_objective"),
            "delta_primary_diag_objective_vs_full": overall.get("delta_primary_diag_objective_vs_full"),
            "critical_leakage_count": gate.get("critical_leakage_count"),
            "needs_review_count": gate.get("needs_review_count"),
            "progress_failed": gate.get("progress_failed"),
        })
        record["passed"] = (
            fallback_hits == 0
            and int(gate.get("critical_leakage_count", 0) or 0) == 0
            and int(gate.get("needs_review_count", 0) or 0) == 0
            and int(gate.get("progress_failed", 0) or 0) == 0
            and bool(overall)
        )
    if not record["passed"]:
        failures.append(record)
    rows.append(record)

classification = [
    {"category": "多选/考试问答源", "sources": "medmcqa, medqa", "reason": "检验医学知识选择、标准化答案粒度和干扰项鲁棒性。"},
    {"category": "开放问答/知识问答源", "sources": "medical_meadow_wikidoc, chatdoctor_healthcaremagic", "reason": "检验开放式医学问题理解、知识检索式回答和短实体抽取。"},
    {"category": "纵向病例/时间序列源", "sources": "pmoa_tts", "reason": "检验跨时间事件、复诊轨迹、记忆更新和时间信号贡献。"},
    {"category": "患者病例摘要源", "sources": "pmc_patients", "reason": "检验真实病例文本到诊断/病情概括的迁移能力。"},
    {"category": "长期记忆 QA benchmark", "sources": "LoCoMo", "reason": "检验通用长对话记忆检索与问答能力，不与医疗污染清洗能力混同解释。"},
]
summary = {
    "run_root": str(run_root),
    "passed": not blocked and bool(rows) and not failures,
    "blocked": blocked,
    "failures": failures,
    "rows": rows,
    "data_source_classification": classification,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = ["# GPU0 Gap Fill and PMOA Temporal Ablation Summary", "", "## Run Results", "", "| Stage | Model | N | Main metric | Delta | Passed | Run dir |", "|---|---|---:|---:|---:|---|---|"]
for row in rows:
    if row["stage"] == "locomo":
        main = row.get("medimem_qa_f1", "")
        delta = ""
    else:
        main = row.get("no_temporal_primary_diag_objective", "")
        delta = row.get("delta_primary_diag_objective_vs_full", "")
    lines.append(f"| {row.get('stage')} | {row.get('model')} | {row.get('n','')} | {main} | {delta} | {row.get('passed')} | `{row.get('run_dir')}` |")
lines.extend(["", "## 数据源任务类型分类", ""])
for item in classification:
    lines.append(f"- **{item['category']}**：`{item['sources']}`。{item['reason']}")
lines.extend(["", "## 为什么构建多类数据源", "", "这些来源覆盖选择题、开放问答、医患对话、病例摘要、纵向时间序列和通用长期记忆 QA，能避免只在单一数据形态上证明方法有效，也能区分知识问答能力、结构化临床抽取能力、长期记忆检索能力和时间序列记忆更新能力。"])
(run_root / "gap_fill_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
if not summary["passed"]:
    raise SystemExit("gap fill summary has failures; see " + str(summary_path))
PY
}

MODELS=(
  "qwen3-vl-8b:/root/models/qwen_vl_size_sweep/8b:locomo,pmoa"
  "qwen3-vl-0_8b:/root/models/qwen_vl_size_sweep/0_8b:pmoa"
  "qwen3-vl-2b:/root/models/qwen_vl_size_sweep/2b:pmoa"
  "qwen3-vl-4b:/root/models/qwen_vl_size_sweep/4b:pmoa"
)

assert_gpu0_available
for spec in "${MODELS[@]}"; do
  IFS=: read -r served_name model_path stages <<< "$spec"
  model_root="$RUN_ROOT/$served_name"
  mkdir -p "$model_root"
  start_vllm "$model_path" "$served_name" "$model_root"
  if [[ "$stages" == *"locomo"* ]]; then
    if status_finished "locomo" "$served_name"; then
      log "SKIP completed 8B LoCoMo model=$served_name"
    else
      run_8b_locomo
    fi
  fi
  if [[ "$stages" == *"pmoa"* ]]; then
    if status_finished "pmoa_temporal_ablation" "$served_name"; then
      log "SKIP completed PMOA temporal ablation model=$served_name"
    else
      run_pmoa_temporal_ablation "$served_name"
    fi
  fi
  stop_vllm "$model_root"
  sleep 10
done
summarize
log "FINISHED summary=$SUMMARY_JSON report=$RUN_ROOT/gap_fill_summary.md"

