#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/home/syh/A-mem}"
PY="${PY:-/root/miniconda3/envs/llmserve/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/syh/mem_ehr_hf_cache}"
DATA_ROOT="${DATA_ROOT:-data/processed/medical_sources_all_baselines_100_20260603}"
RUN_ROOT="${RUN_ROOT:-runs/medical_sources_all_baselines_100_20260603_medimem}"
MAX_WORKERS="${MAX_WORKERS:-96}"
PER_SOURCE_N="${PER_SOURCE_N:-100}"
REPORT_STEM="${REPORT_STEM:-medical_sources_all_baselines_100_medimem}"

if [ -n "${SOURCES_CSV:-}" ]; then
  IFS=',' read -r -a SOURCES <<< "$SOURCES_CSV"
else
  SOURCES=(
    medical_meadow_wikidoc
    medmcqa
    medqa
    chatdoctor_healthcaremagic
    pmoa_tts
  )
fi

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH=.
export MEM_EHR_DATA_CACHE_DIR="$CACHE_DIR"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-http://127.0.0.1:8000/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-/home/yyh/Qwen3-VL-30B-A3B-Instruct-FP8}"
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
import json, sys
path, payload = sys.argv[1], sys.argv[2]
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(json.loads(payload), ensure_ascii=False, sort_keys=True) + "\n")
PY
}

json_quote() {
  "$PY" -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

summarize_results() {
  "$PY" - "$RUN_ROOT" "$PER_SOURCE_N" "$MAX_WORKERS" <<'PY'
import csv
import json
import os
import pathlib
import sys
import time

run_root = pathlib.Path(sys.argv[1])
per_source_n = int(sys.argv[2])
max_workers = int(sys.argv[3])
status_path = run_root / "source_status.jsonl"
blocked_path = run_root / "blocked_sources.jsonl"
completed = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()] if status_path.exists() else []
blocked = [json.loads(line) for line in blocked_path.read_text(encoding="utf-8").splitlines() if line.strip()] if blocked_path.exists() else []

method_alias = {
    "direct_deepseek": "Direct",
    "baseline_polluted_direct_deepseek": "Polluted Direct",
    "baseline_single_cot_agent": "Single-agent CoT",
    "baseline_polluted_single_cot_agent": "Polluted Single-agent CoT",
    "baseline_amem_adapter": "A-MEM",
    "baseline_polluted_amem_adapter": "Polluted A-MEM",
    "baseline_ddo_adapter": "DDO",
    "baseline_polluted_ddo_adapter": "Polluted DDO",
    "baseline_colacare_adapter": "ColaCare",
    "baseline_polluted_colacare_adapter": "Polluted ColaCare",
    "full_medimem_merged": "Full Ours",
    "ablate_no_dynamic_top_k_medimem_merged": "w/o Dynamic top-k",
    "ablate_no_normalization_medimem_merged": "w/o Diagnosis Normalization",
    "ablate_no_memory_cleaning_medimem_merged": "w/o Memory Cleaning",
    "ablate_no_critic_op_guard_medimem_merged": "w/o Critic Op Guard",
    "ablate_no_evidence_note_injection_medimem_merged": "w/o Evidence Note Injection",
    "ablate_no_counterfactual_verification_medimem_merged": "w/o Counterfactual Verification",
    "full_ours_merged": "Full Ours (legacy)",
}
columns = [
    "dataset",
    "pipeline",
    "method",
    "n",
    "primary_diagnosis_top1_accuracy",
    "diagnosis_list_f1",
    "cdr_f1",
    "memory_pollution_control_score",
    "counterfactual_probability_gap",
    "counterfactual_pass_rate",
    "counterfactual_revision_rate",
    "stale_memory_action_accuracy",
    "revision_accuracy",
    "fact_preservation_soft",
    "over_deletion_rate",
    "avg_tokens",
    "run_dir",
]
rows = []
for item in completed:
    metrics_path = pathlib.Path(item.get("copied_run_dir") or item.get("run_dir") or "") / "metrics.csv"
    if not metrics_path.exists():
        continue
    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row.get("method", "")
            if method not in method_alias:
                continue
            out = {col: row.get(col, "") for col in columns}
            out["dataset"] = item["source"]
            out["pipeline"] = method_alias[method]
            out["method"] = method
            out["run_dir"] = str(metrics_path.parent)
            rows.append(out)

report_stem = os.environ.get("REPORT_STEM", "medical_sources_all_baselines_100_medimem")
csv_path = run_root / f"{report_stem}_comparison.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value, digits=3):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)

md_path = run_root / f"{report_stem}_comparison.md"
md_cols = [
    ("dataset", "Dataset"),
    ("pipeline", "Pipeline"),
    ("n", "N"),
    ("primary_diagnosis_top1_accuracy", "Primary Acc"),
    ("diagnosis_list_f1", "Diagnosis F1"),
    ("cdr_f1", "CDR F1"),
    ("memory_pollution_control_score", "MPCS"),
    ("counterfactual_probability_gap", "CPG"),
    ("counterfactual_pass_rate", "CF Pass"),
    ("counterfactual_revision_rate", "CF Revision"),
    ("avg_tokens", "Avg Tokens"),
]
lines = [
    f"# Medical Sources N={per_source_n} All-Baselines + MediMem",
    "",
    f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"- Completed sources: {len(completed)}",
    f"- Blocked attempts: {len(blocked)}",
    f"- Max workers: {max_workers}",
    f"- Per-source N: {per_source_n}",
    "",
    "| " + " | ".join(label for _, label in md_cols) + " |",
    "|" + "|".join("---" if idx < 2 else "---:" for idx, _ in enumerate(md_cols)) + "|",
]
for row in rows:
    vals = []
    for key, _ in md_cols:
        if key in {"dataset", "pipeline"}:
            vals.append(str(row.get(key, "")))
        elif key == "n":
            vals.append(fmt(row.get(key), 0))
        elif key == "avg_tokens":
            vals.append(fmt(row.get(key), 1))
        else:
            vals.append(fmt(row.get(key), 3))
    lines.append("| " + " | ".join(vals) + " |")
if blocked:
    lines.extend(["", "## Blocked Attempts"])
    for item in blocked:
        lines.append(f"- {item.get('source')}: {item.get('stage')} failed; {str(item.get('error', ''))[:300]}")
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest = {
    "track": "medical_sources_all_baselines_100_medimem",
    "per_source_n": per_source_n,
    "max_workers": max_workers,
    "require_real_data": True,
    "counterfactual_policy": "MediMem counterfactual verification requires real OpenAI-compatible API; no offline fallback.",
    "completed_sources": completed,
    "blocked_sources": blocked,
    "comparison_csv": str(csv_path),
    "comparison_md": str(md_path),
}
(run_root / "medical_sources_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_source() {
  local source="$1"
  local source_dir="$RUN_ROOT/$source"
  local data_dir="$DATA_ROOT/$source"
  local dataset_path="$data_dir/${source}_${PER_SOURCE_N}.jsonl"
  mkdir -p "$source_dir" "$data_dir"

  log "START source=$source build n=$PER_SOURCE_N"
  if ! "$PY" -m mem_ehr_agent data build-medical-pool \
      --per-source-n "$PER_SOURCE_N" \
      --sources "$source" \
      --output-dir "$data_dir" \
      --require-real-data \
      --cache-dir "$CACHE_DIR" \
      > "$source_dir/build.log" 2>&1; then
    local err
    err="$(tail -40 "$source_dir/build.log" | tr '\n' ' ')"
    log "BLOCKED source=$source stage=build"
    append_json "$BLOCKED_JSONL" "{\"source\":\"$source\",\"stage\":\"build\",\"error\":$(printf '%s' "$err" | json_quote)}"
    summarize_results
    return 1
  fi

  log "START source=$source validate"
  if ! "$PY" -m mem_ehr_agent data validate --dataset "$dataset_path" > "$source_dir/validate.log" 2>&1; then
    local err
    err="$(tail -40 "$source_dir/validate.log" | tr '\n' ' ')"
    log "BLOCKED source=$source stage=validate"
    append_json "$BLOCKED_JSONL" "{\"source\":\"$source\",\"stage\":\"validate\",\"error\":$(printf '%s' "$err" | json_quote)}"
    summarize_results
    return 1
  fi

  log "START source=$source experiment max_workers=$MAX_WORKERS"
  if ! "$PY" -m mem_ehr_agent experiment-suite \
      --dataset "$dataset_path" \
      --require-api \
      --max-workers "$MAX_WORKERS" \
      > "$source_dir/experiment.log" 2>&1; then
    local err
    err="$(tail -80 "$source_dir/experiment.log" | tr '\n' ' ')"
    log "BLOCKED source=$source stage=experiment"
    append_json "$BLOCKED_JSONL" "{\"source\":\"$source\",\"stage\":\"experiment\",\"error\":$(printf '%s' "$err" | json_quote)}"
    summarize_results
    return 1
  fi

  local run_dir
  run_dir="$(grep -E '^run_dir=' "$source_dir/experiment.log" | tail -1 | cut -d= -f2-)"
  if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    log "BLOCKED source=$source stage=run_dir"
    append_json "$BLOCKED_JSONL" "{\"source\":\"$source\",\"stage\":\"run_dir\",\"error\":\"experiment completed but run_dir was not found\"}"
    summarize_results
    return 1
  fi

  local copied_run_dir="$source_dir/$(basename "$run_dir")"
  cp -a "$run_dir" "$copied_run_dir"
  cp -f "$data_dir/label_quality_report.csv" "$source_dir/label_quality_report.csv" 2>/dev/null || true
  cp -f "$data_dir/label_quality_summary.json" "$source_dir/label_quality_summary.json" 2>/dev/null || true
  printf '%s\n' "$run_dir" > "$source_dir/run_dir.txt"
  append_json "$STATUS_JSONL" "{\"source\":\"$source\",\"dataset_path\":\"$dataset_path\",\"data_dir\":\"$data_dir\",\"run_dir\":\"$run_dir\",\"copied_run_dir\":\"$copied_run_dir\",\"completed_at\":\"$(date '+%Y-%m-%dT%H:%M:%S')\"}"
  summarize_results
  log "DONE source=$source run_dir=$run_dir"
}

log "medical_sources_all_baselines_100_medimem started"
completed=0
for source in "${SOURCES[@]}"; do
  if run_source "$source"; then
    completed=$((completed + 1))
  fi
  log "progress completed=$completed target=${#SOURCES[@]}"
done
summarize_results
if [ "$completed" -lt "${#SOURCES[@]}" ]; then
  log "FINISHED_WITH_BLOCKERS completed=$completed target=${#SOURCES[@]}"
  exit 1
fi
log "FINISHED completed=$completed target=${#SOURCES[@]}"
