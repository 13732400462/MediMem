#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/syh/A-mem-aaai}"
PY="${PY:-/root/miniconda3/envs/qwen3/bin/python}"
VLLM_PY="${VLLM_PY:-/home/yyh/qwen3vl_fast_python.sh}"
MODEL_PATH="${MODEL_PATH:-/root/models/qwen_vl_size_sweep/8b}"
MODEL_ID="${MODEL_ID:-qwen3-vl-8b}"
DATASET="${DATASET:?Set DATASET to the frozen PMOA-TTS 5,000-case JSONL path}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/pairwise_pmoa_ablation_gpu0_20260727_$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-48}"

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/launcher.log") 2>&1

if [ ! -s "$DATASET" ]; then
  echo "missing dataset: $DATASET" >&2
  exit 1
fi
if nvidia-smi --id=0 --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q .; then
  echo "GPU 0 compute process detected; inspect ownership before launch." >&2
  nvidia-smi
  exit 1
fi
if ss -ltn | grep -q ":${PORT} "; then
  echo "port ${PORT} is already in use" >&2
  exit 1
fi

dataset_sha="$("$PY" - "$DATASET" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
printf '%s\n' "$dataset_sha" > "$RUN_ROOT/dataset.sha256"
printf '%s\n' "$MODEL_PATH" > "$RUN_ROOT/model_path.txt"

CUDA_VISIBLE_DEVICES=0 nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --max-num-seqs 48 \
  > "$RUN_ROOT/vllm.log" 2>&1 &
vllm_pid=$!
printf '%s\n' "$vllm_pid" > "$RUN_ROOT/vllm.pid"

cleanup() {
  monitor_pid="$(cat "$RUN_ROOT/monitor.pid" 2>/dev/null || true)"
  if [ -n "$monitor_pid" ] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
  fi
  if kill -0 "$vllm_pid" 2>/dev/null; then
    if ps -p "$vllm_pid" -o args= | grep -Fq -- "$MODEL_PATH"; then
      kill "$vllm_pid" 2>/dev/null || true
      wait "$vllm_pid" 2>/dev/null || true
    else
      echo "refusing to stop pid $vllm_pid: ownership mismatch" >&2
    fi
  fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 240); do
  if "$PY" - "$PORT" "$MODEL_ID" <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1:]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
        ids = [row.get("id") for row in json.load(response).get("data", [])]
    raise SystemExit(0 if model in ids else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    ready=1
    break
  fi
  sleep 5
done
if [ "$ready" -ne 1 ] || ! kill -0 "$vllm_pid" 2>/dev/null; then
  echo "vLLM failed readiness" >&2
  exit 1
fi

(
  echo "timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct,utilization_memory_pct"
  while true; do
    printf '%s,' "$(date --iso-8601=seconds)"
    nvidia-smi --id=0 \
      --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits
    sleep 5
  done
) > "$RUN_ROOT/gpu_utilization.csv" &
echo "$!" > "$RUN_ROOT/monitor.pid"

export PYTHONPATH="$PROJECT_ROOT"
export DEEPSEEK_BASE_URL="http://127.0.0.1:${PORT}/v1"
export DEEPSEEK_API_KEY="local-qwen3"
export DEEPSEEK_MODEL="$MODEL_ID"
export DEEPSEEK_TIMEOUT=600
export DEEPSEEK_MAX_TOKENS=1800
export MEDICAL_PREDICTION_MAX_TOKENS=700
export MEDICAL_JSON_PARSE_RETRIES=2
export MEDICAL_STRICT_NO_LEAK_FILTER=1
export MEDICAL_LLM_SEED=20260606
export CUDA_VISIBLE_DEVICES=0

"$PY" -u -m medimem experiment-suite \
  --dataset "$DATASET" \
  --require-api \
  --max-workers "$WORKERS" \
  --suite-profile fast-formal \
  --baseline-set none \
  --ablation-groups \
    no_evidence_no_cleaning,no_evidence_fixed_top_k,no_cleaning_fixed_top_k \
  --counterfactual-policy risk_sample \
  --counterfactual-sample-rate 0.20 \
  --counterfactual-risk-threshold 0.55 \
  --defer-reports \
  --run-seed 20260606 \
  --output-root "$RUN_ROOT/formal" \
  | tee "$RUN_ROOT/experiment.log"

run_dir="$(grep -E '^run_dir=' "$RUN_ROOT/experiment.log" | tail -1 | cut -d= -f2-)"
if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
  echo "formal run directory was not recorded" >&2
  exit 1
fi
printf '%s\n' "$run_dir" > "$RUN_ROOT/formal_run_dir.txt"

"$PY" - "$run_dir" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
expected = {
    "ablate_no_evidence_no_cleaning_medimem_merged",
    "ablate_no_evidence_fixed_top_k_medimem_merged",
    "ablate_no_cleaning_fixed_top_k_medimem_merged",
}
manifest = json.loads((run / "experiment_manifest.json").read_text(encoding="utf-8"))
if manifest.get("case_count") != 5000:
    raise SystemExit(f"case_count mismatch: {manifest.get('case_count')}")
rows = []
import csv
with (run / "metrics.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
observed = {row.get("method") for row in rows}
missing = sorted(expected - observed)
wrong_n = {
    row.get("method"): row.get("n")
    for row in rows
    if row.get("method") in expected and int(float(row.get("n") or 0)) != 5000
}
progress = [
    json.loads(line)
    for line in (run / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
failed = [row for row in progress if row.get("ok") is not True]
summary = {
    "expected_methods": sorted(expected),
    "observed_methods": sorted(observed),
    "missing_methods": missing,
    "wrong_n": wrong_n,
    "progress_failed": len(failed),
    "passed": not missing and not wrong_n and not failed,
}
(run.parent.parent / "pairwise_audit.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
if not summary["passed"]:
    raise SystemExit("pairwise ablation audit failed")
PY

touch "$RUN_ROOT/FORMAL_COMPLETE"
echo "pairwise ablation complete: $run_dir"

