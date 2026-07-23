#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/syh/A-mem-aaai}"
RUN_ROOT="${RUN_ROOT:-$PROJECT/runs/table2_hierarchical_bge_20260720}"
VLLM_PY="${VLLM_PY:-/root/miniconda3/envs/qwen3/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/models/qwen_vl_size_sweep/8b}"
SERVED_MODEL="${SERVED_MODEL:-qwen3-vl-8b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
mkdir -p "$RUN_ROOT/services"

for port in 8001 8002; do
  if ss -ltn | grep -Eq ":${port}[[:space:]]"; then
    echo "Port ${port} is already occupied; refusing to replace an existing service." >&2
    exit 1
  fi
done

start_one() {
  local gpu="$1"
  local port="$2"
  local label="gpu${gpu}_${port}"
  local pid_file="$RUN_ROOT/services/vllm_${label}.pid"
  local log_file="$RUN_ROOT/services/vllm_${label}.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export VLLM_USE_FLASHINFER_SAMPLER=0
    exec "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL" \
      --host 127.0.0.1 \
      --port "$port" \
      --trust-remote-code \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len 16384 \
      --max-num-seqs 16 \
      --enforce-eager
  ) >"$log_file" 2>&1 &
  echo "$!" >"$pid_file"
}

wait_ready() {
  local port="$1"
  local pid_file="$2"
  local pid
  pid="$(cat "$pid_file")"
  for _ in $(seq 1 360); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM process ${pid} exited before port ${port} became ready." >&2
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for vLLM on port ${port}." >&2
  return 1
}

start_one 0 8001
start_one 1 8002
wait_ready 8001 "$RUN_ROOT/services/vllm_gpu0_8001.pid" &
ready0=$!
wait_ready 8002 "$RUN_ROOT/services/vllm_gpu1_8002.pid" &
ready1=$!
wait "$ready0"
wait "$ready1"

"$VLLM_PY" - <<'PY' >"$RUN_ROOT/services/model_artifact_manifest.json"
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ.get("MODEL_PATH", "/root/models/qwen_vl_size_sweep/8b"))
files = []
aggregate = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    relative = str(path.relative_to(root)).replace("\\", "/")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    files.append({"path": relative, "bytes": path.stat().st_size, "sha256": value})
    aggregate.update(relative.encode())
    aggregate.update(value.encode())
print(json.dumps({
    "model_path": str(root),
    "served_model": os.environ.get("SERVED_MODEL", "qwen3-vl-8b"),
    "artifact_sha256": aggregate.hexdigest(),
    "files": files,
}, indent=2))
PY

printf '%s\n' "ready"
