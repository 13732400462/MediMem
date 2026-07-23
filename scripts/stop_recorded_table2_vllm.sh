#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/syh/A-mem-aaai}"
RUN_ROOT="${RUN_ROOT:-$PROJECT/runs/table2_evidence_rerank_20260723}"
SERVICE_DIR="$RUN_ROOT/services"

for spec in "0:8001" "1:8002"; do
  gpu="${spec%%:*}"
  port="${spec##*:}"
  pid_file="$SERVICE_DIR/vllm_gpu${gpu}_${port}.pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "Missing recorded PID file: $pid_file" >&2
    exit 1
  fi
  pid="$(tr -d '[:space:]' <"$pid_file")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid PID in $pid_file" >&2
    exit 1
  fi
  command="$(ps -p "$pid" -o args= || true)"
  if [[ -z "$command" ]]; then
    continue
  fi
  if [[ "$command" != *"vllm.entrypoints.openai.api_server"* ]] ||
     [[ "$command" != *"--port $port"* ]] ||
     [[ "$command" != *"--served-model-name qwen3-vl-8b"* ]]; then
    echo "PID $pid does not match the recorded experiment vLLM command." >&2
    exit 1
  fi
done

for spec in "0:8001" "1:8002"; do
  gpu="${spec%%:*}"
  port="${spec##*:}"
  pid_file="$SERVICE_DIR/vllm_gpu${gpu}_${port}.pid"
  pid="$(tr -d '[:space:]' <"$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
  fi
done

for _ in $(seq 1 60); do
  alive=0
  for spec in "0:8001" "1:8002"; do
    gpu="${spec%%:*}"
    port="${spec##*:}"
    pid="$(tr -d '[:space:]' <"$SERVICE_DIR/vllm_gpu${gpu}_${port}.pid")"
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  done
  if [[ "$alive" -eq 0 ]]; then
    break
  fi
  sleep 1
done

for spec in "0:8001" "1:8002"; do
  gpu="${spec%%:*}"
  port="${spec##*:}"
  pid="$(tr -d '[:space:]' <"$SERVICE_DIR/vllm_gpu${gpu}_${port}.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Recorded vLLM PID $pid did not exit." >&2
    exit 1
  fi
  if ss -ltn | grep -Eq ":${port}[[:space:]]"; then
    echo "Port $port remains occupied after recorded process exit." >&2
    exit 1
  fi
done

printf '%s\n' "stopped"
