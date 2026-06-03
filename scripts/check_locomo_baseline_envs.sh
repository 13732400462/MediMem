#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_ROOT/baseline_envs}"
EXPORT_FILE="$BASELINE_ROOT/export_baseline_envs.sh"

if [ -f "$EXPORT_FILE" ]; then
  # shellcheck disable=SC1090
  source "$EXPORT_FILE"
fi

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read())[1:-1])'
}

check_one() {
  local method="$1"
  local repo_var="$2"
  local py_var="$3"
  local imports="$4"
  local repo="${!repo_var:-}"
  local py="${!py_var:-}"
  local status="configured"
  local detail=""

  if [ -z "$repo" ] || [ ! -d "$repo" ]; then
    status="missing_repo"
    detail="$repo_var=$repo"
  elif [ -z "$py" ] || [ ! -x "$py" ]; then
    status="missing_python"
    detail="$py_var=$py"
  elif ! "$py" -c "$imports" >/tmp/${method}_baseline_import.out 2>/tmp/${method}_baseline_import.err; then
    status="import_failed"
    detail="$(cat /tmp/${method}_baseline_import.err)"
  fi

  printf '{"method":"%s","status":"%s","repo":"%s","python":"%s","detail":"%s"}' \
    "$method" "$status" "$(printf '%s' "$repo" | json_escape)" "$(printf '%s' "$py" | json_escape)" "$(printf '%s' "$detail" | json_escape)"
}

printf '['
check_one "memoryos" "MEMORYOS_REPO" "MEMORYOS_PY" "import numpy, openai"
printf ','
check_one "meminsight" "MEMINSIGHT_REPO" "MEMINSIGHT_PY" "import numpy, pandas, openai"
printf ','
check_one "gmemory" "GMEMORY_REPO" "GMEMORY_PY" "import yaml, networkx, openai"
printf ','
check_one "ddo" "DDO_REPO" "DDO_PY" "import numpy, pandas, openai"
printf ']\n'
