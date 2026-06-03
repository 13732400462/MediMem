#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASELINE_ROOT="${BASELINE_ROOT:-$PROJECT_ROOT/baseline_envs}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
REPOS_ROOT="$BASELINE_ROOT/repos"
ENVS_ROOT="$BASELINE_ROOT/conda"
LOG_ROOT="$BASELINE_ROOT/logs"
CACHE_ROOT="$BASELINE_ROOT/cache"
REQ_ROOT="$BASELINE_ROOT/requirements"
SOURCE_ROOT="$PROJECT_ROOT/third_party/official_baselines"

mkdir -p "$REPOS_ROOT" "$ENVS_ROOT" "$LOG_ROOT" "$CACHE_ROOT" "$REQ_ROOT"

copy_repo() {
  local name="$1"
  local src="$SOURCE_ROOT/$name"
  local dst="$REPOS_ROOT/$name"
  if [ ! -d "$src" ]; then
    echo "missing source repo: $src" >&2
    return 1
  fi
  if [ ! -d "$dst" ]; then
    cp -a "$src" "$dst"
  fi
}

create_env() {
  local name="$1"
  local python_version="$2"
  local prefix="$ENVS_ROOT/$name"
  if [ ! -x "$prefix/bin/python" ]; then
    "$CONDA_BIN" create -y --prefix "$prefix" "python=$python_version"
  fi
}

pip_install() {
  local name="$1"
  local req="$2"
  local prefix="$ENVS_ROOT/$name"
  "$prefix/bin/python" -m pip install --upgrade pip wheel setuptools
  "$prefix/bin/python" -m pip install -r "$req" 2>&1 | tee "$LOG_ROOT/${name}_pip_install.log"
}

write_requirements() {
  grep -v -E 'faiss-gpu|^#|^$' "$REPOS_ROOT/MemoryOS/memoryos-pypi/requirements.txt" > "$REQ_ROOT/memoryos.txt"
  printf '%s\n' 'faiss-cpu' >> "$REQ_ROOT/memoryos.txt"

  grep -v -E 'faiss-gpu|^#|^$' "$REPOS_ROOT/MemInsight/requirements.txt" > "$REQ_ROOT/meminsight.txt"
  if ! grep -q '^faiss-cpu' "$REQ_ROOT/meminsight.txt"; then
    printf '%s\n' 'faiss-cpu' >> "$REQ_ROOT/meminsight.txt"
  fi

  sed 's/^skimage==0.0$/scikit-image/' "$REPOS_ROOT/GMemory/requirements.txt" > "$REQ_ROOT/gmemory.txt"

  grep -v -E ' @ file://|flash-attn|deepspeed|faiss-gpu|bitsandbytes|xformers|triton|nvidia-|^#|^$' \
    "$REPOS_ROOT/DDO/requirements.txt" > "$REQ_ROOT/ddo.txt"
  if ! grep -q '^faiss-cpu' "$REQ_ROOT/ddo.txt"; then
    printf '%s\n' 'faiss-cpu' >> "$REQ_ROOT/ddo.txt"
  fi
}

write_exports() {
  cat > "$BASELINE_ROOT/export_baseline_envs.sh" <<EOF
#!/usr/bin/env bash
export MEMORYOS_REPO="$REPOS_ROOT/MemoryOS"
export MEMINSIGHT_REPO="$REPOS_ROOT/MemInsight"
export GMEMORY_REPO="$REPOS_ROOT/GMemory"
export DDO_REPO="$REPOS_ROOT/DDO"
export MEMORYOS_PY="$ENVS_ROOT/memoryos/bin/python"
export MEMINSIGHT_PY="$ENVS_ROOT/meminsight/bin/python"
export GMEMORY_PY="$ENVS_ROOT/gmemory/bin/python"
export DDO_PY="$ENVS_ROOT/ddo/bin/python"
export BASELINE_ENVS_ROOT="$BASELINE_ROOT"
EOF
  chmod +x "$BASELINE_ROOT/export_baseline_envs.sh"
}

main() {
  copy_repo MemoryOS
  copy_repo MemInsight
  copy_repo GMemory
  copy_repo DDO
  write_requirements
  write_exports

  if [ "${SKIP_CONDA_CREATE:-0}" = "1" ]; then
    echo "SKIP_CONDA_CREATE=1; wrote repos, requirements, and export file only."
    exit 0
  fi

  create_env memoryos 3.10
  create_env meminsight 3.10
  create_env gmemory 3.12
  create_env ddo 3.9

  if [ "${SKIP_PIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_PIP_INSTALL=1; conda environments created without dependency install."
    exit 0
  fi

  pip_install memoryos "$REQ_ROOT/memoryos.txt"
  pip_install meminsight "$REQ_ROOT/meminsight.txt"
  pip_install gmemory "$REQ_ROOT/gmemory.txt"
  pip_install ddo "$REQ_ROOT/ddo.txt"
}

main "$@"
