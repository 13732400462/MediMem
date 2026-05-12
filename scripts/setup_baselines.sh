#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
mkdir -p "$ROOT/baseline_repos"

if [ ! -d "$ROOT/baseline_repos/DDO/.git" ]; then
  git clone --depth 1 https://github.com/zh-jia/DDO "$ROOT/baseline_repos/DDO"
fi

if [ ! -d "$ROOT/baseline_repos/ColaCare/.git" ]; then
  git clone --depth 1 https://github.com/PKU-AICare/ColaCare "$ROOT/baseline_repos/ColaCare"
fi

echo "Baseline repos are available under $ROOT/baseline_repos"

