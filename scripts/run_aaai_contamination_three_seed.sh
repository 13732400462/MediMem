#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/syh/A-mem-aaai}"
DATASET="${DATASET:-$ROOT/data/processed/aaai_core_20260706/pmoa_pmc_2000.jsonl}"
CORE_ROOT="${CORE_ROOT:-$ROOT/runs/aaai_core_three_seed_20260706}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs/aaai_contamination_three_seed_20260706}"
MAX_WORKERS="${MAX_WORKERS:-16}"
SEEDS="${SEEDS:-20260706 20260707 20260708}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-http://127.0.0.1:8002/v1}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-local-key}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-qwen3-vl-8b-aaai}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$ROOT"
python -m medimem data validate --dataset "$DATASET"
for seed in $SEEDS; do
  python -u -m medimem experiment-suite \
    --dataset "$DATASET" --require-api --max-workers "$MAX_WORKERS" \
    --suite-profile fast-formal --baseline-set none --ablation-groups with_polluted_memory \
    --counterfactual-policy risk_sample --counterfactual-sample-rate 0.20 \
    --counterfactual-risk-threshold 0.55 --defer-reports --run-seed "$seed" \
    --output-root "$OUTPUT_ROOT/seed_${seed}"
done

mapfile -t core_dirs < <(find "$CORE_ROOT" -type d -name predictions | sort)
mapfile -t pollution_dirs < <(find "$OUTPUT_ROOT" -type d -name predictions | sort)
python -m medimem analyze-run \
  --dataset "$DATASET" --predictions "${core_dirs[@]}" "${pollution_dirs[@]}" \
  --output-dir "$OUTPUT_ROOT/statistical_analysis" \
  --target-method ablate_with_polluted_memory_medimem --compare-methods full_medimem \
  --resamples 10000 --random-seed 20260706

