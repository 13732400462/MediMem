#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/syh/A-mem-aaai}"
DATASET="${DATASET:-$ROOT/data/processed/aaai_transfer_3000.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs/aaai_transfer_three_seed_20260706}"
MAX_WORKERS="${MAX_WORKERS:-16}"
SEEDS="${SEEDS:-20260706 20260707 20260708}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-http://127.0.0.1:8002/v1}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-local-key}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-qwen3-vl-8b-aaai}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$ROOT"
python -m medimem data validate --dataset "$DATASET"
[[ "$(wc -l < "$DATASET")" -eq 3000 ]] || { echo "expected 3000 transfer cases" >&2; exit 1; }

for seed in $SEEDS; do
  python -u -m medimem experiment-suite \
    --dataset "$DATASET" --require-api --max-workers "$MAX_WORKERS" \
    --suite-profile fast-formal --baseline-set required --ablation-groups full \
    --counterfactual-policy risk_sample --counterfactual-sample-rate 0.20 \
    --counterfactual-risk-threshold 0.55 --defer-reports --run-seed "$seed" \
    --output-root "$OUTPUT_ROOT/seed_${seed}"
done

mapfile -t prediction_dirs < <(find "$OUTPUT_ROOT" -type d -name predictions | sort)
python -m medimem analyze-run \
  --dataset "$DATASET" --predictions "${prediction_dirs[@]}" \
  --output-dir "$OUTPUT_ROOT/statistical_analysis" --target-method full_medimem \
  --compare-methods direct_deepseek,baseline_single_cot_agent,baseline_static_rag,baseline_amem_adapter \
  --resamples 10000 --random-seed 20260706

