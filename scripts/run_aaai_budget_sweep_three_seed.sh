#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yh_shi/A-mem}"
CORE_DATASET="${CORE_DATASET:-$ROOT/data/processed/aaai_core_20260706/pmoa_pmc_2000.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs/aaai_budget_sweep_20260706}"
MAX_WORKERS="${MAX_WORKERS:-8}"
SEEDS="${SEEDS:-20260706 20260707 20260708}"
BUDGETS="${BUDGETS:-750 1500 3000}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-http://127.0.0.1:8002/v1}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-local-key}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-qwen3-vl-8b-aaai}"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"
python -m mem_ehr_agent data-validate --dataset "$CORE_DATASET"

for budget in $BUDGETS; do
  for seed in $SEEDS; do
    run_root="$OUTPUT_ROOT/budget_${budget}_seed_${seed}"
    mkdir -p "$run_root"
    python -u -m mem_ehr_agent experiment-suite \
      --dataset "$CORE_DATASET" \
      --require-api \
      --max-workers "$MAX_WORKERS" \
      --suite-profile fast-formal \
      --baseline-set required \
      --ablation-groups full \
      --counterfactual-policy risk_sample \
      --counterfactual-sample-rate 0.20 \
      --counterfactual-risk-threshold 0.55 \
      --completion-token-budget "$budget" \
      --defer-reports \
      --run-seed "$seed" \
      --output-root "$run_root"
  done
done

mapfile -t prediction_dirs < <(find "$OUTPUT_ROOT" -type d -name predictions | sort)
python -m mem_ehr_agent analyze-run \
  --dataset "$CORE_DATASET" \
  --predictions "${prediction_dirs[@]}" \
  --output-dir "$OUTPUT_ROOT/statistical_analysis" \
  --target-method full_medimem \
  --compare-methods direct_deepseek,baseline_single_cot_agent,baseline_static_rag,baseline_amem_adapter \
  --resamples 10000 \
  --random-seed 20260706
