#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yh_shi/A-mem}"
SOURCE_DATASET="${SOURCE_DATASET:-$ROOT/data/processed/frozen_20260630/medical_ehr_pool_6000.jsonl}"
CORE_DATASET="${CORE_DATASET:-$ROOT/data/processed/aaai_core_20260706/pmoa_pmc_2000.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/runs/aaai_core_three_seed_20260706}"
MAX_WORKERS="${MAX_WORKERS:-8}"
SEEDS="${SEEDS:-20260706 20260707 20260708}"
ABLATIONS="${ABLATIONS:-full,static_memory_no_critic,no_memory_cleaning,no_dynamic_top_k,no_evidence_note_injection,no_counterfactual_verification,no_sanitization_boundary}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-http://127.0.0.1:8002/v1}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-local-key}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-qwen3-vl-8b-aaai}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$ROOT"
mkdir -p "$(dirname "$CORE_DATASET")" "$OUTPUT_ROOT"

if [[ ! -s "$CORE_DATASET" ]]; then
  python -m medimem filter-sources \
    --dataset "$SOURCE_DATASET" \
    --sources pmoa_tts,pmc_patients \
    --output "$CORE_DATASET"
fi

python -m medimem data validate --dataset "$CORE_DATASET"
[[ "$(wc -l < "$CORE_DATASET")" -eq 2000 ]] || { echo "expected 2000 core cases" >&2; exit 1; }

for seed in $SEEDS; do
  seed_root="$OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$seed_root"
  python -u -m medimem experiment-suite \
    --dataset "$CORE_DATASET" \
    --require-api \
    --max-workers "$MAX_WORKERS" \
    --suite-profile fast-formal \
    --baseline-set required \
    --ablation-groups "$ABLATIONS" \
    --counterfactual-policy risk_sample \
    --counterfactual-sample-rate 0.20 \
    --counterfactual-risk-threshold 0.55 \
    --defer-reports \
    --run-seed "$seed" \
    --output-root "$seed_root"
done

mapfile -t prediction_dirs < <(find "$OUTPUT_ROOT" -type d -name predictions | sort)
python -m medimem analyze-run \
  --dataset "$CORE_DATASET" \
  --predictions "${prediction_dirs[@]}" \
  --output-dir "$OUTPUT_ROOT/statistical_analysis" \
  --target-method full_medimem \
  --compare-methods direct_deepseek,baseline_single_cot_agent,baseline_static_rag,baseline_amem_adapter \
  --resamples 10000 \
  --random-seed 20260706

