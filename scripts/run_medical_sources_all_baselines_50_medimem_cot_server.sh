#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-/home/syh/A-mem}"
export PY="${PY:-/root/miniconda3/envs/llmserve/bin/python}"
export CACHE_DIR="${CACHE_DIR:-/home/syh/mem_ehr_hf_cache}"
export DATA_ROOT="${DATA_ROOT:-data/processed/medical_sources_all_baselines_50_20260603_cot}"
export RUN_ROOT="${RUN_ROOT:-runs/medical_sources_all_baselines_50_20260603_cot}"
export MAX_WORKERS="${MAX_WORKERS:-96}"
export PER_SOURCE_N="${PER_SOURCE_N:-50}"
export REPORT_STEM="${REPORT_STEM:-medical_sources_all_baselines_50_medimem_cot}"
export SOURCES_CSV="${SOURCES_CSV:-medical_meadow_wikidoc,medmcqa,medqa,chatdoctor_healthcaremagic,pmoa_tts}"

export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-http://127.0.0.1:8000/v1}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-/home/yyh/Qwen3-VL-30B-A3B-Instruct-FP8}"
export DEEPSEEK_TIMEOUT="${DEEPSEEK_TIMEOUT:-600}"

exec bash "$SCRIPT_DIR/run_medical_sources_all_baselines_100_medimem_server.sh"
