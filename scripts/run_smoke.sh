#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m mem_ehr_agent data build --n 10 --output data/processed/samples.jsonl
python -m mem_ehr_agent data validate --dataset data/processed/samples.jsonl
python -m mem_ehr_agent optimize --dataset data/processed/samples.jsonl --max-rounds 3

