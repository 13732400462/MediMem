#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m medimem data build --n 10 --output data/processed/samples.jsonl
python -m medimem data validate --dataset data/processed/samples.jsonl
python -m medimem optimize --dataset data/processed/samples.jsonl --max-rounds 3


