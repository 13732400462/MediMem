# Strict no-pollution size sweep

The formal MediMem pipeline treats clean memory as the main line: `full` does not bootstrap `poison_records` into the memory store. Polluted memory is available only as the `ablate_with_polluted_memory` ablation group, so it is reported as an ablation delta rather than a main pipeline.

Server-side strict four-size sweep:

```bash
cd /home/syh/A-mem
PY=/root/miniconda3/envs/llmserve/bin/python \
VLLM_PY=/root/miniconda3/envs/llmserve/bin/python \
bash scripts/run_strict_size_sweep_dual_pipeline.sh
```

Defaults:

- Wave 1: GPU0 `qwen3-vl-0_8b` on port 8000, GPU1 `qwen3-vl-2b` on port 8001.
- Wave 2: GPU0 `qwen3-vl-4b` on port 8000, GPU1 `qwen3-vl-8b` on port 8001.
- Each GPU runs two strict pipelines at the same time: medical pooled `MAX_WORKERS=48` and LoCoMo `MAX_WORKERS=48`.
- Formal runs use `--require-api`, `--require-real-data`, `MEDICAL_STRICT_NO_LEAK_FILTER=1`, and fail if leakage, blocked sources, progress failures, or fallback predictions are detected.

Acceptance gates:

- Medical pooled `full_medimem_merged.primary_diag_objective` must beat the strongest required baseline.
- LoCoMo MediMem QA F1 must beat the A-MEM wrapper and horizontal validation must pass.
- `critical_leakage_count`, `needs_review_count`, fallback prediction count, blocked sources, and progress failures must all be zero.
