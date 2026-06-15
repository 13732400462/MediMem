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
- The default high-throughput schedule is `PIPELINE_SCHEDULE=pool_then_locomo`: run the medical pooled source first, then run LoCoMo after the pooled source finishes. This preserves completed pooled work if the host is interrupted.
- The verified 72-thread/dual-4090 server profile uses `EXCLUSIVE_PIPELINE_WORKERS=128` and `VLLM_MAX_NUM_SEQS=128` for each active model service. On 2026-06-15, the 4B/8B resumed wave reached real vLLM queues under this profile with GPU0/GPU1 at 100% utilization, `qwen3-vl-4b` around 66 running plus 54 waiting requests, `qwen3-vl-8b` around 73 running plus 48 waiting requests, and zero vLLM error/abort counts at the checkpoint.
- If a model size cannot start or remains unstable at 128 because of memory pressure, lower both `EXCLUSIVE_PIPELINE_WORKERS` and `VLLM_MAX_NUM_SEQS` to 96 for that wave. Do not replace missing outputs with fallback predictions.
- The legacy promote schedule remains available with `PIPELINE_SCHEDULE=promote`: each GPU starts medical pooled `MAX_WORKERS=48` and LoCoMo `MAX_WORKERS=48`; after one pipeline finishes, the remaining pipeline is rerun/promoted at `EXCLUSIVE_PIPELINE_WORKERS`.
- Formal runs use `--require-api`, `--require-real-data`, `MEDICAL_STRICT_NO_LEAK_FILTER=1`, and fail if leakage, blocked sources, progress failures, or fallback predictions are detected.

Acceptance gates:

- Medical pooled `full_medimem_merged.primary_diag_objective` must beat the strongest required baseline.
- LoCoMo MediMem QA F1 must beat the A-MEM wrapper and horizontal validation must pass.
- `critical_leakage_count`, `needs_review_count`, fallback prediction count, blocked sources, and progress failures must all be zero.
