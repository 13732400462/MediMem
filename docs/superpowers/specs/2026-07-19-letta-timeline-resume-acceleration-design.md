# Letta Timeline Benchmark Resume Acceleration

## Goal

Reduce the wall-clock time of the frozen non-medical Letta timeline benchmark without changing its sample sets, memory capacity, compaction prompt, model, answer protocol, or metrics.

## Constraints

- GPU0 is unavailable while its existing Qwen3-VL-30B vLLM process remains alive.
- GPU1 serves Qwen3-VL-8B with `max-num-seqs=8`.
- Existing validated per-sample predictions must be preserved and must not be recomputed.
- A final dataset run is valid only when every frozen sample has exactly one prediction and the existing horizontal validation passes.

## Design

Add resumable execution to the official Letta timeline adapter:

1. Accept one or more prior prediction JSONL files through a resume input.
2. Validate resumed rows against the current dataset, method, sample IDs, and endpoint/model protocol.
3. Group samples by conversation. A conversation is skipped only when all of its questions already have valid predictions; partially completed conversations are re-ingested, but already completed questions are not asked again.
4. Seed the final result map with accepted prior predictions and append new predictions from the current workers.
5. Record resume sources, accepted row count, recomputed row count, and duplicate/conflict checks in the experiment manifest.
6. Run eight workers on GPU1, matching the vLLM server's `max-num-seqs=8`.

The recovery queue will combine valid rows from the interrupted V5 and active V6 runs, then execute the remaining DialSim, RHELM, and LongMemEval samples. Dataset-level judging and metrics are regenerated from the complete merged predictions.

## Safety and Failure Handling

- Do not stop the active V6 queue until the resume implementation passes syntax, unit, and one-sample smoke checks.
- Snapshot V6 prediction files before termination.
- Reject resumed rows with unknown sample IDs, wrong datasets or methods, empty answers, duplicate conflicts, or failed guard flags.
- Never touch GPU0 while its existing process is present.
- If the accelerated run fails, its accepted resume set remains reusable in the next recovery run.

## Verification

- Unit tests cover resume filtering, complete and partial conversation handling, duplicate conflicts, and deterministic final ordering.
- A smoke run verifies merged prediction validation on a small frozen manifest.
- After launch, verify eight Letta workers, GPU1-only visibility, increasing prediction counts, and no GPU0 process changes.
- Re-estimate ETA from the first 30–60 minutes of accelerated throughput.
