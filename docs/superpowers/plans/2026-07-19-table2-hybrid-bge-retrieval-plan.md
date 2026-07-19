# Table 2 Hybrid BGE Retrieval Implementation Plan

## Objective

Implement the approved hybrid lexical/BGE retriever as an opt-in Table 2
configuration, validate it locally, select one configuration on the frozen
LoCoMo development set, and run the four-source formal test once.

## Task 1: Retrieval primitives

Files:

- `mem_ehr_agent/benchmark.py`
- `tests/test_benchmark.py`

Steps:

1. Add retriever configuration normalization with `lexical` as the default.
2. Add deterministic card text windowing.
3. Add a lazy CPU BGE encoder wrapper that normalizes embeddings and exposes
   the resolved model identity.
4. Add maximum-window semantic similarity.
5. Add stable weighted RRF with deterministic lexical/entity/time tie breaks.
6. Test windowing, normalized scores, RRF order, tie behavior, and lexical
   backward compatibility.

## Task 2: Cache, CLI, and manifest isolation

Files:

- `mem_ehr_agent/benchmark.py`
- `mem_ehr_agent/cli.py`
- `pyproject.toml`
- `tests/test_benchmark.py`

Steps:

1. Add the four approved CLI arguments and pass them through the benchmark
   entry point.
2. Include retriever, encoder identity/revision, window size, and RRF weight in
   memory and embedding cache identities.
3. Cache static card-window embeddings behind an atomic per-timeline lock.
4. Record the full retrieval configuration in predictions and the benchmark
   manifest.
5. Add `sentence-transformers` as an explicit semantic-retrieval optional
   dependency while keeping lexical installations unchanged.
6. Test cache isolation, CLI parsing, manifest propagation, label exclusion,
   top-five enforcement, and the full-context guard.

## Task 3: Local verification and commit

Steps:

1. Run targeted retrieval and CLI tests.
2. Run the complete test suite.
3. Run `git diff --check` and inspect the diff for unrelated changes.
4. Commit the implementation separately from the approved design and plan.

## Task 4: Server synchronization and environment audit

Steps:

1. Verify both GPUs and ports 8001/8002 are idle.
2. Verify no unrelated user process would be affected.
3. Synchronize the committed code to `/home/syh/A-mem-aaai`.
4. Install or verify the pinned BGE encoder dependency and download/cache the
   frozen encoder before starting formal services.
5. Record the encoder artifact revision/hash.

## Task 5: Frozen LoCoMo development selection

Steps:

1. Start two identical Qwen3-VL-8B vLLM services with separate PID files.
2. Run the existing 200-question LoCoMo development manifest for semantic RRF
   weights 1.0, 2.0, and 3.0, with all other settings fixed.
3. Validate complete unique IDs and zero blocked/fallback/API/worker/judge/
   guard/leakage failures.
4. Require Evidence R@5 >= 70%.
5. Select by Answer Accuracy, Evidence R@5, token use, then lower weight.
6. Write a frozen-selection manifest before any formal run.
7. If no candidate qualifies, archive the development results, stop both vLLM
   services, verify ports and GPU memory, and stop.

## Task 6: One frozen four-source formal run

Steps:

1. Run LoCoMo 1,000 and LongMemEval 500 on one service.
2. Run DialSim 1,000 and RHELM 1,305 on the other service.
3. Resume only missing frozen sample IDs if an operational interruption occurs.
4. Validate all formal gates before calculating accepted metrics.
5. Compute source metrics, aggregate metrics, LoCoMo QA F1/BLEU-1, token use,
   latency, and 10,000 paired bootstrap resamples against Mem0.

## Task 7: Archive, paper gate, and shutdown

Steps:

1. Synchronize all development and formal artifacts to the local
   `server_results` archive.
2. If both Mem0 thresholds are strictly exceeded, update Table 2, the related
   prose, project contexts, PDF, Overleaf ZIP, and Overleaf project; compile and
   render-check every page.
3. Otherwise leave the paper unchanged and record the run as unadopted.
4. Stop only this run's two vLLM process groups.
5. Confirm ports 8001/8002 are released and both GPUs report zero memory use.
