# Table 2 Evidence-Rerank Implementation Plan

## Objective

Implement and evaluate an opt-in `evidence_rerank` retriever for the frozen
four-source Table 2 MediMem row. Preserve all historical defaults and do not
rerun any baseline or unrelated experiment.

## Task 1: Add failing retrieval-unit tests

Modify `tests/test_benchmark.py` to specify:

- retriever configuration validation and defaults;
- deterministic cross-encoder reranking;
- bounded candidate pools;
- near-duplicate suppression;
- temporal-neighbor packing;
- cache/config identity propagation; and
- gold, answer, judge, and evaluator field exclusion.

Run the focused tests and confirm that the new expectations fail before
implementation.

## Task 2: Implement the isolated retriever

Modify `medimem/benchmark.py`:

- add a frozen cross-encoder wrapper with model identity and thread-safe batch
  prediction;
- implement lexical/dense candidate fusion;
- implement cross-encoder reranking;
- implement deterministic redundancy filtering and neighbor packing;
- add the opt-in retrieval path without changing lexical, hybrid-BGE, or
  hierarchical-BGE behavior;
- propagate diagnostics into predictions and manifests; and
- include every retrieval and model parameter in cache identity.

Modify `medimem/cli.py` to expose the explicit evidence-rerank arguments.

Run:

```text
pytest tests/test_benchmark.py -q
pytest -q
```

## Task 3: Add frozen experiment drivers

Add Table 2-specific scripts for:

- offline development retrieval screening;
- development QA selection over exactly the declared finalists;
- one-shot four-source formal execution;
- completeness, leakage, operational, metric, and relative-gap audit; and
- safe vLLM shutdown using only recorded PIDs.

The offline grid and selection order must be serialized before execution.
Formal launch must create an irreversible marker that prevents an accidental
second launch under the same run root.

## Task 4: Verify and commit local implementation

Run all tests, inspect `git diff --check`, review the complete diff, and commit
only the experiment implementation and tests. Do not touch the dirty paper
repository.

## Task 5: Discover and verify server state

Locate the existing authenticated server entry point without exposing
credentials. On the server:

- confirm `/home/syh/A-mem-aaai` and the expected branch/commit;
- inspect `nvidia-smi`, GPU 0/1 process ownership, ports, and existing PID files;
- verify the Qwen3-VL-8B artifact and Python environment;
- locate or install a fixed cross-encoder revision and record its hash; and
- sync only the committed experiment changes.

Do not stop or overwrite unrelated processes or uncommitted server work.

## Task 6: Run development selection

Create the frozen grid manifest before scoring. Use only the existing disjoint
development manifests. First run offline retrieval screening, then run Qwen QA
for the declared finalists. Validate all development artifacts and select one
configuration using the frozen ordering rule.

If the retrieval design is defective, fix it and repeat development work only.
Do not inspect formal Table 2 outputs while tuning.

## Task 7: Run the formal experiment once

After configuration freeze:

- start two identical vLLM services on GPU 0 and GPU 1;
- record commands, ports, PIDs, model identity, and health checks;
- launch the four frozen MediMem jobs with one shared configuration;
- resume only missing IDs under the identical configuration if interrupted;
- wait for all 3,805 predictions and judges; and
- prohibit a second formal configuration.

## Task 8: Audit and decide paper adoption

Verify complete unique IDs and zero blocked, fallback, API, worker, judge,
progress, leakage, and review failures. Recompute every Table 2 metric from raw
artifacts.

For every comparable quantitative Table 2 column, report the unchanged
baseline best, current MediMem value, candidate MediMem value, and old/new
nonnegative gaps. Adopt the new row only if the unweighted mean gap strictly
shrinks and all operational gates pass.

If adopted, update only the existing paper sources, contexts, result archives,
PDF, and Overleaf package through the established paper workflow. If rejected,
archive the run without modifying the paper.

## Task 9: Archive and shut down vLLM

Copy the formal predictions, judges, metrics, manifests, validations, audit,
logs, and exact commands into the local result archive. Then stop every vLLM
process recorded as started by this experiment, verify process exit and port
release, and leave unrelated services untouched.

