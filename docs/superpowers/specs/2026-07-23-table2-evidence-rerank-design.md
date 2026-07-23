# Table 2 Evidence-Rerank Design

## Goal and Scope

Improve the MediMem row in the frozen four-source non-medical Table 2 without
rerunning any baseline, medical experiment, ablation, or independent LoCoMo
long-dialogue experiment. The change is isolated to an opt-in Table 2
non-medical timeline retrieval mode. Existing defaults and every other
experiment path remain unchanged.

The formal run continues to use the frozen LoCoMo 1,000, LongMemEval 500,
DialSim 1,000, and RHELM 1,305 sample IDs, Qwen3-VL-8B checkpoint, answer
prompt, decoding configuration, blinded judge, and metric definitions.

## Selected Approach

Add an `evidence_rerank` timeline retriever that uses:

1. the existing lexical and dense BGE ranks to generate a broad candidate pool;
2. reciprocal-rank fusion to combine those candidates;
3. a frozen cross-encoder to score each question--candidate pair;
4. deterministic redundancy control and temporal-neighbor packing; and
5. a fixed number of final evidence units passed to the unchanged QA prompt.

This approach directly targets the Evidence R@5 deficit while preserving exact
entity, number, and date matches. It is preferred over generative evidence
selection because it adds no stochastic evidence-selection call, and over
simple context expansion because the previous hybrid-BGE run already used a
context budget similar to Mem0 without closing the quality gap.

## Isolation and Configuration

The new retriever is opt-in and available only through explicit timeline
benchmark arguments. The default retriever remains unchanged. No medical
entry point may select `evidence_rerank` implicitly.

The manifest and cache identity record:

- lexical and dense candidate budgets;
- embedding model identifier, resolved revision, and artifact hash;
- cross-encoder identifier, resolved revision, and artifact hash;
- RRF weights;
- cross-encoder rerank budget;
- redundancy threshold and temporal-neighbor radius;
- final evidence-unit budget and character budget;
- card granularity and card size;
- dataset and sample-manifest hashes; and
- code commit.

If an appropriate cross-encoder already exists on the server, it is preferred.
Otherwise a fixed model revision is downloaded before selection. The chosen
artifact is frozen and hashed before development QA runs.

## Retrieval Data Flow

Static memory cards are built only from fields visible to the model under the
existing Table 2 protocol. The question may be used at retrieval time. Gold
answers, gold evidence, judge outputs, evaluator fields, and formal test
metrics are forbidden from index construction, candidate generation, rerank
features, and packing.

For each question:

1. Build the existing expanded lexical query.
2. Produce lexical and dense candidate ranks over the permitted memory cards.
3. Fuse the ranks deterministically with weighted RRF.
4. Score the bounded fused pool with the frozen cross-encoder.
5. Remove near-duplicate candidates while preserving the highest-scored unit.
6. Add bounded adjacent temporal units only when they fit the frozen packing
   budget.
7. Return the fixed number of evidence units in deterministic order to the
   unchanged QA prompt.

Embedding and cross-encoder results are cached using complete configuration
identities. A cache produced by another dataset, model revision, card format,
or retrieval configuration cannot be reused.

## Development Selection

Selection uses only existing disjoint development manifests. Formal Table 2
sample IDs and their results are not inspected for parameter selection.

Development proceeds in two stages:

1. Offline retrieval screening compares a small, declared grid of candidate
   pool, RRF, rerank, redundancy, neighbor, and final packing budgets. Evidence
   recall and context size are computed only on development data.
2. A small number of retrieval finalists run the unchanged Qwen QA and judge.
   Candidates are ordered by development answer quality, evidence recall,
   token use, and latency.

The exact grid and ordering rule are written into the development manifest
before any candidate runs. The selected configuration is frozen before the
formal run. If offline retrieval remains clearly below the existing MediMem
evidence level, development continues without spending a full formal QA run.
Formal test results are never used to start another tuning round.

## Formal Run

After configuration freeze, run MediMem once on all four frozen sources:

- LoCoMo: 1,000 samples;
- LongMemEval: 500 samples;
- DialSim: 1,000 samples; and
- RHELM: 1,305 samples.

No dataset-level configuration splicing is allowed. Missing items may be
resumed only with the identical frozen configuration and cache identity.
Direct, Static RAG, A-MEM, DDO, G-Memory, MemInsight, MemoryOS, Mem0, and
Letta/MemGPT are not rerun.

## Validation and Leakage Controls

Unit tests cover:

- deterministic lexical/dense fusion and stable tie breaking;
- cross-encoder score propagation and bounded reranking;
- duplicate removal and temporal-neighbor packing;
- fixed evidence-unit and character budgets;
- cache isolation across all model and retrieval settings;
- CLI and manifest propagation;
- backward compatibility of all existing retriever modes; and
- exclusion of answer, gold evidence, judge, and evaluator fields.

Each formal source must contain its complete unique frozen ID set. The
following counts must all be zero: blocked, fallback, API error, worker
failure, judge missing, progress failure, critical leakage, and needs review.
Validation and manifest guards must pass before any result is considered.

## Paper Adoption Rule

Paper adoption is based on relative Table 2 competitiveness, not fixed
predeclared MediMem score thresholds.

For every quantitative Table 2 column in which MediMem and at least one other
method have comparable reported values:

1. determine the best value among the unchanged baseline rows and the candidate
   MediMem row;
2. compute MediMem's nonnegative gap to that column's best value; and
3. compare the candidate gap with the current-paper MediMem gap.

All percentage-valued columns use percentage points, so no cross-scale
normalization is applied. Missing or non-comparable cells are excluded. The
overall gap is the unweighted mean of the included column gaps. The candidate
is adopted if the new overall gap is strictly smaller than the current-paper
overall gap and every operational, completeness, and leakage gate passes.
Individual columns may decline if larger improvements elsewhere reduce the
overall gap. A direct column win has zero gap.

The audit reports every included column, its baseline best, old MediMem value,
new MediMem value, old gap, new gap, and the two overall means. This prevents a
post-hoc change to the adoption calculation.

If adopted, update the existing Table 2 row, analysis, conclusion, result
archive, and project contexts using only the single complete formal run. If
not adopted, archive the run and leave the paper unchanged.

## Server Operation and Cleanup

Before launch, inspect GPU 0 and GPU 1 utilization, running processes, ports,
and process ownership. Start two identical Qwen3-VL-8B vLLM services only when
both GPUs are available. Record service commands, ports, process groups, and
PID files.

After all four runs, validation, aggregation, and artifact synchronization are
complete, stop every vLLM process started for this experiment. Do not stop
unrelated processes. Confirm that the recorded ports are released and both
experiment service process groups have exited.
