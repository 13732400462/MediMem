# Table 2 Hybrid BGE Retrieval Design

## Goal

Improve the MediMem row in the frozen four-source non-medical Table 2 without
changing the Qwen3-VL-8B checkpoint, QA prompt, decoding parameters, answer
evaluator, medical experiments, or baseline results. The new configuration is
accepted for the paper only if its four-source mean Answer Accuracy is strictly
greater than Mem0's 51.01% and its three-source mean Evidence R@5 is strictly
greater than Mem0's 69.18%.

## Motivation

The rejected session-card run improved the four-source mean Answer Accuracy
from 39.90% to 40.88% and Evidence R@5 from 31.19% to 49.70%, but it did not
pass the replacement gate. Its retrieval score is a set-token cosine with
small entity and temporal bonuses. This works for exact lexical overlap but is
weak when a question paraphrases the relevant memory. The problem is most
visible on RHELM, where the rejected run reached only 18.87% Evidence R@5 over
a large collection of long-lived character memories.

## Considered Approaches

1. **Hybrid BGE plus lexical retrieval (selected).** Combine a frozen semantic
   rank with the existing lexical rank through reciprocal rank fusion (RRF).
   Preserve exact entity, date, number, and keyword behavior while adding
   paraphrase sensitivity.
2. **Dense-only BGE retrieval.** Simpler, but more likely to lose exact temporal
   or named-entity matches and makes the change harder to attribute.
3. **Qwen query rewriting followed by lexical retrieval.** Potentially useful,
   but adds another generative call, latency, and stochastic behavior. It also
   makes the protocol less clean than a deterministic frozen encoder.

## Configuration Isolation

Add an explicit Table 2 retrieval mode:

- `--timeline-retriever {lexical,hybrid_bge}`, defaulting to `lexical`.
- `--timeline-embedding-model`, used only by `hybrid_bge`, defaulting to
  `BAAI/bge-small-en-v1.5`.
- `--timeline-semantic-rrf-weight`, used only by `hybrid_bge`.
- `--timeline-embedding-window-tokens`, a fixed window size for long cards.

The default configuration must reproduce historical lexical behavior. No
medical entry point may opt into the new retriever implicitly.

The benchmark manifest and cache identity must record the retriever, encoder
identifier, resolved encoder artifact hash or revision, embedding window,
semantic RRF weight, card granularity, card character limit, top-k, coarse-k,
dataset hash, and code commit.

## Index and Retrieval Flow

The underlying memory representation remains the frozen `session_chunk=4000`
card format. Each card is encoded without using the current question. Long
cards are split deterministically into fixed token windows in source order.
The semantic score for a card is the maximum cosine similarity between the
query embedding and its window embeddings. Embeddings are normalized.

For each question:

1. Build the existing expanded lexical query from the question and permitted
   category metadata.
2. Encode the same query with the frozen BGE query instruction.
3. Rank active or flagged cards independently by the existing lexical score
   and by semantic similarity.
4. Fuse the two complete ranks with RRF. The lexical contribution has weight
   1.0; the semantic contribution uses the development-selected weight.
5. Use deterministic entity and temporal exact-match bonuses only as final
   tie-breaking metadata. Do not use answer, gold evidence, judge output, or
   evaluator fields.
6. Return exactly the top five original session cards to the unchanged QA
   prompt. Do not synthesize question-dependent card text or expand to full
   context.

The encoder runs on CPU and its static card embeddings are cached and shared
between samples that use the same timeline. The two GPUs remain dedicated to
the two identical Qwen3-VL-8B vLLM services.

## Development Selection

Use only the existing disjoint 200-question LoCoMo development set. Keep
`session_chunk=4000`, `top-k=5`, the QA model, prompt, decoding, evaluator, and
all other settings fixed. Compare exactly three semantic RRF weights:

- 1.0
- 2.0
- 3.0

Every candidate must pass the strict audit and reach development Evidence R@5
of at least 70%. Among eligible candidates, select by:

1. higher Answer Accuracy;
2. higher Evidence R@5;
3. lower average token use;
4. lower semantic RRF weight.

Freeze the selected configuration before reading any new formal result. If no
candidate passes the development evidence threshold, stop and archive the
development study without another formal run.

## Formal Run

Run MediMem once with the frozen configuration on:

- LoCoMo: 1,000 frozen questions;
- LongMemEval: 500 questions;
- DialSim: 1,000 frozen questions;
- RHELM: 1,305 questions.

Start two identical Qwen3-VL-8B vLLM services on separate GPUs, ports, and PID
files. Divide the four frozen jobs across the two services. Do not rerun
Direct, Static RAG, A-MEM, DDO, G-Memory, MemInsight, MemoryOS, Mem0, or Letta.
Missing sample IDs may be completed only with the identical frozen
configuration.

## Validation and Leakage Controls

Unit tests must cover:

- deterministic long-card windowing;
- normalized semantic scoring and maximum-window aggregation;
- stable weighted RRF ranking and tie breaks;
- default lexical backward compatibility;
- cache isolation across encoder, revision/hash, window, and RRF weight;
- CLI and manifest propagation;
- absence of question, answer, gold evidence, judge, and evaluator fields from
  the static card index;
- top-five enforcement and the full-context shortcut guard.

Development and formal artifacts must contain the complete unique sample ID
set. The following counts must all be zero: blocked, fallback, API error,
worker failure, judge missing, guard failure, and leakage. Validation manifests
must pass before metrics are accepted.

## Acceptance and Paper Action

Recompute per-source metrics, the four-source mean Answer Accuracy, the
three-source mean Evidence R@5, LoCoMo QA F1 and BLEU-1, token use, latency, and
10,000 paired bootstrap resamples against Mem0.

Update Table 2, the associated analysis, abstract, conclusion, project
contexts, result archive, PDF, and Overleaf package only if both aggregate
metrics are strictly greater than the Mem0 thresholds. Reuse the same formal
LoCoMo predictions for the appendix. If either aggregate fails, keep the
current paper unchanged and archive the run as an unadopted experiment.

After artifacts and gates are verified, stop only the two vLLM process groups
started for this experiment. Confirm both ports are released and both GPUs
return to zero memory use.
