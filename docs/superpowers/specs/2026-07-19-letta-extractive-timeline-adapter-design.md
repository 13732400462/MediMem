# Fast Extractive Letta Timeline Adapter

## Goal

Complete the four non-medical Letta benchmark datasets in approximately 12 hours on two RTX 4090 GPUs while retaining the full frozen sample sets, Qwen3-VL-8B answer model, 6500-character Letta core-memory budget, answer judging, and existing metrics.

## Protocol Status

This is a timeline protocol adapter, not an official native reproduction of each dataset in Letta. The table will show the clean method name `Letta/MemGPT`; the adapter qualification will appear once outside the table with the other protocol-adapter notes.

Results from the earlier rolling-LLM ingestion runs cannot be mixed with this adapter. All four Letta dataset cells, including LoCoMo, must be regenerated with the same adapter.

## Query-Independent Extractive Memory

The adapter receives only the frozen timeline turns. It must never inspect benchmark questions, answers, evidence labels, or method-specific evaluation metadata.

Each turn is converted to a dated chronological record containing its session/date reference, speaker, and normalized content. The bounded memory builder then:

1. reserves a small header describing the extraction;
2. groups records by session so every session receives chronological coverage;
3. assigns a minimum per-session character budget when all sessions fit;
4. distributes remaining capacity by session content size;
5. samples sessions uniformly when the memory cannot represent every session, then samples records uniformly within retained sessions;
6. truncates individual records only after session and temporal coverage are preserved;
7. emits selected records in global chronological order at no more than 6500 characters.

The output is deterministic for the same timeline and independent of all questions. It is written through the existing official Letta block API. Letta `AgentLoop` remains responsible for answering every question.

## Execution

- Add `--ingestion-mode rolling_llm|extractive`, retaining `rolling_llm` as the compatibility default.
- Formal fast runs explicitly select `extractive`.
- Run LoCoMo and DialSim on one GPU and RHELM and LongMemEval on the other, with eight workers per active dataset.
- Do not resume predictions produced by `rolling_llm`.
- Preserve the existing answer parser, pure-answer retry, judging, validation gates, sample manifests, and seeds.

## Manifest and Audit

Record the ingestion mode, extraction algorithm version, input turn count, retained record count, retained session count, output character count, and whether any record was truncated. Worker progress files must include these diagnostics for every conversation.

## Verification

- Unit tests establish determinism, length bounds, temporal/session coverage, question independence, and behavior for empty and oversized turns.
- A real one-sample smoke test must pass through Letta, judging, and the formal validation gate.
- Formal launch is allowed only after both local/static checks and server smoke checks pass.
- After launch, verify two active 8-worker datasets, both GPUs busy, all four manifests using `extractive`, and prediction counts increasing.
