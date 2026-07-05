# LoCoMo memory batch-save optimization

## Goal

Reduce LoCoMo cache preparation time without changing memory-card contents, ordering, IDs, retrieval, prompts, model calls, samples, or metrics.

## Design

`build_locomo_memory_store` continues to create one card per valid turn through `MemoryStore.write_card`. It enriches each in-memory card with speaker, entity, and temporal metadata exactly as before. Instead of rewriting the complete JSONL file after every enriched card, it performs one `store.save()` after all turns are processed. `write_card` may append intermediate rows during construction, but the final save atomically rewrites the same ordered card list with all metadata.

Existing non-empty stores remain unchanged and return immediately. Empty or all-invalid conversations do not trigger a redundant save.

## Validation

- A regression test counts full-store saves and requires exactly one save for a multi-turn conversation.
- The test reloads the JSONL file and verifies enriched metadata persists.
- Run the benchmark, memory, and metrics tests before publishing.
- On the server, rebuild the strict smoke and require identical validation rules: all three methods present, blocked/fallback/API failures equal zero.

## Rollback

Revert the optimization commit and restart from a new run directory. Incomplete run directories remain as provenance and are never merged into formal results.
