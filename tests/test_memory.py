from mem_ehr_agent.memory import MemoryStore


def test_jsonl_memory_state_transition(tmp_path):
    store = MemoryStore.load("p1", tmp_path / "memory.jsonl")
    store.write_card(
        summary="Initial working diagnosis was viral syndrome.",
        evidence_refs=[],
        time_scope={"start": 0, "end": 0},
        confidence=0.3,
        tags=["initial"],
    )
    touched = store.invalidate_or_discard("viral syndrome", "Discard", "contradicted")
    assert touched
    reloaded = MemoryStore.load("p1", tmp_path / "memory.jsonl")
    assert reloaded.cards[0]["status"] == "discarded"

