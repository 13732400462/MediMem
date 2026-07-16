from mem_ehr_agent.amem_baseline import (
    SourceAlignedAMEMSystem,
    SourceAlignedEmbeddingRetriever,
    build_amem_context,
    cosine,
    event_to_note_fields,
    hash_embedding,
    run_amem_adapter,
)
from mem_ehr_agent.data_builder import build_case
from mem_ehr_agent.data_sources import fallback_pmc_rows, fallback_pmoa_rows


def test_source_aligned_amem_system_links_and_retrieves_notes():
    case = build_case(fallback_pmoa_rows()[0], fallback_pmc_rows()[0], 1)
    system = SourceAlignedAMEMSystem()
    for event in case["events"]:
        system.add_note(**event_to_note_fields(case, event))
    memory_text = system.find_related_memories_raw("final diagnosis pathology imaging", k=3)
    assert "memory content:" in memory_text
    assert "memory context:" in memory_text
    assert len(system.retriever.search("diagnosis", k=3)) <= 3


def test_vectorized_retriever_preserves_stable_cosine_ranking():
    retriever = SourceAlignedEmbeddingRetriever()
    retriever.model = None
    documents = ["alpha beta", "alpha gamma", "delta epsilon", "alpha beta"]
    retriever.add_documents(documents)
    query = "alpha beta"
    query_embedding = hash_embedding(query)
    expected = sorted(
        range(len(documents)),
        key=lambda index: (-cosine(query_embedding, retriever.embeddings[index]), index),
    )[:3]
    assert retriever.search(query, 3) == expected


def test_amem_adapter_prediction_schema_without_client():
    case = build_case(fallback_pmoa_rows()[0], fallback_pmc_rows()[0], 1)
    pred = run_amem_adapter(case, None)
    assert pred["method"] == "baseline_amem_adapter"
    assert pred["retrieved_memory_count"] > 0
    assert pred["baseline_source"]["embedding_model"] == "all-MiniLM-L6-v2"
    assert "memory_layer.py" in pred["baseline_source"]["source_files"]


def test_amem_context_contains_retrieved_notes():
    case = build_case(fallback_pmoa_rows()[0], fallback_pmc_rows()[0], 1)
    context, count = build_amem_context(case, top_k=2)
    assert count == 2
    assert "[A-MEM SOURCE-ALIGNED RETRIEVED MEMORY NOTES]" in context
