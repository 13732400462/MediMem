import json
import sys
import types

from mem_ehr_agent.benchmark import (
    FULL_CONTEXT_SHORTCUT_ADVICE,
    amem_retrieved_context,
    assert_no_full_context_shortcut,
    build_timeline_semantic_index,
    build_locomo_memory_store,
    locomo_expanded_query,
    evaluate_benchmark_predictions,
    evaluate_locomo_benchmark_predictions,
    evidence_recall_at5,
    load_longmemeval_samples,
    load_dialsim_samples,
    load_locomo_samples,
    load_rhelm_samples,
    locomo_cache_key,
    locomo_memory_path,
    load_frozen_sample_ids,
    maximum_window_similarity,
    normalize_hierarchical_retriever_config,
    normalize_timeline_retriever_config,
    normalized_vector_cosine,
    parse_int_list,
    parse_methods,
    qa_prompt,
    retrieve_locomo_memories,
    retrieve_hierarchical_timeline_bundles,
    run_locomo_official_wrapper,
    run_locomo_ours_memory_pipeline,
    static_rag_retrieved_context,
    timeline_card_windows,
    select_frozen_samples,
    validate_horizontal_run,
    weighted_rrf_scores,
    judge_prediction,
)
from mem_ehr_agent.cli import build_parser
from mem_ehr_agent.memory import MemoryStore
from scripts.screen_table2_hierarchical_retrieval import (
    retrieve_hierarchical_grid,
)


def test_parse_methods_normalizes_comma_list():
    assert parse_methods("ours, amem") == ["ours", "amem"]


def test_qa_prompt_is_blind_to_method_name():
    sample = {"dataset": "longmemeval", "question": "When?"}
    prompt = qa_prompt(sample, "secret_method", "Timeline context")
    assert "secret_method" not in prompt[1]["content"]
    assert "[METHOD]" not in prompt[1]["content"]


def test_frozen_sample_manifest_preserves_declared_order(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"sample_ids": ["s2", "s1"]}), encoding="utf-8")
    ids = load_frozen_sample_ids(path)
    samples = [{"sample_id": "s1"}, {"sample_id": "s2"}]
    assert [row["sample_id"] for row in select_frozen_samples(samples, ids)] == ["s2", "s1"]


def test_frozen_sample_manifest_accepts_nested_test_split(tmp_path):
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps({"test": {"sample_ids": ["s1"]}}), encoding="utf-8")
    assert load_frozen_sample_ids(path) == ["s1"]


def test_load_dialsim_can_resolve_frozen_ids_without_random_sample_size(
    tmp_path, monkeypatch
):
    subset = tmp_path / "subset_a"
    subset.mkdir()
    parquet_path = subset / "part.parquet"
    parquet_path.touch()
    family = "easy_qs_ans_w_time"
    rows = [
        {
            "Episode": "e0",
            "Session": 1,
            "Date": "2025-01-01",
            "Script": "first",
            f"{family}_questions": ["q0"],
            f"{family}_answers": ["a0"],
            f"{family}_options": [[]],
            f"{family}_idxes": [0],
        },
        {
            "Episode": "e1",
            "Session": 2,
            "Date": "2025-01-02",
            "Script": "second",
            f"{family}_questions": ["q1"],
            f"{family}_answers": ["a1"],
            f"{family}_options": [[]],
            f"{family}_idxes": [1],
        },
    ]

    class FakeBatch:
        def __init__(self, row):
            self.row = row

        def to_pylist(self):
            return [self.row]

    class FakeParquet:
        schema_arrow = types.SimpleNamespace(names=list(rows[0]))

        def __init__(self, _path):
            pass

        def iter_batches(self, **_kwargs):
            return [FakeBatch(row) for row in rows]

    pyarrow = types.ModuleType("pyarrow")
    parquet = types.ModuleType("pyarrow.parquet")
    parquet.ParquetFile = FakeParquet
    pyarrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)

    frozen_id = f"subset_a__r00001__s0002__{family}__00000"
    samples = load_dialsim_samples(tmp_path, required_sample_ids=[frozen_id])
    assert [sample["sample_id"] for sample in samples] == [frozen_id]
    assert samples[0]["turns"][-1]["text"] == "second"


def test_load_longmemeval_preserves_sessions_dates_and_evidence(tmp_path):
    path = tmp_path / "longmemeval_s_cleaned.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "temporal-reasoning",
                    "question": "When was the trip?",
                    "answer": "Monday",
                    "question_date": "2025-01-02",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2025-01-01"],
                    "haystack_sessions": [[{"role": "user", "content": "The trip was Monday."}]],
                    "answer_session_ids": ["s1"],
                }
            ]
        ),
        encoding="utf-8",
    )
    samples = load_longmemeval_samples(path)
    assert samples[0]["sample_id"] == "q1"
    assert samples[0]["evidence"] == ["s1"]
    assert samples[0]["turns"][0]["evidence_refs"] == ["s1", "s1:0"]
    assert samples[0]["turns"][0]["session_date"] == "2025-01-01"


def test_load_rhelm_combines_conversation_email_attachment_and_qa(tmp_path):
    repo = tmp_path / "rhelm"
    root = repo / "data"
    for name in ("QA_final", "conversations/Alice", "emails/Alice", "attachments/Alice"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "conversations/Alice/conversation_2025_01_01.json").write_text(
        json.dumps(
            {
                "date": "2025-01-01",
                "conversation": [
                    {"turn": 1, "timestamp": "2025-01-01T09:00:00", "user": "I moved to Boston.", "assistant": "Noted."}
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "emails/Alice/01_email_2025_01_01.txt").write_text("Subject: travel\nBoston plans", encoding="utf-8")
    (root / "attachments/Alice/note.md").write_text("# Note\nBoston", encoding="utf-8")
    (root / "QA_final/low_score_qa_Alice_all_validated.jsonl").write_text(
        json.dumps(
            {
                "id": "fact_1",
                "question": "Where did I move?",
                "answer": "Boston",
                "question_date": "2025-02-01",
                "question_type": "fact",
                "supporting_evidence": ["2025-01-01:1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_rhelm_samples(repo)
    assert len(samples) == 1
    assert samples[0]["evidence"] == ["2025-01-01:1"]
    assert {turn["speaker"] for turn in samples[0]["turns"]} >= {"user", "email", "attachment"}
    assert "2025-01-01:1" in samples[0]["turns"][0]["evidence_refs"]
    assert any("Emails_2025-01-01:Email" in turn["evidence_refs"] for turn in samples[0]["turns"])


def test_static_rag_and_evidence_recall_use_actual_top_five_refs():
    sample = {
        "sample_id": "q1",
        "dataset": "longmemeval",
        "question": "Where did Alice move?",
        "answer": "Boston",
        "evidence": ["s2"],
        "turns": [
            {"text": "Unrelated note.", "evidence_refs": ["s1"]},
            {"text": "Alice moved to Boston.", "evidence_refs": ["s2"]},
        ],
    }
    _, refs, refs_at5 = static_rag_retrieved_context(sample, top_k=2)
    assert "s2" in refs
    assert evidence_recall_at5(sample, {"retrieved_evidence_refs_at5": refs_at5}) == 1.0


def test_load_locomo_samples_from_official_json(tmp_path):
    path = tmp_path / "locomo.json"
    payload = [
        {
            "qa": [{"question": "When did A visit?", "answer": "Monday", "category": 2, "evidence": ["D1:0"]}],
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "Monday",
                "session_1": [{"speaker": "A", "dia_id": "D1:0", "text": "I visited on Monday."}],
            },
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    samples = load_locomo_samples(path)
    assert len(samples) == 1
    assert samples[0]["dataset"] == "locomo"
    assert "I visited on Monday" in samples[0]["context"]
    assert samples[0]["conversation_id"] == "conv-0"
    assert samples[0]["category"] == 2
    assert samples[0]["category_name"] == "temporal"
    assert samples[0]["turns"][0]["event_id"] == "D1:0"
    assert samples[0]["turns"][0]["tags"] == ["dialogue", "locomo"]
    assert samples[0]["turns"][0]["evidence_refs"] == ["D1:0"]


def test_load_locomo_normalizes_combined_and_zero_padded_evidence(tmp_path):
    path = tmp_path / "locomo.json"
    payload = [
        {
            "qa": [{"question": "What?", "answer": "x", "category": 4, "evidence": ["D1:00; D1:1", "D"]}],
            "conversation": {
                "session_1_date_time": "Monday",
                "session_1": [
                    {"speaker": "A", "dia_id": "D1:0", "text": "first"},
                    {"speaker": "B", "dia_id": "D1:1", "text": "second"},
                ],
            },
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_locomo_samples(path)[0]["evidence"] == ["D1:0", "D1:1"]


def test_locomo_turns_build_jsonl_memory_cards(tmp_path):
    sample = {
        "sample_id": "conv-1__qa_0000",
        "conversation_id": "conv-1",
        "turns": [
            {
                "event_id": "D1:0",
                "dia_id": "D1:0",
                "session": "1",
                "session_date": "Monday",
                "time": "D1:0",
                "speaker": "A",
                "text": "A visited Boston on Monday.",
                "tags": ["dialogue", "locomo"],
            }
        ],
    }
    path = locomo_memory_path(tmp_path, "conv-1")
    store = build_locomo_memory_store(sample, path)
    assert path.exists()
    assert len(store.cards) == 1
    card = store.cards[0]
    assert card["memory_id"]
    assert card["summary"] == "A visited Boston on Monday."
    assert card["evidence_refs"] == ["D1:0"]
    assert card["time_scope"]["session"] == "1"
    assert card["entities"] == ["Boston", "Monday"]
    assert "Monday" in card["temporal_markers"]
    assert "locomo" in card["tags"]


def test_locomo_memory_store_saves_enriched_cards_once(tmp_path, monkeypatch):
    sample = {
        "sample_id": "conv-batch__qa_0000",
        "conversation_id": "conv-batch",
        "turns": [
            {
                "event_id": "D1:0",
                "dia_id": "D1:0",
                "session": "1",
                "session_date": "Monday",
                "time": "D1:0",
                "speaker": "A",
                "text": "Alice visited Boston on Monday.",
            },
            {
                "event_id": "D1:1",
                "dia_id": "D1:1",
                "session": "1",
                "session_date": "Monday",
                "time": "D1:1",
                "speaker": "B",
                "text": "Bob scheduled follow-up on Tuesday.",
            },
        ],
    }
    save_calls = 0
    original_save = MemoryStore.save

    def counted_save(store):
        nonlocal save_calls
        save_calls += 1
        return original_save(store)

    monkeypatch.setattr(MemoryStore, "save", counted_save)
    path = locomo_memory_path(tmp_path, "conv-batch")
    store = build_locomo_memory_store(sample, path)

    assert save_calls == 1
    assert len(store.cards) == 2
    reloaded = MemoryStore.load("conv-batch", path)
    assert reloaded.cards == store.cards
    assert reloaded.cards[0]["entities"] == ["Alice", "Boston", "Monday"]
    assert "Tuesday" in reloaded.cards[1]["temporal_markers"]


def test_session_chunk_memory_cards_preserve_order_refs_and_hide_labels(tmp_path):
    sample = {
        "sample_id": "conv-session__qa_0000",
        "conversation_id": "conv-session",
        "dataset": "locomo",
        "question": "Where did Alice go?",
        "answer": "SECRET GOLD ANSWER",
        "evidence": ["SECRET GOLD REF"],
        "turns": [
            {
                "speaker": "Alice",
                "text": "First visible turn.",
                "session": "1",
                "session_date": "2025-01-01",
                "evidence_refs": ["D1:0"],
            },
            {
                "speaker": "Bob",
                "text": "Second visible turn.",
                "session": "1",
                "session_date": "2025-01-01",
                "evidence_refs": ["D1:1"],
            },
            {
                "speaker": "Alice",
                "text": "Third session turn.",
                "session": "2",
                "session_date": "2025-01-02",
                "evidence_refs": ["D2:0"],
            },
        ],
    }

    store = build_locomo_memory_store(
        sample,
        tmp_path / "session.memory.jsonl",
        card_granularity="session_chunk",
        card_max_chars=4000,
    )

    assert len(store.cards) == 2
    first = store.cards[0]
    assert first["time_scope"]["session"] == "1"
    assert first["evidence_refs"] == ["D1:0", "D1:1"]
    assert first["summary"].index("First visible turn.") < first["summary"].index("Second visible turn.")
    serialized = json.dumps(store.cards, ensure_ascii=False)
    assert "SECRET GOLD ANSWER" not in serialized
    assert "SECRET GOLD REF" not in serialized


def test_session_chunk_memory_cards_split_without_losing_turns(tmp_path):
    sample = {
        "sample_id": "conv-split__qa_0000",
        "conversation_id": "conv-split",
        "dataset": "locomo",
        "turns": [
            {
                "speaker": "Alice",
                "text": f"Visible turn {index} " + ("x" * 45),
                "session": "1",
                "evidence_refs": [f"D1:{index}"],
            }
            for index in range(4)
        ],
    }

    store = build_locomo_memory_store(
        sample,
        tmp_path / "split.memory.jsonl",
        card_granularity="session_chunk",
        card_max_chars=100,
    )

    assert len(store.cards) == 4
    assert [card["evidence_refs"] for card in store.cards] == [[f"D1:{index}"] for index in range(4)]
    assert [card["chunk_index"] for card in store.cards] == [0, 1, 2, 3]


def test_locomo_cache_key_isolates_card_granularity_and_size():
    sample = {"sample_id": "q1", "conversation_id": "c1", "turns": [{"text": "visible"}]}
    turn_key = locomo_cache_key(
        sample,
        dataset_hash="abc123",
        top_k=5,
        coarse_k=5,
        card_granularity="turn",
        card_max_chars=4000,
    )
    session_4000 = locomo_cache_key(
        sample,
        dataset_hash="abc123",
        top_k=5,
        coarse_k=5,
        card_granularity="session_chunk",
        card_max_chars=4000,
    )
    session_6500 = locomo_cache_key(
        sample,
        dataset_hash="abc123",
        top_k=5,
        coarse_k=5,
        card_granularity="session_chunk",
        card_max_chars=6500,
    )
    hybrid_weight_1 = locomo_cache_key(
        sample,
        dataset_hash="abc123",
        top_k=5,
        coarse_k=5,
        card_granularity="session_chunk",
        card_max_chars=4000,
        retriever="hybrid_bge",
        semantic_rrf_weight=1.0,
    )
    hybrid_weight_3 = locomo_cache_key(
        sample,
        dataset_hash="abc123",
        top_k=5,
        coarse_k=5,
        card_granularity="session_chunk",
        card_max_chars=4000,
        retriever="hybrid_bge",
        semantic_rrf_weight=3.0,
    )
    hierarchical_bundle_5 = locomo_cache_key(
        sample,
        dataset_hash="abc123",
        top_k=5,
        coarse_k=5,
        card_granularity="session_chunk",
        card_max_chars=4000,
        retriever="hierarchical_bge",
        hierarchical_bundle_k=5,
    )
    hierarchical_bundle_8 = locomo_cache_key(
        sample,
        dataset_hash="abc123",
        top_k=5,
        coarse_k=5,
        card_granularity="session_chunk",
        card_max_chars=4000,
        retriever="hierarchical_bge",
        hierarchical_bundle_k=8,
    )

    assert len(
        {
            turn_key,
            session_4000,
            session_6500,
            hybrid_weight_1,
            hybrid_weight_3,
            hierarchical_bundle_5,
            hierarchical_bundle_8,
        }
    ) == 7


def test_timeline_retriever_config_normalization_and_validation():
    assert normalize_timeline_retriever_config("HYBRID_BGE", "model", 2, 128) == (
        "hybrid_bge",
        "model",
        2.0,
        128,
    )
    try:
        normalize_timeline_retriever_config("unknown", "model", 1, 128)
    except ValueError as exc:
        assert "Unsupported timeline retriever" in str(exc)
    else:
        raise AssertionError("unknown retriever should fail")
    assert normalize_hierarchical_retriever_config(8, 5, 1, 900) == (8, 5, 1, 900)


def test_timeline_card_windows_are_stable_and_lossless():
    text = "one two three four five"
    assert timeline_card_windows(text, max_tokens=2) == ["one two", "three four", "five"]
    assert " ".join(timeline_card_windows(text, max_tokens=2)) == text


def test_semantic_similarity_and_weighted_rrf_are_deterministic():
    assert normalized_vector_cosine([1.0, 0.0], [2.0, 0.0]) == 1.0
    assert maximum_window_similarity([1.0, 0.0], [[0.0, 1.0], [3.0, 0.0]]) == 1.0
    scores = weighted_rrf_scores(["lex", "sem"], ["sem", "lex"], semantic_weight=3.0)
    assert scores["sem"] > scores["lex"]
    assert scores == weighted_rrf_scores(["lex", "sem"], ["sem", "lex"], semantic_weight=3.0)


class FakeSemanticEncoder:
    tokenizer = None
    identity = {
        "model": "fake-bge",
        "revision": "test",
        "artifact_sha256": "0" * 64,
    }

    def encode(self, texts, *, is_query=False):
        vectors = []
        for text in texts:
            lower = text.lower()
            semantic = 1.0 if any(term in lower for term in ("automobile", "car", "vehicle")) else 0.0
            distractor = 1.0 if "rumor" in lower else 0.0
            vectors.append([semantic, distractor])
        return vectors


def test_offline_hierarchical_grid_matches_individual_retrievals(tmp_path):
    sample = {
        "sample_id": "q-grid",
        "conversation_id": "c-grid",
        "dataset": "locomo",
        "category_name": "multi-hop",
        "question": "How are the car and vehicle plans related?",
        "turns": [
            {
                "speaker": "A",
                "text": "The red car was purchased.",
                "session": "s1",
                "session_date": "2025-01-01",
                "evidence_refs": ["D1:0"],
            },
            {
                "speaker": "A",
                "text": "The vehicle needed repairs.",
                "session": "s1",
                "session_date": "2025-01-01",
                "evidence_refs": ["D1:1"],
            },
            {
                "speaker": "B",
                "text": "A later vehicle trip was planned.",
                "session": "s2",
                "session_date": "2025-02-01",
                "evidence_refs": ["D2:0"],
            },
        ],
    }
    store = build_locomo_memory_store(
        sample,
        tmp_path / "grid.memory.jsonl",
        card_granularity="session_chunk",
        card_max_chars=4000,
    )
    encoder = FakeSemanticEncoder()
    grid = retrieve_hierarchical_grid(
        store,
        sample,
        semantic_encoder=encoder,
        semantic_rrf_weight=2.0,
        embedding_window_tokens=16,
        parent_ks=[1, 2],
        neighbor_radii=[0, 1],
        bundle_max_chars_values=[400, 900],
        bundle_k=2,
    )
    for parent_k in (1, 2):
        for neighbor_radius in (0, 1):
            for bundle_max_chars in (400, 900):
                expected = retrieve_hierarchical_timeline_bundles(
                    store,
                    sample,
                    semantic_encoder=encoder,
                    semantic_rrf_weight=2.0,
                    embedding_window_tokens=16,
                    parent_k=parent_k,
                    bundle_k=2,
                    neighbor_radius=neighbor_radius,
                    bundle_max_chars=bundle_max_chars,
                )
                assert (
                    grid[(parent_k, neighbor_radius, bundle_max_chars)]
                    == expected
                )


def test_hybrid_retrieval_uses_static_semantic_index_without_labels(tmp_path):
    sample = {
        "sample_id": "q1",
        "conversation_id": "c1",
        "dataset": "locomo",
        "question": "Which automobile was purchased?",
        "answer": "SECRET ANSWER",
        "evidence": ["SECRET REF"],
        "turns": [
            {
                "speaker": "A",
                "text": "A rumor mentioned the automobile.",
                "session": "1",
                "evidence_refs": ["D1"],
            },
            {
                "speaker": "B",
                "text": "Maya purchased a red car.",
                "session": "2",
                "evidence_refs": ["D2"],
            },
        ],
    }
    store = build_locomo_memory_store(
        sample,
        tmp_path / "hybrid.memory.jsonl",
        card_granularity="session_chunk",
        card_max_chars=4000,
    )
    encoder = FakeSemanticEncoder()
    index = build_timeline_semantic_index(store, encoder, embedding_window_tokens=16)
    serialized = json.dumps(index)
    assert "SECRET ANSWER" not in serialized
    assert "SECRET REF" not in serialized
    retrieved = retrieve_locomo_memories(
        store,
        sample,
        top_k=1,
        coarse_k=2,
        retriever="hybrid_bge",
        semantic_encoder=encoder,
        semantic_rrf_weight=3.0,
        embedding_window_tokens=16,
    )
    assert retrieved[0]["evidence_refs"] == ["D2"]
    assert "semantic_retrieval_score" in retrieved[0]


def test_hierarchical_retrieval_packs_visible_adjacent_turns_and_real_refs(tmp_path):
    sample = {
        "sample_id": "q-hierarchical",
        "conversation_id": "c-hierarchical",
        "dataset": "locomo",
        "category_name": "temporal",
        "question": "Which automobile was purchased before the later trip?",
        "answer": "SECRET GOLD ANSWER",
        "evidence": ["SECRET GOLD REF"],
        "turns": [
            {
                "speaker": "Maya",
                "text": "Maya purchased a red car.",
                "session": "s1",
                "session_date": "2025-01-01",
                "evidence_refs": ["D1:0"],
            },
            {
                "speaker": "Lee",
                "text": "Lee congratulated Maya on the vehicle.",
                "session": "s1",
                "session_date": "2025-01-01",
                "evidence_refs": ["D1:1"],
            },
            {
                "speaker": "Maya",
                "text": "A later trip included an unrelated rumor.",
                "session": "s2",
                "session_date": "2025-02-01",
                "evidence_refs": ["D2:0"],
            },
        ],
    }
    store = build_locomo_memory_store(
        sample,
        tmp_path / "hierarchical.memory.jsonl",
        card_granularity="session_chunk",
        card_max_chars=4000,
    )
    retrieved = retrieve_hierarchical_timeline_bundles(
        store,
        sample,
        semantic_encoder=FakeSemanticEncoder(),
        semantic_rrf_weight=2.0,
        embedding_window_tokens=16,
        parent_k=2,
        bundle_k=2,
        neighbor_radius=1,
        bundle_max_chars=900,
    )

    assert len(retrieved) == 2
    assert [item["time_scope"]["date"] for item in retrieved] == sorted(
        item["time_scope"]["date"] for item in retrieved
    )
    assert any(item["evidence_refs"] == ["D1:0", "D1:1"] for item in retrieved)
    assert all("D2:0" not in item["evidence_refs"] or "D1:1" not in item["evidence_refs"] for item in retrieved)
    serialized = json.dumps(retrieved, ensure_ascii=False)
    assert "SECRET GOLD ANSWER" not in serialized
    assert "SECRET GOLD REF" not in serialized


def test_hierarchical_multihop_prefers_distinct_sessions(tmp_path):
    sample = {
        "sample_id": "q-multihop",
        "conversation_id": "c-multihop",
        "dataset": "locomo",
        "category_name": "multi-hop",
        "question": "How are the car and vehicle plans related?",
        "turns": [
            {
                "speaker": "A",
                "text": "The red car was purchased.",
                "session": "s1",
                "evidence_refs": ["D1:0"],
            },
            {
                "speaker": "A",
                "text": "The vehicle needed repairs.",
                "session": "s1",
                "evidence_refs": ["D1:1"],
            },
            {
                "speaker": "B",
                "text": "A later vehicle trip was planned.",
                "session": "s2",
                "evidence_refs": ["D2:0"],
            },
        ],
    }
    store = build_locomo_memory_store(
        sample,
        tmp_path / "multihop.memory.jsonl",
        card_granularity="session_chunk",
        card_max_chars=4000,
    )
    retrieved = retrieve_hierarchical_timeline_bundles(
        store,
        sample,
        semantic_encoder=FakeSemanticEncoder(),
        semantic_rrf_weight=2.0,
        embedding_window_tokens=16,
        parent_k=2,
        bundle_k=2,
        neighbor_radius=0,
        bundle_max_chars=400,
    )
    assert {item["time_scope"]["session"] for item in retrieved} == {"s1", "s2"}


def test_benchmark_cli_exposes_hierarchical_retrieval_flags():
    args = build_parser().parse_args(
        [
            "benchmark",
            "run",
            "--dataset",
            "locomo",
            "--methods",
            "medimem",
            "--timeline-retriever",
            "hierarchical_bge",
            "--timeline-semantic-rrf-weight",
            "2",
            "--timeline-embedding-window-tokens",
            "128",
            "--timeline-hierarchical-parent-k",
            "12",
            "--timeline-hierarchical-bundle-k",
            "8",
            "--timeline-hierarchical-neighbor-radius",
            "1",
            "--timeline-hierarchical-bundle-max-chars",
            "700",
        ]
    )
    assert args.timeline_retriever == "hierarchical_bge"
    assert args.timeline_semantic_rrf_weight == 2.0
    assert args.timeline_embedding_window_tokens == 128
    assert args.timeline_hierarchical_parent_k == 12
    assert args.timeline_hierarchical_bundle_k == 8
    assert args.timeline_hierarchical_neighbor_radius == 1
    assert args.timeline_hierarchical_bundle_max_chars == 700


def test_parse_int_list_for_top_k_sweep():
    assert parse_int_list("8, 16,32") == [8, 16, 32]


def test_locomo_two_stage_retrieval_uses_expanded_entity_time_text(tmp_path):
    turns = [
        {
            "event_id": "D1:0",
            "dia_id": "D1:0",
            "session": "1",
            "session_date": "Monday",
            "time": "D1:0",
            "speaker": "A",
            "text": "Maya said the appointment moved to Tuesday.",
            "tags": ["dialogue", "locomo"],
        },
        {
            "event_id": "D1:1",
            "dia_id": "D1:1",
            "session": "1",
            "session_date": "Monday",
            "time": "D1:1",
            "speaker": "B",
            "text": "They also discussed a Boston trip.",
            "tags": ["dialogue", "locomo"],
        },
    ]
    sample = {
        "sample_id": "conv-2__qa_0000",
        "conversation_id": "conv-2",
        "dataset": "locomo",
        "category_name": "temporal",
        "turns": turns,
        "question": "When is Maya's appointment?",
    }
    store = build_locomo_memory_store(sample, locomo_memory_path(tmp_path, "conv-2"))
    retrieved = retrieve_locomo_memories(store, sample, top_k=1, coarse_k=2)
    assert len(retrieved) == 1
    assert "Tuesday" in retrieved[0]["summary"]
    assert "time date session" in locomo_expanded_query(sample)


def test_full_context_guard_rejects_marker_and_gives_advice():
    sample = {"context_line_count": 3, "turns": []}
    try:
        assert_no_full_context_shortcut("ours_locomo_memory_pipeline", "[FULL LONG-TERM CONTEXT]\nall text", sample)
    except RuntimeError as exc:
        assert FULL_CONTEXT_SHORTCUT_ADVICE in str(exc)
    else:
        raise AssertionError("guard should reject full-context marker")


def test_full_context_guard_rejects_retrieved_count_matching_context_lines():
    sample = {"context_line_count": 3, "turns": []}
    try:
        assert_no_full_context_shortcut(
            "ours_locomo_memory_pipeline",
            "[RETRIEVED_MEMORY_CARDS]\n...",
            sample,
            retrieved_memory_count=3,
            retrieval_budget=2,
        )
    except RuntimeError as exc:
        assert "retrieved_memory_count equals original context_line_count" in str(exc)
        assert FULL_CONTEXT_SHORTCUT_ADVICE in str(exc)
    else:
        raise AssertionError("guard should reject context-line-count shortcut")


def test_full_context_guard_allows_budgeted_retrieval_on_short_timeline():
    turns = [{"text": "x" * 100}, {"text": "brief"}]
    sample = {"context_line_count": 2, "turns": turns}
    assert_no_full_context_shortcut(
        "medimem_dialsim_memory_pipeline",
        "[RETRIEVED_MEMORY_CARDS]\n" + turns[0]["text"],
        sample,
        retrieved_memory_count=1,
        retrieval_budget=32,
    )


def test_judge_recovers_unambiguous_boolean_from_truncated_json():
    class Result:
        text = '{"correct": true, "reason": "A reason that never closes'
        usage = {"total_tokens": 20}

    class Client:
        def chat(self, *_args, **_kwargs):
            return Result()

    result = judge_prediction(
        {"sample_id": "s1", "question": "Q", "answer": "A"},
        {"method": "m", "answer": "A"},
        Client(),
        require_api=True,
    )
    assert result["judge_correct"] is True
    assert result["judge_parse_recovered"] is True


def test_run_locomo_ours_memory_pipeline_uses_retrieved_cards_not_full_context(tmp_path):
    turns = [
        {
            "event_id": f"D1:{idx}",
            "dia_id": f"D1:{idx}",
            "session": "1",
            "session_date": "Monday",
            "time": f"D1:{idx}",
            "speaker": "A",
            "text": f"Memory turn {idx} says A visited city {idx}.",
            "tags": ["dialogue", "locomo"],
        }
        for idx in range(10)
    ]
    sample = {
        "sample_id": "conv-1__qa_0000",
        "conversation_id": "conv-1",
        "dataset": "locomo",
        "split": "official",
        "context": "\n".join(turn["text"] for turn in turns),
        "context_line_count": 10,
        "turn_count": 10,
        "turns": turns,
        "question": "Which city did A visit in turn 7?",
        "answer": "city 7",
    }
    pred = run_locomo_ours_memory_pipeline(sample, None, run_dir=tmp_path, top_k=3, require_api=False)
    assert pred["method"] == "medimem_locomo_memory_pipeline"
    assert pred["retrieved_memory_count"] <= 3
    assert pred["locomo_top_k"] == 3
    assert pred["locomo_coarse_k"] == 32
    assert pred["memory_card_count"] == 10
    assert "[FULL LONG-TERM CONTEXT]" not in pred.get("fallback_reason", "")
    assert (tmp_path / "memory" / "ours" / "conv-1_conv-1__qa_0000.memory.jsonl").exists()


def test_locomo_official_wrapper_uses_unified_schema_and_adapter_note(tmp_path, monkeypatch):
    repo = tmp_path / "MemoryOS"
    repo.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("MEMORYOS_REPO", str(repo))
    monkeypatch.setenv("MEMORYOS_PY", str(python))
    sample = {
        "sample_id": "conv-1__qa_0000",
        "conversation_id": "conv-1",
        "dataset": "locomo",
        "split": "official",
        "context": "A visited Boston on Monday.",
        "context_line_count": 1,
        "turn_count": 1,
        "turns": [
            {
                "event_id": "D1:0",
                "dia_id": "D1:0",
                "session": "1",
                "session_date": "Monday",
                "time": "D1:0",
                "speaker": "A",
                "text": "A visited Boston on Monday.",
                "tags": ["dialogue", "locomo"],
            }
        ],
        "question": "Where did A visit?",
        "answer": "Boston",
    }
    pred = run_locomo_official_wrapper(sample, "memoryos", None, run_dir=tmp_path, require_api=False)
    assert pred["method"] == "official_memoryos_locomo_wrapper"
    assert pred["baseline_reproduction_level"] == "official_native_locomo_wrapper"
    assert "unified LoCoMo" in pred["adapter_note"]
    assert pred["retrieved_memory_count"] <= 8


def test_amem_runner_uses_source_aligned_memory_flow():
    sample = {
        "sample_id": "conv-1__qa_0000",
        "conversation_id": "conv-1",
        "dataset": "locomo",
        "question": "Where did A visit?",
        "turns": [
            {
                "time": "D1:0",
                "text": "A visited Boston on Monday.",
                "conversation_id": "conv-1",
                "tags": ["dialogue", "locomo"],
            }
        ],
    }
    context, retrieved = amem_retrieved_context(sample, top_k=8)
    assert retrieved == 1
    assert "[A-MEM RETRIEVED MEMORY]" in context
    assert "A visited Boston" in context


def test_amem_local_method_is_reported_as_wrapper(tmp_path):
    from mem_ehr_agent.benchmark import run_local_method

    sample = {
        "sample_id": "conv-1__qa_0000",
        "conversation_id": "conv-1",
        "dataset": "locomo",
        "split": "official",
        "question": "Where did A visit?",
        "answer": "Boston",
        "turns": [
            {
                "time": "D1:0",
                "text": "A visited Boston on Monday.",
                "conversation_id": "conv-1",
                "tags": ["dialogue", "locomo"],
            }
        ],
    }
    pred = run_local_method(sample, "amem", None, require_api=False, run_dir=tmp_path)
    assert pred["method"] == "official_amem_locomo_wrapper"
    assert pred["baseline_reproduction_level"] == "official_native_locomo_wrapper"
    assert "same LoCoMo samples" in pred["adapter_note"]


def test_horizontal_validation_requires_same_sample_ids_per_method():
    samples = [
        {"sample_id": "s1", "split": "official"},
        {"sample_id": "s2", "split": "official"},
    ]
    predictions = [
        {"sample_id": "s1", "method": "ours"},
        {"sample_id": "s2", "method": "ours"},
        {"sample_id": "s1", "method": "amem"},
    ]
    validation = validate_horizontal_run(samples, predictions)
    assert not validation["passed"]
    assert validation["failures"][0]["method"] == "amem"


def test_benchmark_metrics_are_grouped_by_method_and_split():
    samples = [{"sample_id": "s1", "split": "official", "answer": "Monday", "dataset": "locomo", "category": 2, "category_name": "temporal"}]
    predictions = [
        {
            "sample_id": "s1",
            "method": "ours",
            "split": "official",
            "answer": "Monday",
            "usage": {"total_tokens": 10},
            "retrieved_memory_count": 3,
        }
    ]
    rows = evaluate_benchmark_predictions(samples, predictions)
    assert rows[0]["method"] == "ours"
    assert rows[0]["exact_match"] == 1.0
    assert rows[0]["qa_f1"] == 1.0


def test_locomo_metrics_split_main_auxiliary_and_category():
    samples = [
        {"sample_id": "s1", "split": "official", "answer": "Monday", "dataset": "locomo", "category": 2, "category_name": "temporal"},
        {"sample_id": "s2", "split": "official", "answer": "Boston", "dataset": "locomo", "category": 4, "category_name": "single-hop"},
    ]
    predictions = [
        {"sample_id": "s1", "method": "ours_locomo_memory_pipeline", "split": "official", "answer": "Monday", "usage": {"total_tokens": 10}, "retrieved_memory_count": 8, "guard_passed": True},
        {"sample_id": "s2", "method": "ours_locomo_memory_pipeline", "split": "official", "answer": "Wrong", "usage": {"total_tokens": 30}, "retrieved_memory_count": 8, "guard_passed": True},
    ]
    result = evaluate_locomo_benchmark_predictions(samples, predictions)
    overall = result["overall"][0]
    assert overall["qa_f1"] == 0.5
    assert overall["avg_tokens"] == 20.0
    assert "soft_match" not in overall
    by_category = {row["category"]: row for row in result["by_category"]}
    assert by_category["temporal"]["n"] == 1
    assert by_category["single-hop"]["n"] == 1
    assert {"multi-hop", "temporal", "open-domain", "single-hop", "adversarial"} <= set(by_category)
    assert by_category["adversarial"]["n"] == 0
    auxiliary = result["auxiliary"][0]
    assert "retrieved_memory_count" in auxiliary
    assert "soft_match" in auxiliary
    assert "sbert_similarity" in result["official_style"][0]
    assert "bert_f1" in result["skipped_metrics"]
