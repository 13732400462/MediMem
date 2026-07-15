import json

from mem_ehr_agent.benchmark import (
    FULL_CONTEXT_SHORTCUT_ADVICE,
    amem_retrieved_context,
    assert_no_full_context_shortcut,
    build_locomo_memory_store,
    locomo_expanded_query,
    evaluate_benchmark_predictions,
    evaluate_locomo_benchmark_predictions,
    evidence_recall_at5,
    load_longmemeval_samples,
    load_locomo_samples,
    load_rhelm_samples,
    locomo_memory_path,
    load_frozen_sample_ids,
    parse_int_list,
    parse_methods,
    qa_prompt,
    retrieve_locomo_memories,
    run_locomo_official_wrapper,
    run_locomo_ours_memory_pipeline,
    static_rag_retrieved_context,
    select_frozen_samples,
    validate_horizontal_run,
)
from mem_ehr_agent.memory import MemoryStore


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
    root = tmp_path / "data"
    for name in ("QA_final", "conversations/Alice", "emails/Alice", "attachments/Alice"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "conversations/Alice/2025-01-01.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "I moved to Boston."}]}), encoding="utf-8"
    )
    (root / "emails/Alice/mail.txt").write_text("Subject: travel\nBoston plans", encoding="utf-8")
    (root / "attachments/Alice/note.md").write_text("# Note\nBoston", encoding="utf-8")
    (root / "QA_final/low_score_qa_Alice_all_validated.jsonl").write_text(
        json.dumps(
            {
                "id": "fact_1",
                "question": "Where did I move?",
                "answer": "Boston",
                "question_date": "2025-02-01",
                "question_type": "fact",
                "supporting_evidence": ["2025-01-01:0"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_rhelm_samples(root)
    assert len(samples) == 1
    assert samples[0]["evidence"] == ["2025-01-01:0"]
    assert {turn["speaker"] for turn in samples[0]["turns"]} >= {"user", "email", "attachment"}


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
