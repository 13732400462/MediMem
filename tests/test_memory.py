from mem_ehr_agent.llm import LLMResult
from mem_ehr_agent.memory import MemoryStore, apply_critique
from mem_ehr_agent.agents import (
    case_context,
    compact_case_context,
    diagnosis_event_candidates,
    feature_state,
    prefer_visible_diagnosis_candidate,
    pollution_memory_context,
    resolve_top_k,
    run_single_cot_agent,
    safe_memory_ops_for_prompt,
    source_aligned_evidence_notes,
)


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


def test_dynamic_top_k_uses_complexity_and_evidence():
    simple = {"events": [{"text": "short symptom note"} for _ in range(10)]}
    medium = {"events": [{"text": "routine visit note"} for _ in range(25)]}
    long_case = {
        "events": [{"time": 0, "text": "plain event"} for _ in range(82)]
        + [
            {"time": 10, "text": "pathology confirmed malignancy"},
            {"time": 20, "text": "treatment therapy started"},
        ]
    }
    strategy = {"features": {"top_k_is_auto": True}}
    assert resolve_top_k(simple, strategy) == 3
    assert resolve_top_k(medium, strategy) == 5
    assert resolve_top_k(long_case, strategy) == 8


def test_ablation_feature_state_and_fixed_top_k():
    strategy = {
        "top_k": 3,
        "fallback_top_k": 3,
        "features": {
            "top_k_is_auto": True,
            "disable_dynamic_top_k": True,
            "disable_normalization": True,
            "disable_memory_cleaning": True,
        },
    }
    assert resolve_top_k({"events": [{"text": "pathology diagnosis"} for _ in range(90)]}, strategy) == 3
    assert feature_state(strategy) == {
        "diagnosis_normalization": False,
        "dynamic_top_k": False,
        "memory_cleaning": False,
        "critic_op_guard": True,
        "evidence_note_injection": True,
        "profile_adaptive_memory_cleaning": True,
        "profile_adaptive_evidence_notes": False,
        "counterfactual_verification": True,
    }


def test_revise_marks_old_memory_superseded_and_writes_replacement(tmp_path):
    store = MemoryStore.load("p1", tmp_path / "memory.jsonl")
    store.write_card(
        summary="Early working impression: viral syndrome was plausible because fever was present.",
        evidence_refs=["ev_000"],
        time_scope={"start": 0, "end": 0},
        confidence=0.35,
        tags=["initial"],
    )
    touched, revised = store.revise(
        target_text="viral syndrome",
        revised_summary="Viral syndrome was an early time-limited interpretation superseded by biopsy.",
        reason="later biopsy",
        preserved_facts=["fever was present"],
        evidence_refs=["ev_000", "ev_005"],
        time_scope={"start": 0, "end": 5},
    )
    assert touched
    assert revised["updated_by_op"] == "Revise"
    reloaded = MemoryStore.load("p1", tmp_path / "memory.jsonl")
    assert reloaded.cards[0]["status"] == "superseded"
    assert any(card["updated_by_op"] == "Revise" and card["status"] == "active" for card in reloaded.cards)


def test_apply_critique_emits_revision_metadata_without_gold_fields(tmp_path):
    case = {
        "case_id": "case_x",
        "events": [{"event_id": "ev_000", "type": "clinical", "text": "fever"}, {"event_id": "ev_005", "type": "diagnosis", "text": "biopsy confirmed lymphoma"}],
        "poison_records": [
            {
                "poison_id": "poison_x",
                "text": "Early working impression: viral syndrome was plausible because fever was present.",
                "pollution_type": "outdated_initial_diagnosis",
                "claim_type": "interpretation",
                "source_event_id": "ev_000",
                "supporting_evidence": ["fever was present"],
            }
        ],
    }
    store = MemoryStore.load("case_x", tmp_path / "memory.jsonl")
    store.write_card(
        summary="Early working impression: viral syndrome was plausible because fever was present.",
        evidence_refs=["ev_000"],
        time_scope={"start": 0, "end": 0},
        confidence=0.35,
        tags=["initial"],
    )
    ops = apply_critique(case, store)
    assert ops[0]["op"] == "Revise"
    assert ops[0]["revised_memory_id"]
    assert ops[0]["preserved_facts"] == ["fever was present"]


def test_apply_critique_discards_cross_patient_memory(tmp_path):
    case = {
        "case_id": "case_x",
        "events": [{"event_id": "ev_000", "type": "clinical", "text": "fever"}],
        "poison_records": [
            {
                "poison_id": "poison_x",
                "text": "Similar case memory: another patient had COVID ARDS; transfer this prior case context.",
                "pollution_type": "similar_case_mistransfer",
                "claim_type": "interpretation",
                "source_event_id": "pmc_other",
                "supporting_evidence": ["another patient had COVID ARDS"],
            }
        ],
    }
    store = MemoryStore.load("case_x", tmp_path / "memory.jsonl")
    store.write_card(
        summary="Similar case memory: another patient had COVID ARDS; transfer this prior case context.",
        evidence_refs=["poison_x"],
        time_scope={},
        confidence=0.2,
        tags=["poison"],
    )
    ops = apply_critique(case, store)
    assert ops[0]["op"] == "Discard"


class FakeCriticClient:
    def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
        return LLMResult(
            text=(
                '{"operations":[{"op":"Revise","target":"poison_x","reason":"Later evidence confirms lymphoma.",'
                '"revised_claim":"Fever supports lymphoma rather than viral syndrome.",'
                '"preserved_facts":["fever was present","lymphoma is likely"]}]}'
            ),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_s=0.0,
        )


class WrongOpCriticClient:
    def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
        return LLMResult(
            text=(
                '{"operations":[{"op":"Discard","target":"poison_x","reason":"Overly aggressive deletion.",'
                '"revised_claim":"","preserved_facts":[]}]}'
            ),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_s=0.0,
        )


def test_critic_guard_forces_partial_truth_to_revise(tmp_path):
    case = {
        "case_id": "case_x",
        "events": [{"event_id": "ev_000", "type": "clinical", "text": "fever"}],
        "poison_records": [
            {
                "poison_id": "poison_x",
                "text": "Early interpretation was viral syndrome because fever was present.",
                "pollution_type": "partial_truth_misleading_memory",
                "source_event_id": "ev_000",
                "supporting_evidence": ["fever was present"],
            }
        ],
    }
    store = MemoryStore.load("case_x", tmp_path / "memory.jsonl")
    store.write_card(
        summary="Early interpretation was viral syndrome because fever was present.",
        evidence_refs=["poison_x"],
        time_scope={},
        confidence=0.2,
        tags=["poison"],
    )
    ops = apply_critique(case, store, WrongOpCriticClient())
    assert ops[0]["op"] == "Revise"
    assert ops[0]["preserved_facts"] == ["fever was present"]
    assert ops[0]["revised_claim"]


def test_critic_guard_can_be_disabled_for_ablation(tmp_path):
    case = {
        "case_id": "case_x",
        "events": [{"event_id": "ev_000", "type": "clinical", "text": "fever"}],
        "poison_records": [
            {
                "poison_id": "poison_x",
                "text": "Early interpretation was viral syndrome because fever was present.",
                "pollution_type": "partial_truth_misleading_memory",
                "source_event_id": "ev_000",
                "supporting_evidence": ["fever was present"],
            }
        ],
    }
    store = MemoryStore.load("case_x", tmp_path / "memory.jsonl")
    store.write_card(
        summary="Early interpretation was viral syndrome because fever was present.",
        evidence_refs=["poison_x"],
        time_scope={},
        confidence=0.2,
        tags=["poison"],
    )
    ops = apply_critique(case, store, WrongOpCriticClient(), enforce_op_guard=False)
    assert ops[0]["op"] == "Discard"


def test_critic_diagnostic_text_is_not_exposed_to_final_prompt_or_active_memory(tmp_path):
    case = {
        "case_id": "case_x",
        "events": [
            {"event_id": "ev_000", "type": "clinical", "text": "fever"},
            {"event_id": "ev_005", "type": "diagnosis", "text": "biopsy confirmed lymphoma"},
        ],
        "poison_records": [
            {
                "poison_id": "poison_x",
                "text": "Early working impression: viral syndrome was plausible because fever was present.",
                "pollution_type": "partial_truth_misleading_memory",
                "claim_type": "interpretation",
                "source_event_id": "ev_000",
                "supporting_evidence": ["fever was present"],
            }
        ],
    }
    store = MemoryStore.load("case_x", tmp_path / "memory.jsonl")
    store.write_card(
        summary="Early working impression: viral syndrome was plausible because fever was present.",
        evidence_refs=["poison_x"],
        time_scope={},
        confidence=0.2,
        tags=["poison"],
    )
    ops = apply_critique(case, store, FakeCriticClient())
    assert "lymphoma" in ops[0]["revised_claim"]

    prompt_ops = safe_memory_ops_for_prompt(case, ops)
    assert "revised_claim" not in prompt_ops[0]
    assert "reason" not in prompt_ops[0]
    assert "preserved_facts" not in prompt_ops[0]
    assert prompt_ops[0]["preserved_fact_count"] > 0
    assert "lymphoma" not in str(prompt_ops).lower()

    active_summaries = " ".join(
        str(card.get("summary", "")) for card in store.cards if card.get("status") == "active"
    )
    assert "lymphoma" not in active_summaries.lower()


def test_source_aligned_evidence_notes_include_active_high_value_cards_only():
    cards = [
        {
            "summary": "biopsy confirmed malignancy",
            "status": "active",
            "tags": ["diagnosis"],
            "time_scope": {"start": 2},
            "evidence_refs": ["ev_2"],
            "confidence": 0.7,
        },
        {
            "summary": "discarded old diagnosis",
            "status": "discarded",
            "tags": ["diagnosis"],
            "time_scope": {"start": 1},
            "evidence_refs": ["ev_old"],
            "confidence": 0.1,
        },
        {
            "summary": "social history note",
            "status": "active",
            "tags": ["clinical"],
            "time_scope": {"start": 3},
            "evidence_refs": ["ev_3"],
            "confidence": 0.4,
        },
    ]
    notes = source_aligned_evidence_notes(cards)
    assert len(notes) == 1
    assert notes[0]["refs"] == ["ev_2"]


def test_runtime_context_filters_target_like_answer_text():
    case = {
        "case_id": "case_x",
        "demographics": {},
        "events": [
            {"time": 0, "type": "clinical", "text": "fever and neck stiffness"},
            {"time": 1, "type": "diagnosis", "text": "Final answer or diagnosis target: meningitis"},
            {"time": 2, "type": "diagnosis", "text": "biopsy confirmed lymphoma"},
        ],
        "synthetic_labs": [],
    }
    context = case_context(case)
    assert "Final answer" not in context
    assert "diagnosis target" not in context
    assert "meningitis" not in context
    assert "biopsy confirmed lymphoma" in context


def test_polluted_memory_context_filters_target_like_answer_text():
    case = {
        "case_id": "case_x",
        "poison_records": [
            {"poison_id": "p1", "pollution_type": "label_leak", "text": "Correct answer: tuberculosis"},
            {"poison_id": "p2", "pollution_type": "stale", "text": "Earlier visit suggested viral syndrome."},
        ],
    }
    context = pollution_memory_context(case)
    assert "Correct answer" not in context
    assert "tuberculosis" not in context
    assert "viral syndrome" in context


def test_source_aligned_evidence_notes_filter_target_like_memory_cards():
    cards = [
        {
            "summary": "SOAP assessment target: pulmonary embolism",
            "status": "active",
            "tags": ["diagnosis"],
            "time_scope": {"start": 1},
            "evidence_refs": ["ev_leak"],
        },
        {
            "summary": "CT angiography showed a filling defect.",
            "status": "active",
            "tags": ["imaging"],
            "time_scope": {"start": 2},
            "evidence_refs": ["ev_2"],
        },
    ]
    notes = source_aligned_evidence_notes(cards)
    assert len(notes) == 1
    assert notes[0]["refs"] == ["ev_2"]
    assert "pulmonary embolism" not in str(notes)


def test_diagnosis_event_candidates_use_visible_timeline_only():
    case = {
        "labels": {"primary_diagnosis": "hidden gold"},
        "events": [
            {"event_id": "ev_1", "time": 1, "type": "clinical", "text": "symptoms"},
            {"event_id": "ev_2", "time": 2, "type": "diagnosis", "text": "dMMR"},
        ],
    }
    candidates = diagnosis_event_candidates(case)
    assert candidates == [{"time": 2, "event_id": "ev_2", "text": "dMMR"}]
    assert "hidden gold" not in str(candidates)


def test_diagnosis_event_candidates_filter_target_like_text():
    case = {
        "events": [
            {"event_id": "ev_leak", "time": 1, "type": "diagnosis", "text": "Doctor assessment target: asthma"},
            {"event_id": "ev_2", "time": 2, "type": "diagnosis", "text": "diagnosed with pneumonia"},
        ],
    }
    candidates = diagnosis_event_candidates(case)
    assert candidates == [{"time": 2, "event_id": "ev_2", "text": "diagnosed with pneumonia"}]
    assert "asthma" not in str(candidates)


def test_single_cot_agents_return_normalized_prediction_json():
    case = {
        "case_id": "case_x",
        "demographics": {},
        "events": [
            {"event_id": "ev_1", "time": 0, "type": "clinical", "text": "fever and cough"},
            {"event_id": "ev_2", "time": 1, "type": "diagnosis", "text": "diagnosed with pneumonia"},
        ],
        "poison_records": [{"poison_id": "p1", "pollution_type": "stale", "text": "Correct answer: asthma"}],
    }
    clean = run_single_cot_agent(case, None)
    polluted = run_single_cot_agent(case, None, polluted=True)
    assert clean["method"] == "baseline_single_cot_agent"
    assert polluted["method"] == "baseline_polluted_single_cot_agent"
    for pred in (clean, polluted):
        assert pred["primary_diagnosis"]
        assert isinstance(pred["diagnosis_list"], list)
        assert isinstance(pred["confidence"], float)
        assert isinstance(pred["evidence"], list)
        assert isinstance(pred["reasoning_summary"], str)


def test_run_baselines_writes_cot_pairs_to_baselines_jsonl(tmp_path, monkeypatch):
    from mem_ehr_agent import optimizer

    def fake_adapter(name, case, client, *, fail_on_llm_error=False, polluted=False):
        method = f"baseline_{'polluted_' if polluted else ''}{name}_adapter"
        return {
            "case_id": case["case_id"],
            "method": method,
            "primary_diagnosis": "pneumonia",
            "diagnosis_list": ["pneumonia"],
            "confidence": 0.7,
            "evidence": [],
            "reasoning_summary": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    monkeypatch.setattr(optimizer, "run_baseline", fake_adapter)
    case = {
        "case_id": "case_x",
        "demographics": {},
        "events": [
            {"event_id": "ev_1", "time": 0, "type": "clinical", "text": "fever and cough"},
            {"event_id": "ev_2", "time": 1, "type": "diagnosis", "text": "diagnosed with pneumonia"},
        ],
        "poison_records": [{"poison_id": "p1", "pollution_type": "stale", "text": "Correct answer: asthma"}],
    }
    preds = optimizer.run_baselines([case], None, tmp_path, max_workers=2, baseline_set="focused")
    methods = {pred["method"] for pred in preds}
    assert "baseline_single_cot_agent" in methods
    assert "baseline_polluted_single_cot_agent" in methods
    assert (tmp_path / "predictions" / "baselines.jsonl").exists()


def test_required_baseline_set_runs_required_unpolluted_pipelines(tmp_path, monkeypatch):
    from mem_ehr_agent import optimizer

    def fake_adapter(name, case, client, *, fail_on_llm_error=False, polluted=False):
        method = f"baseline_{'polluted_' if polluted else ''}{name}_adapter"
        return {
            "case_id": case["case_id"],
            "method": method,
            "primary_diagnosis": "pneumonia",
            "diagnosis_list": ["pneumonia"],
            "confidence": 0.7,
            "evidence": [],
            "reasoning_summary": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    monkeypatch.setattr(optimizer, "run_baseline", fake_adapter)
    case = {
        "case_id": "case_required",
        "demographics": {},
        "events": [
            {"event_id": "ev_1", "time": 0, "type": "clinical", "text": "fever and cough"},
            {"event_id": "ev_2", "time": 1, "type": "diagnosis", "text": "diagnosed with pneumonia"},
        ],
        "poison_records": [{"poison_id": "p1", "pollution_type": "stale", "text": "Correct answer: asthma"}],
    }
    preds = optimizer.run_baselines([case], None, tmp_path, max_workers=2, baseline_set="required")
    methods = {pred["method"] for pred in preds}
    assert methods == {
        "direct_deepseek",
        "baseline_single_cot_agent",
        "baseline_amem_adapter",
        "baseline_ddo_adapter",
        "baseline_colacare_adapter",
    }


def test_fast_formal_ablation_group_parser_selects_three_groups():
    from mem_ehr_agent.optimizer import parse_ablation_groups

    groups = parse_ablation_groups(
        "full,no_memory_cleaning,no_evidence_note_injection",
        default=[],
    )
    assert [name for name, _ in groups] == [
        "full",
        "ablate_no_memory_cleaning",
        "ablate_no_evidence_note_injection",
    ]


def test_compact_case_context_limits_long_timelines():
    case = {
        "case_id": "case_x",
        "demographics": {},
        "events": [
            {"event_id": f"ev_{idx}", "time": idx, "type": "clinical", "text": f"routine note {idx}"}
            for idx in range(60)
        ]
        + [{"event_id": "ev_dx", "time": 61, "type": "diagnosis", "text": "biopsy confirmed lymphoma"}],
        "synthetic_labs": [],
    }
    context = compact_case_context(case, max_events=10)
    assert "biopsy confirmed lymphoma" in context
    assert "shown_events=" in context
    assert "routine note 30" not in context


def test_visible_diagnosis_candidate_can_refine_primary_without_gold():
    case = {
        "labels": {"primary_diagnosis": "hidden"},
        "events": [
            {"event_id": "ev_1", "time": -1, "type": "diagnosis", "text": "diagnosed with pneumonia"},
            {"event_id": "ev_2", "time": 0, "type": "diagnosis", "text": "no history of chronic pulmonary disease"},
        ],
    }
    pred = {
        "primary_diagnosis": "bronchioloalveolar carcinoma",
        "diagnosis_list": ["bronchioloalveolar carcinoma", "Pneumonia"],
        "evidence": ["diagnosed with pneumonia"],
        "reasoning_summary": "",
    }
    updated = prefer_visible_diagnosis_candidate(case, pred)
    assert updated["primary_diagnosis"] == "pneumonia"
    assert updated["primary_selection"]["event_id"] == "ev_1"
    assert "hidden" not in str(updated)
