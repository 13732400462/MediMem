import json

from mem_ehr_agent.llm import LLMResult
from mem_ehr_agent.memory import MemoryStore, apply_critique, batched_critic_prompt, bootstrap_memory, llm_critic_ops, memory_card_candidate, parse_critic_operations
from mem_ehr_agent.agents import (
    case_context,
    compact_case_context,
    diagnosis_event_candidates,
    feature_state,
    prefer_visible_diagnosis_candidate,
    pollution_memory_context,
    resolve_top_k,
    run_llm_prediction,
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


def test_strict_prediction_parse_failure_falls_back_with_error_marker():
    class TruncatedJsonClient:
        def __init__(self):
            self.config = type("Config", (), {"max_tokens": 128})()
            self.remaining_completion_tokens = 96

        def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
            return LLMResult(
                text='{"primary_diagnosis": "chronic',
                usage={"prompt_tokens": 7, "completion_tokens": 32, "total_tokens": 39, "calls": 1, "latency_ms": 1},
                latency_s=0.001,
            )

    case = {
        "case_id": "case_parse_fallback",
        "source": "PMOA-TTS",
        "events": [{"event_id": "ev_1", "time": 1, "type": "diagnosis", "text": "Confirmed pneumonia."}],
        "labels": {"primary_diagnosis": "pneumonia", "diagnosis_list": ["pneumonia"]},
    }
    pred = run_llm_prediction(
        case,
        method="strict_parse_fallback_test",
        client=TruncatedJsonClient(),
        context=case_context(case),
        fail_on_llm_error=True,
    )
    assert pred["llm_error_type"] == "malformed_json"
    assert "LLM parse fallback reason" in pred["reasoning_summary"]
    assert pred["usage"]["completion_tokens"] == 32


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
        "polluted_memory": False,
        "critic_op_guard": True,
        "evidence_note_injection": True,
        "profile_adaptive_memory_cleaning": True,
        "profile_adaptive_evidence_notes": False,
        "temporal_signal": True,
        "counterfactual_verification": True,
    }


def test_enable_polluted_memory_feature_state():
    strategy = {"features": {"enable_polluted_memory": True}}
    assert feature_state(strategy)["polluted_memory"] is True


def test_bootstrap_memory_omits_poison_by_default(tmp_path):
    case = {
        "case_id": "case_clean_memory",
        "events": [{"event_id": "ev1", "text": "Confirmed pneumonia.", "type": "diagnosis", "time": 1}],
        "memory_seed": [],
        "poison_records": [{"poison_id": "p1", "text": "Stale incorrect diagnosis.", "pollution_type": "stale"}],
    }
    clean = bootstrap_memory(case, tmp_path / "clean.memory.jsonl")
    polluted = bootstrap_memory(case, tmp_path / "polluted.memory.jsonl", include_poison=True)
    assert all("poison" not in card.get("tags", []) for card in clean.cards)
    assert any("poison" in card.get("tags", []) for card in polluted.cards)


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


def test_apply_critique_without_client_keeps_hidden_poison_metadata(tmp_path):
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
    assert ops[0]["op"] == "Keep"
    assert ops[0]["revised_memory_id"] is None
    assert ops[0]["preserved_facts"] == []


def test_apply_critique_prompt_hides_poison_metadata_and_allows_revision(tmp_path):
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
                "pollution_type": "outdated_initial_diagnosis",
                "staleness_type": "superseded_by_later_diagnosis",
                "expected_op": "Revise",
                "revised_claim": "hidden gold revise",
                "preserved_facts": ["hidden fever fact"],
                "claim_type": "interpretation",
                "source_event_id": "ev_000",
                "supporting_evidence": ["fever was present"],
            }
        ],
    }
    store = MemoryStore.load("case_x", tmp_path / "memory.jsonl")
    card = store.write_card(
        summary="Early working impression: viral syndrome was plausible because fever was present.",
        evidence_refs=["ev_000"],
        time_scope={"start": 0, "end": 0},
        confidence=0.35,
        tags=["initial"],
    )

    class CleanReviseClient:
        def __init__(self):
            self.prompt = ""

        def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
            self.prompt = "\n".join(message["content"] for message in messages)
            forbidden = ("poison_x", "pollution_type", "staleness_type", "expected_op", "hidden gold revise")
            assert not any(item in self.prompt for item in forbidden)
            return LLMResult(
                text=(
                    '{"operations":[{"op":"Revise","target":"'
                    + card["memory_id"]
                    + '","reason":"Later visible evidence changes interpretation.",'
                    '"revised_claim":"Early impression is time-limited.",'
                    '"preserved_facts":["fever was present"]}]}'
                ),
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                latency_s=0.0,
            )

    client = CleanReviseClient()
    ops = apply_critique(case, store, client)
    assert ops[0]["op"] == "Revise"
    assert ops[0]["target"] == card["memory_id"]
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
    card = store.write_card(
        summary="Similar case memory: another patient had COVID ARDS; transfer this prior case context.",
        evidence_refs=["poison_x"],
        time_scope={},
        confidence=0.2,
        tags=["poison"],
    )

    class CleanDiscardClient:
        def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
            prompt = "\n".join(message["content"] for message in messages)
            assert "similar_case_mistransfer" not in prompt
            assert "poison_x" not in prompt
            return LLMResult(
                text=(
                    '{"operations":[{"op":"Discard","target":"'
                    + card["memory_id"]
                    + '","reason":"Not supported by visible timeline.",'
                    '"revised_claim":"","preserved_facts":[]}]}'
                ),
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                latency_s=0.0,
            )

    ops = apply_critique(case, store, CleanDiscardClient())
    assert ops[0]["op"] == "Discard"


def test_apply_critique_reviews_clean_memory_without_poison_records(tmp_path):
    case = {
        "case_id": "case_clean",
        "events": [{"event_id": "ev_001", "type": "diagnosis", "text": "Confirmed pneumonia."}],
        "poison_records": [],
    }
    store = MemoryStore.load("case_clean", tmp_path / "memory.jsonl")
    card = store.write_card(
        summary="Confirmed pneumonia on chest imaging.",
        evidence_refs=["ev_001"],
        time_scope={"start": 1, "end": 1},
        confidence=0.72,
        tags=["diagnosis"],
    )
    ops = apply_critique(case, store)
    assert ops == [
        {
            "op": "Keep",
            "raw_op": None,
            "guarded_op": None,
            "guard_applied": False,
            "target": card["memory_id"],
            "touched_memory_ids": [card["memory_id"]],
            "revised_memory_id": None,
            "preserved_facts": [],
            "revised_claim": "",
            "reason": "Memory candidate reviewed against visible timeline evidence.",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    ]


def test_clean_memory_card_can_be_revised_by_critic_without_pollution_guard(tmp_path):
    case = {
        "case_id": "case_clean_revise",
        "events": [
            {"event_id": "ev_001", "type": "diagnosis", "text": "Early diagnosis considered viral syndrome."},
            {"event_id": "ev_002", "type": "diagnosis", "text": "Later PCR confirmed bacterial pneumonia."},
        ],
        "poison_records": [],
    }
    store = MemoryStore.load("case_clean_revise", tmp_path / "memory.jsonl")
    card = store.write_card(
        summary="Viral syndrome is the current diagnosis.",
        evidence_refs=["ev_001"],
        time_scope={"start": 1, "end": 1},
        confidence=0.62,
        tags=["diagnosis"],
    )

    class CleanReviseClient:
        def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
            return LLMResult(
                text=(
                    '{"operations":[{"op":"Revise","target":"'
                    + card["memory_id"]
                    + '","reason":"Later visible evidence changes the temporal interpretation.",'
                    '"revised_claim":"Treat viral syndrome as an early time-limited impression.",'
                    '"preserved_facts":["early viral syndrome was considered"]}]}'
                ),
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                latency_s=0.0,
            )

    ops = apply_critique(case, store, CleanReviseClient())
    assert ops[0]["op"] == "Revise"
    assert ops[0]["guarded_op"] == ""
    assert ops[0]["guard_applied"] is False
    assert ops[0]["revised_memory_id"]
    assert any(memory["status"] == "superseded" for memory in store.cards if memory["memory_id"] == card["memory_id"])


def test_clean_memory_without_risk_type_does_not_trigger_pollution_guard(tmp_path):
    case = {
        "case_id": "case_clean_no_guard",
        "events": [{"event_id": "ev_001", "type": "clinical", "text": "Visible clinical note."}],
        "poison_records": [],
    }
    store = MemoryStore.load("case_clean_no_guard", tmp_path / "memory.jsonl")
    card = store.write_card(
        summary="Unsupported imported memory from another patient.",
        evidence_refs=["ev_001"],
        time_scope={},
        confidence=0.4,
        tags=["diagnosis"],
    )

    class CleanDiscardClient:
        def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
            return LLMResult(
                text=(
                    '{"operations":[{"op":"Discard","target":"'
                    + card["memory_id"]
                    + '","reason":"Not supported by visible case evidence.",'
                    '"revised_claim":"","preserved_facts":[]}]}'
                ),
                usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                latency_s=0.0,
            )

    ops = apply_critique(case, store, CleanDiscardClient())
    assert ops[0]["op"] == "Discard"
    assert ops[0]["guarded_op"] == ""
    assert ops[0]["guard_applied"] is False


def test_llm_critic_ops_splits_many_clean_cards_into_budgeted_batches():
    case = {
        "case_id": "case_many_cards",
        "events": [
            {
                "event_id": f"ev_{idx:03d}",
                "time": idx,
                "type": "diagnosis" if idx % 3 == 0 else "clinical",
                "text": "visible timeline evidence " + ("diagnosis confirmed. " * 12),
            }
            for idx in range(35)
        ],
        "poison_records": [],
    }
    candidates = [
        {
            "candidate_id": f"mem_{idx}",
            "candidate_kind": "memory_card",
            "text": "clean memory card " + str(idx) + " " + ("long details " * 120),
            "supporting_evidence": [f"ev_{idx % 10:03d}", "extra evidence " * 30],
            "tags": ["diagnosis"] * 12,
            "status": "active",
        }
        for idx in range(14)
    ]

    class RecordingClient:
        def __init__(self):
            self.calls = []

        def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
            prompt = "\n".join(message["content"] for message in messages)
            assert len(prompt) < 20000
            self.calls.append((messages, max_tokens))
            payload = []
            for candidate in candidates:
                if f'"id": "{candidate["candidate_id"]}"' in prompt:
                    payload.append(
                        {
                            "op": "Keep",
                            "target": candidate["candidate_id"],
                            "reason": "Visible evidence supports this memory.",
                            "revised_claim": "",
                            "preserved_facts": [],
                        }
                    )
            return LLMResult(
                text='{"operations":' + json.dumps(payload) + "}",
                usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                latency_s=0.0,
            )

    client = RecordingClient()
    ops = llm_critic_ops(case, candidates, client, fail_on_llm_error=True)
    assert len(ops) == len(candidates)
    assert all(op["op"] == "Keep" for op in ops)
    assert len(client.calls) > 1
    assert all(max_tokens <= 512 for _, max_tokens in client.calls)


def test_single_oversized_clean_card_is_truncated_and_still_reviewed():
    case = {
        "case_id": "case_long_card",
        "events": [
            {
                "event_id": "ev_long",
                "time": 1,
                "type": "diagnosis",
                "text": "confirmed diagnosis " + ("visible detail " * 1000),
            }
        ],
        "poison_records": [],
    }
    candidate = {
        "candidate_id": "mem_long",
        "candidate_kind": "memory_card",
        "text": "large clean memory " + ("x" * 50000),
        "supporting_evidence": ["ev_long", "support " * 3000],
        "tags": ["diagnosis"],
        "status": "active",
    }

    class RejectFirstClient:
        def __init__(self):
            self.calls = []

        def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
            prompt = "\n".join(message["content"] for message in messages)
            self.calls.append(prompt)
            if len(self.calls) == 1:
                raise RuntimeError("context length")
            assert len(prompt) < 8000
            return LLMResult(
                text='{"operations":[{"op":"Keep","target":"mem_long","reason":"ok","revised_claim":"","preserved_facts":[]}]}',
                usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
                latency_s=0.0,
            )

    client = RejectFirstClient()
    ops = llm_critic_ops(case, [candidate], client, fail_on_llm_error=True)
    assert ops[0]["op"] == "Keep"
    assert len(client.calls) == 2
    assert "x" * 1000 not in client.calls[-1]


def test_truncated_critic_output_uses_valid_single_target():
    ops = parse_critic_operations(
        '{"op": "Keep", "target": "mem_fcd37cd188a444444444444444444444444444444444444444444444444',
        valid_targets=["mem_fcd37cd188a"],
    )
    assert ops == [
        {
            "op": "Keep",
            "target": "mem_fcd37cd188a",
            "reason": "Parsed from truncated critic output.",
            "revised_claim": "",
            "preserved_facts": [],
        }
    ]


def test_medimem_prediction_prompt_budgets_long_case_and_extra():
    case = {
        "case_id": "pmoa_tts_0686_like",
        "demographics": {},
        "events": [
            {
                "event_id": f"ev_{idx:03d}",
                "time": idx,
                "type": "diagnosis" if idx % 9 == 0 else "clinical",
                "text": "visible longitudinal event " + ("detail " * 220),
            }
            for idx in range(90)
        ],
        "synthetic_labs": [],
        "poison_records": [],
        "counterfactuals": [],
    }

    class BudgetRecordingClient:
        def __init__(self):
            self.prompts = []

        def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
            prompt = "\n".join(message["content"] for message in messages)
            self.prompts.append((prompt, max_tokens))
            assert len(prompt) < 21000
            assert "extra card " * 900 not in prompt
            return LLMResult(
                text='{"primary_diagnosis":"pneumonia","confidence":0.7}',
                usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                latency_s=0.0,
            )

    client = BudgetRecordingClient()
    pred = run_llm_prediction(
        case,
        method="medimem_topk8_round1",
        client=client,
        context=case_context(case, include_labs=True),
        extra="[MEMORY_CARDS]\n" + ("extra card " * 9000),
        fail_on_llm_error=True,
    )

    assert pred["primary_diagnosis"] == "pneumonia"
    assert client.prompts
    assert client.prompts[0][1] <= 128


def test_strict_prediction_retries_compacted_prompt_without_fallback():
    case = {
        "case_id": "case_retry",
        "events": [{"event_id": "ev_1", "time": 0, "type": "clinical", "text": "fever " * 5000}],
        "synthetic_labs": [],
    }

    class RejectThenAcceptClient:
        def __init__(self):
            self.calls = []

        def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
            prompt = "\n".join(message["content"] for message in messages)
            self.calls.append(prompt)
            if len(self.calls) == 1:
                raise RuntimeError("maximum context length")
            assert len(prompt) < len(self.calls[0])
            return LLMResult(
                text='{"primary_diagnosis":"viral syndrome","confidence":0.4}',
                usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                latency_s=0.0,
            )

    client = RejectThenAcceptClient()
    pred = run_llm_prediction(
        case,
        method="medimem_topk8_round1",
        client=client,
        context=case_context(case),
        extra="extra " * 8000,
        fail_on_llm_error=True,
    )

    assert pred["primary_diagnosis"] == "viral syndrome"
    assert "llm_error" not in pred
    assert len(client.calls) == 2


def test_safe_prompt_ops_compaction_keeps_critic_fields_hidden():
    ops = [
        {
            "op": "Revise",
            "target": "poison_x",
            "reason": "Later evidence says lymphoma.",
            "revised_claim": "This is lymphoma.",
            "preserved_facts": ["fever", "biopsy"],
            "touched_memory_ids": ["m1", "m2"],
            "revised_memory_id": "m3",
        }
    ]
    prompt_ops = safe_memory_ops_for_prompt({"case_id": "case_x"}, ops)
    assert "reason" not in str(prompt_ops)
    assert "revised_claim" not in str(prompt_ops)
    assert "lymphoma" not in str(prompt_ops).lower()


class FakeCriticClient:
    def __init__(self, target="poison_x"):
        self.target = target

    def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
        return LLMResult(
            text=(
                '{"operations":[{"op":"Revise","target":"'
                + self.target
                + '","reason":"Later evidence confirms lymphoma.",'
                '"revised_claim":"Fever supports lymphoma rather than viral syndrome.",'
                '"preserved_facts":["fever was present","lymphoma is likely"]}]}'
            ),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_s=0.0,
        )


class WrongOpCriticClient:
    def __init__(self, target="poison_x"):
        self.target = target

    def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
        return LLMResult(
            text=(
                '{"operations":[{"op":"Discard","target":"'
                + self.target
                + '","reason":"Overly aggressive deletion.",'
                '"revised_claim":"","preserved_facts":[]}]}'
            ),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_s=0.0,
        )


def test_hidden_poison_metadata_does_not_force_partial_truth_to_revise(tmp_path):
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
    card = store.write_card(
        summary="Early interpretation was viral syndrome because fever was present.",
        evidence_refs=["poison_x"],
        time_scope={},
        confidence=0.2,
        tags=["poison"],
    )
    ops = apply_critique(case, store, WrongOpCriticClient(card["memory_id"]))
    assert ops[0]["op"] == "Discard"
    assert ops[0]["guarded_op"] == ""
    assert ops[0]["preserved_facts"] == []


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
    card = store.write_card(
        summary="Early interpretation was viral syndrome because fever was present.",
        evidence_refs=["poison_x"],
        time_scope={},
        confidence=0.2,
        tags=["poison"],
    )
    ops = apply_critique(case, store, WrongOpCriticClient(card["memory_id"]), enforce_op_guard=False)
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
    card = store.write_card(
        summary="Early working impression: viral syndrome was plausible because fever was present.",
        evidence_refs=["poison_x"],
        time_scope={},
        confidence=0.2,
        tags=["poison"],
    )
    ops = apply_critique(case, store, FakeCriticClient(card["memory_id"]))
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
            "baseline_static_rag",
            "baseline_amem_adapter",
        "baseline_ddo_adapter",
        "baseline_colacare_adapter",
    }


def test_fast_formal_ablation_group_parser_selects_pollution_ablation():
    from mem_ehr_agent.optimizer import parse_ablation_groups

    groups = parse_ablation_groups(
        "full,no_memory_cleaning,no_evidence_note_injection,with_polluted_memory",
        default=[],
    )
    assert [name for name, _ in groups] == [
        "full",
        "ablate_no_memory_cleaning",
        "ablate_no_evidence_note_injection",
        "ablate_with_polluted_memory",
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
