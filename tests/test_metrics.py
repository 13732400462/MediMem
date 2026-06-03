from mem_ehr_agent.metrics import (
    build_leakage_audit,
    build_expected_memory_op_distribution,
    build_memory_op_confusion,
    add_merged_ours_summaries,
    build_slice_breakdown,
    build_metric_gate,
    diagnosis_match,
    evaluate_predictions,
    action_quality,
    fact_preservation_rate,
    fact_preservation_soft,
    list_f1,
    memory_pollution_control_score,
    over_deletion_rate,
    revision_quality,
    revision_accuracy,
    stale_memory_action_accuracy,
    stale_memory_challenge_score,
    token_f1,
)
from mem_ehr_agent.medical_terms import canonicalize_diagnosis


def test_token_f1_overlap():
    assert token_f1("acute myeloid leukemia", "myeloid leukemia") > 0.7


def test_diagnosis_match_substring():
    assert diagnosis_match("non small cell lung cancer", "Non-small-cell lung cancer (NSCLC)")


def test_list_f1():
    score = list_f1(["acute myeloid leukemia", "anemia"], ["AML", "acute myeloid leukemia"])
    assert score > 0.4


def test_medical_aliases_match_current_error_patterns():
    assert canonicalize_diagnosis("CAH in newborn screening test") == "congenital adrenal hyperplasia"
    assert diagnosis_match("SLE", "Systemic lupus erythematosus")
    assert diagnosis_match("NSTEMI", "acute non-STEMI")
    assert diagnosis_match("Phosphaturic mesenchymal tumor, mixed type", "Tumor-induced osteomalacia")
    assert diagnosis_match("Non-mucinous BAC with EGFR exon 19 deletion", "Bronchioloalveolar carcinoma (BAC)")
    assert diagnosis_match("Hydatid disease (Echinococcus granulosus)", "Echinococcosis")
    assert diagnosis_match("Inferior vena cava aneurysm with renal vein thrombosis", "IVC aneurysm")
    assert diagnosis_match("acute non-ST-segment elevation myocardial infarction", "NSTEMI")
    assert diagnosis_match("K-wire migration into urinary bladder", "K-wire migration")
    assert diagnosis_match("Azygos vein anomaly", "Azygos lobe")


def test_metric_gate_uses_strictest_amem_and_dual_margin():
    rows = [
        {
            "method": "baseline_amem_adapter",
            "primary_diagnosis_top1_accuracy": "0.50",
            "diagnosis_list_f1": "0.48",
            "cdr_f1": "0.48",
            "counterfactual_robustness_proxy": "0.24",
        },
        {
            "method": "baseline_polluted_amem_adapter",
            "primary_diagnosis_top1_accuracy": "0.52",
            "diagnosis_list_f1": "0.46",
            "cdr_f1": "0.46",
            "counterfactual_robustness_proxy": "0.23",
        },
        {
            "method": "full_ours_topk8_round1",
            "primary_diagnosis_top1_accuracy": "0.62",
            "diagnosis_list_f1": "0.59",
            "cdr_f1": "0.59",
            "counterfactual_robustness_proxy": "0.35",
            "stale_memory_action_accuracy": "1.0",
            "revision_accuracy": "1.0",
            "over_deletion_rate": "0.0",
        },
    ]

    gate = build_metric_gate(rows)

    assert gate["passed"]
    diag_check = next(check for check in gate["checks"] if check["metric"] == "diagnosis_list_f1")
    assert diag_check["reference"] == 0.48
    assert diag_check["target"] == 0.58


def test_metric_gate_defaults_to_full_ours_merged():
    rows = [
        {
            "method": "baseline_amem_adapter",
            "primary_diagnosis_top1_accuracy": "0.50",
            "diagnosis_list_f1": "0.50",
            "cdr_f1": "0.50",
            "counterfactual_robustness_proxy": "0.20",
        },
        {
            "method": "full_ours_topk3_round1",
            "primary_diagnosis_top1_accuracy": "1.00",
            "diagnosis_list_f1": "1.00",
            "cdr_f1": "1.00",
            "counterfactual_robustness_proxy": "0.40",
            "stale_memory_action_accuracy": "1.0",
            "revision_accuracy": "1.0",
            "over_deletion_rate": "0.0",
        },
        {
            "method": "full_ours_merged",
            "primary_diagnosis_top1_accuracy": "0.55",
            "diagnosis_list_f1": "0.55",
            "cdr_f1": "0.55",
            "counterfactual_robustness_proxy": "0.25",
            "stale_memory_action_accuracy": "1.0",
            "revision_accuracy": "1.0",
            "over_deletion_rate": "0.0",
        },
    ]

    gate = build_metric_gate(rows)

    assert gate["method"] == "full_ours_merged"
    assert not gate["passed"]


def test_cdr_and_counterfactual_use_recalled_diagnosis_list_items():
    case = {
        "case_id": "case_list_recall",
        "labels": {"primary_diagnosis": "Rosai-Dorfman-Destombes disease", "diagnosis_list": ["Rosai-Dorfman-Destombes disease"]},
        "qa_tasks": [{"type": "CDR", "answer": "Rosai-Dorfman-Destombes disease"}],
        "counterfactuals": [{"remove": "diagnosis evidence"}],
    }
    pred = {
        "case_id": "case_list_recall",
        "method": "full_ours_topk8_round1",
        "primary_diagnosis": "renal mass",
        "diagnosis_list": ["rosai dorfman destombes disease", "kidney calculi"],
        "evidence": ["visible diagnosis event supports rosai dorfman destombes disease"],
        "counterfactual_verification": {
            "enabled": True,
            "cpg": 0.40,
            "threshold": 0.40,
            "passed": True,
        },
    }

    summary = evaluate_predictions([case], [pred])["summary"][0]

    assert summary["cdr_f1"] == 1.0
    assert summary["counterfactual_probability_gap"] == 0.40
    assert summary["counterfactual_pass_rate"] == 1.0
    assert summary["counterfactual_robustness_score"] == 1.0
    assert summary["counterfactual_robustness_proxy"] == 1.0


def test_stale_memory_metrics_reward_revision_and_fact_preservation():
    case = {
        "expected_memory_ops": [
            {
                "op": "Revise",
                "target": "poison_1",
                "preserved_facts": ["fever was present"],
                "should_preserve_fact": True,
            }
        ]
    }
    pred = {
        "memory_ops": [
            {
                "op": "Revise",
                "target": "poison_1",
                "revised_memory_id": "mem_new",
                "revised_claim": "early interpretation superseded",
                "preserved_facts": ["fever was present"],
            }
        ]
    }
    assert stale_memory_action_accuracy(case, pred) == 1.0
    assert revision_accuracy(case, pred) == 1.0
    assert fact_preservation_rate(case, pred) == 1.0
    assert action_quality(case, pred) == 1.0
    assert revision_quality(case, pred) == 1.0
    assert fact_preservation_soft(case, pred) == 1.0
    assert memory_pollution_control_score(case, pred) > 0.999
    assert over_deletion_rate(case, pred) == 0.0
    assert stale_memory_challenge_score(case, pred) > 0.999


def test_over_deletion_penalizes_discard_when_revision_expected():
    case = {
        "expected_memory_ops": [
            {
                "op": "Revise",
                "target": "poison_1",
                "preserved_facts": ["fever was present"],
                "should_preserve_fact": True,
            }
        ]
    }
    pred = {"memory_ops": [{"op": "Discard", "target": "poison_1", "preserved_facts": []}]}
    assert stale_memory_action_accuracy(case, pred) == 0.0
    assert revision_accuracy(case, pred) == 0.0
    assert fact_preservation_rate(case, pred) == 0.0
    assert over_deletion_rate(case, pred) == 1.0
    assert action_quality(case, pred) == 0.0
    assert revision_quality(case, pred) == 0.0
    assert fact_preservation_soft(case, pred) == 0.0
    assert memory_pollution_control_score(case, pred) == 0.0


def test_soft_pollution_metrics_give_partial_credit():
    case = {
        "expected_memory_ops": [
            {
                "op": "Invalidate",
                "target": "poison_1",
                "preserved_facts": ["fever was present"],
                "should_preserve_fact": True,
            }
        ]
    }
    pred = {
        "memory_ops": [
            {
                "op": "Revise",
                "target": "poison_1",
                "revised_memory_id": "mem_new",
                "revision_note": "preserve fact but narrow scope",
                "preserved_facts": ["fever present"],
            }
        ]
    }
    assert action_quality(case, pred) == 0.55
    assert fact_preservation_soft(case, pred) > 0.7
    assert 0.0 < memory_pollution_control_score(case, pred) < 1.0


def test_unexposed_baseline_is_not_scored_as_pollution_failure():
    case = {
        "case_id": "case_1",
        "labels": {"primary_diagnosis": "pneumonia", "diagnosis_list": ["pneumonia"]},
        "qa_tasks": [{"type": "CDR", "answer": "pneumonia"}],
        "expected_memory_ops": [{"op": "Discard", "target": "poison_1"}],
        "counterfactuals": [],
    }
    clean_baseline = {
        "case_id": "case_1",
        "method": "baseline_amem_adapter",
        "primary_diagnosis": "pneumonia",
        "diagnosis_list": ["pneumonia"],
        "confidence": 0.8,
        "evidence": [],
    }
    polluted_baseline = clean_baseline | {"method": "baseline_polluted_amem_adapter", "pollution_exposed": True}
    result = evaluate_predictions([case], [clean_baseline, polluted_baseline])
    by_case_row = {row["method"]: row for row in result["case_rows"]}
    assert by_case_row["baseline_amem_adapter"]["stale_memory_action_accuracy"] is None
    assert by_case_row["baseline_polluted_amem_adapter"]["stale_memory_action_accuracy"] is None
    assert by_case_row["baseline_polluted_amem_adapter"]["memory_pollution_control_score"] is None
    by_summary = {row["method"]: row for row in result["summary"]}
    assert by_summary["baseline_polluted_amem_adapter"]["memory_pollution_control_score"] is None


def test_ours_without_memory_ops_is_scored_as_pollution_failure():
    case = {
        "case_id": "case_1",
        "labels": {"primary_diagnosis": "pneumonia", "diagnosis_list": ["pneumonia"]},
        "qa_tasks": [{"type": "CDR", "answer": "pneumonia"}],
        "expected_memory_ops": [{"op": "Discard", "target": "poison_1"}],
        "counterfactuals": [],
    }
    pred = {
        "case_id": "case_1",
        "method": "ablate_no_memory_cleaning_ours_topk8_round1",
        "primary_diagnosis": "pneumonia",
        "diagnosis_list": ["pneumonia"],
        "evidence": [],
    }
    result = evaluate_predictions([case], [pred])
    row = result["case_rows"][0]
    assert row["stale_memory_action_accuracy"] == 0.0
    assert row["memory_pollution_control_score"] == 0.0


def test_keep_expected_op_rewards_non_intervention():
    case = {
        "expected_memory_ops": [
            {"op": "Keep", "target": "poison_1", "preserved_facts": ["fever"], "should_preserve_fact": True}
        ]
    }
    pred = {"memory_ops": []}
    assert stale_memory_action_accuracy(case, pred) == 1.0
    assert action_quality(case, pred) == 1.0
    assert memory_pollution_control_score(case, pred) > 0.0


def test_expected_op_distribution_groups_by_pollution_type():
    cases = [
        {
            "case_id": "case_1",
            "poison_records": [
                {"poison_id": "poison_keep", "pollution_type": "valid_historical_fact"},
                {"poison_id": "poison_flag", "pollution_type": "ambiguous_low_confidence_memory"},
            ],
            "expected_memory_ops": [
                {"op": "Keep", "target": "poison_keep"},
                {"op": "Flag", "target": "poison_flag"},
            ],
        }
    ]
    assert build_expected_memory_op_distribution(cases) == {
        "ambiguous_low_confidence_memory": {"Flag": 1},
        "valid_historical_fact": {"Keep": 1},
    }


def test_leakage_audit_and_memory_confusion_are_offline_only():
    case = {
        "case_id": "case_1",
        "labels": {"primary_diagnosis": "pneumonia", "diagnosis_list": ["pneumonia"]},
        "memory_seed": [{"summary": "fever and cough"}],
        "poison_records": [{"poison_id": "poison_1", "text": "old viral interpretation"}],
        "expected_memory_ops": [{"op": "Revise", "target": "poison_1"}],
        "source_real": True,
    }
    pred = {
        "case_id": "case_1",
        "method": "ours_topk3_round1",
        "primary_diagnosis": "viral syndrome",
        "prompt_memory_ops": [{"op": "Revise", "target": "poison_1", "preserved_facts": ["fever"]}],
        "memory_ops": [{"op": "Revise", "target": "poison_1"}],
    }
    assert build_leakage_audit([case], [pred]) == {
        "runtime_gold_mentions": 0,
        "prompt_memory_ops_gold_mentions": 0,
        "counterfactual_runtime_gold_mentions": 0,
        "poison_expected_op": 0,
        "poison_revised_claim": 0,
        "poison_expected_memory_ops": 0,
        "source_real_false": 0,
    }
    assert build_memory_op_confusion([case], [pred]) == {"Revise->Revise": 1}


def test_full_ours_merged_aggregates_dynamic_topk_rows():
    cases = [
        {
            "case_id": "case_0001",
            "labels": {"primary_diagnosis": "pneumonia", "diagnosis_list": ["pneumonia"]},
            "qa_tasks": [{"type": "CDR", "answer": "pneumonia"}],
            "expected_memory_ops": [],
            "counterfactuals": [],
            "data_quality_flags": {"event_count": 10, "diagnosis_event_count": 1, "has_follow_up": True},
            "poison_records": [],
        },
        {
            "case_id": "case_0002",
            "labels": {"primary_diagnosis": "asthma", "diagnosis_list": ["asthma"]},
            "qa_tasks": [{"type": "CDR", "answer": "asthma"}],
            "expected_memory_ops": [],
            "counterfactuals": [],
            "data_quality_flags": {"event_count": 80, "diagnosis_event_count": 3, "has_follow_up": False},
            "poison_records": [],
        },
    ]
    preds = [
        {
            "case_id": "case_0001",
            "method": "full_ours_topk3_round1",
            "primary_diagnosis": "pneumonia",
            "diagnosis_list": ["pneumonia"],
            "evidence": [],
            "memory_ops": [],
        },
        {
            "case_id": "case_0002",
            "method": "full_ours_topk8_round1",
            "primary_diagnosis": "bronchitis",
            "diagnosis_list": ["bronchitis"],
            "evidence": [],
            "memory_ops": [],
        },
    ]
    result = add_merged_ours_summaries(evaluate_predictions(cases, preds))
    summary = {row["method"]: row for row in result["summary"]}
    assert summary["full_ours_merged"]["n"] == 2
    assert summary["full_ours_merged"]["primary_diagnosis_top1_accuracy"] == 0.5
    slice_rows = build_slice_breakdown(cases, result["case_rows"])
    assert any(row["slice"] == "all" and row["method"] == "full_ours_merged" for row in slice_rows)
