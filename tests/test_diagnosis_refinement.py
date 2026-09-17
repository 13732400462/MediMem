from medimem.agents import (
    answer_entity_source_candidates,
    evidence_driven_diagnosis_rerank,
    evidence_gated_diagnosis_recall,
    refine_diagnosis_list,
    run_counterfactual_verification,
    run_ours,
    select_primary_for_task_profile,
)
from medimem.llm import LLMResult
from medimem.medical_terms import canonicalize_diagnosis


def test_refinement_keeps_primary_as_first_item():
    case = {
        "case_id": "case_0002",
        "events": [
            {"event_id": "ev_1", "type": "diagnosis", "time": 0, "text": "diagnosed with CAH in newborn screening test"}
        ],
    }
    pred = {
        "primary_diagnosis": "congenital adrenal hyperplasia",
        "diagnosis_list": ["hypogonadotropic hypogonadism", "severe oligozoospermia", "obesity"],
    }

    refined = refine_diagnosis_list(
        case,
        pred,
        evidence_notes=[],
        diagnosis_candidates=[{"time": 0, "event_id": "ev_1", "text": "diagnosed with CAH in newborn screening test"}],
    )

    assert refined["primary_diagnosis"] == pred["primary_diagnosis"]
    assert refined["diagnosis_list"][0] == "congenital adrenal hyperplasia"


def test_refinement_filters_non_diagnostic_phrases():
    pred = {
        "primary_diagnosis": "pneumonia",
        "diagnosis_list": [
            "chronic cough",
            "high-risk screening MRI",
            "frozen section confirmed GIST histology",
            "preliminary diagnosis of ischemic injury from suspected arterial occlusion",
        ],
    }

    refined = refine_diagnosis_list({}, pred, evidence_notes=[], diagnosis_candidates=[])

    assert "chronic cough" not in refined["diagnosis_list"]
    assert "high-risk screening MRI" not in refined["diagnosis_list"]
    assert "gastrointestinal stromal tumor" in refined["diagnosis_list"]
    assert "ischemic injury from suspected arterial occlusion" in refined["diagnosis_list"]


def test_requested_aliases_are_canonicalized():
    assert canonicalize_diagnosis("borderline resectable PDAC") == "pancreatic ductal adenocarcinoma"
    assert canonicalize_diagnosis("GIST histology") == "gastrointestinal stromal tumor"
    assert canonicalize_diagnosis("dMMR") == "mismatch repair deficiency"
    assert canonicalize_diagnosis("COVID-19") == "covid 19"
    assert canonicalize_diagnosis("papillary thyroid carcinoma") == "papillary thyroid cancer"
    assert canonicalize_diagnosis("Brain metastases") == "brain metastasis"
    assert canonicalize_diagnosis("Peritoneal carcinomatosis") == "peritoneal carcinomatosis"
    assert canonicalize_diagnosis("Necrobiotic xanthogranuloma") == "necrobiotic xanthogranuloma"
    assert canonicalize_diagnosis("VSD") == "ventricular septal defect"
    assert canonicalize_diagnosis("ASD") == "atrial septal defect"
    assert canonicalize_diagnosis("d-TGA") == "transposition of the great arteries"
    assert canonicalize_diagnosis("acute non-STEMI") == "acute non st elevation myocardial infarction"
    assert canonicalize_diagnosis("non-mucinous BAC") == "bronchioloalveolar carcinoma"
    assert canonicalize_diagnosis("azygos vein anomaly") == "azygos lobe"
    assert canonicalize_diagnosis("K-wire migration") == "k wire migration"
    assert canonicalize_diagnosis("sHLH") == "hemophagocytic lymphohistiocytosis"
    assert canonicalize_diagnosis("HCC hemorrhaging/rupture") == "hepatocellular carcinoma"
    assert canonicalize_diagnosis("lower extremity DVT") == "deep vein thrombosis"


def test_evidence_gated_recall_expands_list_without_changing_primary():
    case = {
        "case_id": "case_heart",
        "events": [
            {
                "event_id": "ev_1",
                "type": "diagnosis",
                "time": 1,
                "text": "Diagnosed with single-ventricle physiology, unbalanced atrioventricular canal, severe pulmonary stenosis, and abdominal situs ambiguous.",
            }
        ],
    }
    pred = {
        "primary_diagnosis": "single ventricle physiology",
        "diagnosis_list": ["single ventricle physiology"],
    }

    updated = evidence_gated_diagnosis_recall(
        case,
        pred,
        evidence_notes=[],
        diagnosis_candidates=[case["events"][0]],
    )

    assert updated["primary_diagnosis"] == "single ventricle physiology"
    assert "unbalanced atrioventricular canal" in updated["diagnosis_list"]
    assert "pulmonary stenosis" in updated["diagnosis_list"]
    assert updated["diagnosis_recall_added_count"] == 2


def test_evidence_driven_rerank_recalls_visible_entities_without_labels():
    case = {
        "case_id": "case_regression",
        "events": [
            {
                "event_id": "ev_1",
                "type": "diagnosis",
                "time": 1,
                "text": "Diagnosed with recurrent contralateral pneumothorax and azygos lobe anomaly.",
            },
            {
                "event_id": "ev_2",
                "type": "pathology",
                "time": 2,
                "text": "Surgical pathology confirmed non-mucinous BAC with EGFR exon 19 deletion.",
            },
            {
                "event_id": "ev_3",
                "type": "imaging",
                "time": 3,
                "text": "CT showed inferior vena cava aneurysm and venous malformations with gastritis.",
            },
        ],
    }
    pred = {
        "primary_diagnosis": "pneumothorax",
        "diagnosis_list": ["pneumothorax", "chronic cough", "CT scan"],
        "evidence": ["diagnosed with recurrent contralateral pneumothorax"],
    }

    updated = evidence_driven_diagnosis_rerank(
        case,
        pred,
        evidence_notes=[],
        diagnosis_candidates=[
            {"event_id": "ev_1", "time": 1, "text": case["events"][0]["text"]},
        ],
    )

    assert updated["primary_diagnosis"] == "pneumothorax"
    assert "recurrent contralateral pneumothorax" in updated["diagnosis_list"]
    assert "azygos lobe" in updated["diagnosis_list"]
    assert "bronchioloalveolar carcinoma" in updated["diagnosis_list"]
    assert "inferior vena cava aneurysm" in updated["diagnosis_list"]
    assert "venous malformation" in updated["diagnosis_list"]
    assert "chronic cough" not in updated["diagnosis_list"]


def test_medical_answer_entity_primary_can_be_non_disease_option():
    case = {
        "case_id": "medmcqa_demo",
        "task_profile": "medical_answer_entity",
        "answer_options": [
            {"label": "A", "text": "Mite"},
            {"label": "B", "text": "Fungus"},
            {"label": "C", "text": "Virus"},
        ],
        "events": [
            {"event_id": "ev_q", "time": 0, "type": "clinical", "text": "Scabies is caused by which organism?"},
            {"event_id": "ev_o", "time": 1, "type": "clinical", "text": "Answer options: A. Mite; B. Fungus; C. Virus"},
        ],
    }
    pred = {
        "primary_diagnosis": "parasitic infestation",
        "diagnosis_list": ["parasitic infestation", "Mite", "skin disease", "itching"],
        "evidence": ["The selected option is Mite."],
        "reasoning_summary": "Mite best matches scabies.",
    }

    updated = select_primary_for_task_profile(case, pred, evidence_notes=[], diagnosis_candidates=[])

    assert updated["primary_diagnosis"] == "Mite"
    assert len(updated["diagnosis_list"]) <= 3
    assert updated["diagnosis_granularity"] == "answer_entity"


def test_medical_instruction_primary_uses_visible_question_topic_when_uncertain():
    case = {
        "case_id": "wikidoc_demo",
        "task_profile": "medical_answer_entity",
        "data_quality_flags": {
            "source_type": "medical_instruction",
            "source_dataset": "medical_meadow_wikidoc",
            "no_leak_protocol": True,
        },
        "events": [
            {
                "event_id": "ev_title",
                "time": 0,
                "type": "clinical",
                "text": "Initial clinical task/source title: What information is available on secondary hyperparathyroidism?",
            },
            {
                "event_id": "ev_question",
                "time": 0,
                "type": "clinical",
                "text": "Clinical question or presentation: What information is available on secondary hyperparathyroidism?",
            },
        ],
    }
    pred = {
        "primary_diagnosis": "uncertain",
        "diagnosis_list": ["uncertain"],
        "evidence": ["The provided source is a clinical question."],
        "reasoning_summary": "The answer entity is uncertain from the sparse prompt.",
    }

    updated = select_primary_for_task_profile(case, pred, evidence_notes=[], diagnosis_candidates=[])

    assert updated["primary_diagnosis"] == "secondary hyperparathyroidism"
    assert updated["primary_selection_pass"]["source"] == "visible_instruction_question_topic"
    assert len(updated["diagnosis_list"]) <= 3


def test_longitudinal_primary_selector_keeps_main_diagnosis_over_complication():
    case = {
        "case_id": "pmoa_demo",
        "task_profile": "longitudinal_diagnosis",
        "events": [
            {"event_id": "ev_title", "time": 0, "type": "diagnosis", "text": "Initial clinical task/source title: Case report of non-small cell lung cancer"},
            {"event_id": "ev_comp", "time": 5, "type": "diagnosis", "text": "Later course complicated by pleural effusion and respiratory failure."},
        ],
    }
    pred = {
        "primary_diagnosis": "respiratory failure",
        "diagnosis_list": ["respiratory failure", "non-small cell lung cancer", "pleural effusion"],
        "evidence": ["case report of non-small cell lung cancer"],
        "reasoning_summary": "",
    }

    updated = select_primary_for_task_profile(
        case,
        pred,
        evidence_notes=[],
        diagnosis_candidates=[{"event_id": "ev_title", "time": 0, "text": case["events"][0]["text"]}],
    )

    assert updated["primary_diagnosis"] == "non small cell lung cancer"
    assert len(updated["diagnosis_list"]) <= 8


def test_pmoa_primary_selector_prefers_disease_entity_over_stage_state():
    case = {
        "case_id": "pmoa_stage_demo",
        "task_profile": "longitudinal_diagnosis",
        "data_quality_flags": {"source_dataset": "pmoa_tts", "source_type": "longitudinal_case"},
        "events": [
            {
                "event_id": "ev_path",
                "time": 2,
                "type": "pathology",
                "text": "Pathology positive for squamous cell carcinoma and small cell carcinoma.",
            },
            {"event_id": "ev_stage", "time": 3, "type": "diagnosis", "text": "Stage IV disease was documented."},
        ],
    }
    pred = {
        "primary_diagnosis": "stage IV disease",
        "diagnosis_list": ["stage IV disease", "pathology positive for squamous cell carcinoma", "COPD"],
        "evidence": ["pathology positive for squamous cell carcinoma"],
        "reasoning_summary": "Stage and pathology were both mentioned.",
    }

    updated = select_primary_for_task_profile(
        case,
        pred,
        evidence_notes=[],
        diagnosis_candidates=[{"event_id": "ev_stage", "time": 3, "text": "Stage IV disease was documented."}],
    )

    assert updated["primary_diagnosis"] == "squamous cell carcinoma"


class CounterfactualClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
        self.messages.append(messages)
        text = self.responses.pop(0)
        return LLMResult(
            text=text,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_s=0.0,
        )


def test_counterfactual_verification_computes_cpg_without_gold():
    case = {
        "case_id": "case_cf",
        "counterfactuals": [
            {
                "intervention": "Remove or negate this evidence: fever and lobar opacity",
                "target_diagnosis": "hidden gold pneumonia",
                "minimum_drop": 0.40,
            }
        ],
    }
    pred = {"primary_diagnosis": "pneumonia", "confidence": 0.90}
    client = CounterfactualClient(
        [
            '{"primary_diagnosis":"uncertain","confidence_for_original_diagnosis":0.30,'
            '"counterfactual_primary_diagnosis":"viral syndrome","causal_consistency_summary":"key evidence removed"}'
        ]
    )

    verification, usage = run_counterfactual_verification(
        case,
        pred,
        client,
        context="visible timeline only",
        extra="visible memory only",
    )

    assert round(verification["cpg"], 2) == 0.60
    assert verification["passed"]
    assert usage["total_tokens"] == 15
    prompt_text = str(client.messages)
    assert "hidden gold pneumonia" not in prompt_text
    assert "target_diagnosis" not in prompt_text


def test_counterfactual_verification_requires_real_client():
    case = {
        "case_id": "case_cf",
        "counterfactuals": [{"intervention": "Remove or negate this evidence: biopsy"}],
    }
    pred = {"primary_diagnosis": "lymphoma", "confidence": 0.80}

    try:
        run_counterfactual_verification(case, pred, None, context="visible", extra="")
    except RuntimeError as exc:
        assert "real LLM API client" in str(exc)
    else:
        raise AssertionError("counterfactual verification must not use offline fallback")


def test_run_ours_keeps_counterfactual_audit_only_without_high_confidence_contradiction(tmp_path):
    case = {
        "case_id": "case_cf_revision",
        "demographics": {},
        "events": [
            {"event_id": "ev_1", "time": 0, "type": "clinical", "text": "fever and cough"},
            {"event_id": "ev_2", "time": 1, "type": "imaging", "text": "chest opacity"},
        ],
        "synthetic_labs": [],
        "memory_seed": [],
        "poison_records": [],
        "counterfactuals": [{"intervention": "Remove or negate this evidence: chest opacity"}],
    }
    client = CounterfactualClient(
        [
            '{"primary_diagnosis":"pneumonia","diagnosis_list":["pneumonia"],"confidence":0.80,'
            '"evidence":["chest opacity"],"reasoning_summary":"opacity supports pneumonia"}',
            '{"primary_diagnosis":"pneumonia","confidence_for_original_diagnosis":0.70,'
            '"counterfactual_primary_diagnosis":"uncertain","causal_consistency_summary":"weak drop"}',
            '{"primary_diagnosis":"undifferentiated respiratory illness","diagnosis_list":["respiratory illness"],'
            '"confidence":0.45,"evidence":["fever and cough"],"reasoning_summary":"causal support remains weak"}',
        ]
    )

    pred = run_ours(
        case,
        client,
        memory_dir=tmp_path,
        strategy={
            "rounds": 1,
            "temperature": 0.0,
            "features": {
                "top_k_is_auto": True,
                "disable_memory_cleaning": True,
                "disable_evidence_note_injection": True,
            },
        },
    )

    assert pred["method"] == "medimem_topk3_round1"
    assert pred["counterfactual_revision_triggered"] is False
    assert "pre_counterfactual_prediction" not in pred
    assert pred["primary_diagnosis"] == "pneumonia"
    assert round(pred["counterfactual_verification"]["cpg"], 2) == 0.10
    assert pred["counterfactual_revision_policy"] == "audit_only_unless_high_confidence_contradiction"


def test_run_ours_risk_sample_skips_low_risk_counterfactual(tmp_path):
    case = {
        "case_id": "case_cf_skip",
        "demographics": {},
        "events": [
            {"event_id": "ev_1", "time": 0, "type": "clinical", "text": "fever and cough"},
            {"event_id": "ev_2", "time": 1, "type": "imaging", "text": "right lower lobe opacity"},
            {"event_id": "ev_3", "time": 2, "type": "diagnosis", "text": "diagnosed with pneumonia"},
        ],
        "synthetic_labs": [],
        "memory_seed": [],
        "poison_records": [],
        "counterfactuals": [{"intervention": "Remove or negate this evidence: right lower lobe opacity"}],
    }
    client = CounterfactualClient(
        [
            '{"primary_diagnosis":"pneumonia","diagnosis_list":["pneumonia"],"confidence":0.90,'
            '"evidence":["right lower lobe opacity","fever and cough"],"reasoning_summary":"evidence supports pneumonia"}',
            '{"primary_diagnosis":"pneumonia","diagnosis_list":["pneumonia"],"confidence":0.92,'
            '"evidence":["right lower lobe opacity","fever and cough"],"reasoning_summary":"stable diagnosis"}',
        ]
    )

    pred = run_ours(
        case,
        client,
        memory_dir=tmp_path,
        strategy={
            "rounds": 1,
            "temperature": 0.0,
            "features": {
                "top_k_is_auto": True,
                "disable_memory_cleaning": True,
                "counterfactual_policy": "risk_sample",
                "counterfactual_sample_rate": 0.0,
                "counterfactual_risk_threshold": 0.55,
            },
        },
    )

    assert len(client.messages) == 2
    assert pred["counterfactual_verification"]["enabled"] is False
    assert pred["counterfactual_verification"]["skip_reasons"] == ["low_risk_not_sampled"]
    assert pred["counterfactual_revision_triggered"] is False


def test_run_ours_medical_answer_entity_skips_second_pass_and_prefers_options(tmp_path):
    case = {
        "case_id": "case_mcqa",
        "task_profile": "medical_answer_entity",
        "demographics": {},
        "events": [
            {"event_id": "ev_1", "time": 0, "type": "clinical", "text": "Scabies is caused by which organism?"},
            {"event_id": "ev_2", "time": 1, "type": "clinical", "text": "Answer options: A. Mite; B. Fungus; C. Virus; D. Bacterium"},
        ],
        "answer_options": [
            {"label": "A", "text": "Mite"},
            {"label": "B", "text": "Fungus"},
            {"label": "C", "text": "Virus"},
            {"label": "D", "text": "Bacterium"},
        ],
        "synthetic_labs": [],
        "memory_seed": [],
        "poison_records": [],
        "counterfactuals": [{"intervention": "Remove or negate this evidence: mite"}],
        "data_quality_flags": {"source_dataset": "medmcqa", "source_type": "medical_mcqa"},
    }
    client = CounterfactualClient(
        [
            '{"primary_diagnosis":"parasitic infestation","diagnosis_list":["Mite"],"confidence":0.70,'
            '"evidence":["Answer options include Mite"],"reasoning_summary":"Mite is the answer entity"}',
        ]
    )

    pred = run_ours(
        case,
        client,
        memory_dir=tmp_path,
        strategy={
            "rounds": 1,
            "temperature": 0.0,
            "features": {
                "top_k_is_auto": True,
                "disable_memory_cleaning": True,
                "disable_counterfactual_verification": True,
            },
        },
    )

    assert len(client.messages) == 1
    assert pred["method"] == "medimem_topk8_round1"
    assert pred["primary_diagnosis"] == "Mite"
    assert pred["primary_selection_pass"]["source"] == "answer_options"
    assert pred["primary_selection_pass"]["constraint"] == "visible_medical_mcqa_options"
    assert pred["source_evidence_note_count"] == 0
    assert pred["diagnosis_candidate_count"] == 0
    assert pred["diagnosis_second_pass"] == {"enabled": False, "reason": "medical_answer_entity_option_locked"}


def test_run_ours_pmc_patients_full_skips_memory_cleaning_by_default(tmp_path):
    case = {
        "case_id": "case_pmc",
        "task_profile": "longitudinal_diagnosis",
        "data_quality_flags": {"source_dataset": "pmc_patients"},
        "demographics": {},
        "events": [
            {"event_id": "ev_1", "time": 0, "type": "clinical", "text": "Patient had fever and hypotension."},
            {"event_id": "ev_2", "time": 1, "type": "diagnosis", "text": "Blood culture confirmed Neisseria meningitidis infection."},
        ],
        "synthetic_labs": [],
        "memory_seed": [
            {
                "memory_id": "m1",
                "summary": "Initial stale impression suggested viral illness.",
                "status": "active",
                "tags": ["initial_hypothesis"],
                "confidence": 0.3,
                "time_scope": {"start": 0},
                "evidence_refs": ["ev_1"],
            }
        ],
        "poison_records": [],
        "counterfactuals": [{"intervention": "Remove culture result"}],
    }
    client = CounterfactualClient(
        [
            '{"primary_diagnosis":"Neisseria meningitidis infection","diagnosis_list":["Neisseria meningitidis infection"],'
            '"confidence":0.80,"evidence":["Blood culture confirmed infection"],"reasoning_summary":"culture supported"}',
        ]
    )

    pred = run_ours(
        case,
        client,
        memory_dir=tmp_path,
        strategy={
            "rounds": 1,
            "temperature": 0.0,
            "features": {
                "top_k_is_auto": True,
                "disable_counterfactual_verification": True,
            },
        },
    )

    assert pred["memory_ops"] == []
    assert pred["method"] == "medimem_topk8_round1"
    assert pred["optimization_features"]["profile_adaptive_memory_cleaning"] is True
    assert pred["primary_diagnosis"] == "Neisseria meningitidis infection"


def test_answer_entity_source_candidates_ignore_gold_candidate_markers():
    case = {
        "events": [
            {
                "text": "Assessment entity candidate: What information is obtainable regarding bacterial vaginosis?",
            }
        ]
    }

    assert answer_entity_source_candidates(case) == []

