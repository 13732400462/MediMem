from mem_ehr_agent.expanded_data import (
    MEDICAL_DATASET_SPECS,
    generic_row_to_pmoa_like,
    parse_source_names,
    source_rows_to_cases,
    validate_medical_source_cases,
)
from mem_ehr_agent.agents import case_context
from mem_ehr_agent.data_sources import load_cached_rows


def test_load_cached_rows_prefers_enough_nested_cache(tmp_path):
    root = tmp_path
    (root / "small.json").write_text('{"rows": [{"row": {"id": "small"}}]}', encoding="utf-8")
    nested = root / "hf_downloads" / "source"
    nested.mkdir(parents=True)
    (nested / "large.json").write_text(
        "\n".join(
            [
                '{"id": "row-1", "text": "one"}',
                '{"id": "row-2", "text": "two"}',
                '{"id": "row-3", "text": "three"}',
            ]
        ),
        encoding="utf-8",
    )
    rows = load_cached_rows(root, ["small.json", "hf_downloads/source/large.json"], 3)
    assert [row["id"] for row in rows] == ["row-1", "row-2", "row-3"]


def test_parse_source_names_defaults_to_configured_pool():
    names = parse_source_names(None)
    assert "pmoa_tts" in names
    assert "pmc_patients" in names
    assert "medmcqa" in names
    assert len(names) >= 7


def test_generic_medmcqa_row_can_be_converted_to_case_schema():
    row = {
        "id": "demo-medmcqa-1",
        "question": "A patient has fever and neck stiffness. What is the diagnosis?",
        "opa": "Migraine",
        "opb": "Meningitis",
        "opc": "Asthma",
        "opd": "Diabetes",
        "cop": 1,
        "exp": "Correct answer: meningitis because of fever and neck stiffness.",
        "subject_name": "Medicine",
    }
    spec = MEDICAL_DATASET_SPECS["medmcqa"]
    pmoa_like = generic_row_to_pmoa_like(row, spec, 1)
    assert pmoa_like["_source_dataset"] == "medmcqa"
    assert len(pmoa_like["textual_timeseries"]) >= 3
    runtime_text = " ".join(item["event"] for item in pmoa_like["textual_timeseries"])
    assert "Final answer or diagnosis target" not in runtime_text
    assert "Answer options: A. Migraine; B. Meningitis; C. Asthma; D. Diabetes" in runtime_text
    assert "Correct answer" not in runtime_text
    assert "Assessment entity candidate" not in runtime_text
    assert "Reference answer evidence" not in runtime_text
    cases = source_rows_to_cases(
        "medmcqa",
        [row],
        [{"patient_uid": "pmc-demo", "patient": "A similar patient had diagnostic revision after follow-up."}],
    )
    case = cases[0]
    assert case["case_id"] == "medmcqa_0001"
    assert case["source_refs"][0]["dataset"] == "medmcqa"
    assert case["task_profile"] == "medical_answer_entity"
    assert case["answer_options"][1]["text"] == "Meningitis"
    assert "Answer options: A. Migraine; B. Meningitis; C. Asthma; D. Diabetes" in case_context(case)
    assert case["labels"]["primary_diagnosis"]
    assert "Meningitis" in case["labels"]["label_aliases"]
    for poison in case["poison_records"]:
        assert "expected_op" not in poison
        assert "revised_claim" not in poison


def test_medmcqa_prompt_keeps_options_without_answer_markers():
    row = {
        "id": "demo-medmcqa-2",
        "question": "Scabies is caused by which organism?",
        "opa": "Mite",
        "opb": "Fungus",
        "opc": "Virus",
        "opd": "Bacterium",
        "cop": 0,
        "exp": "Ans. is 'a' Mite. Ref: textbook. Scabies causes intense pruritus.",
    }
    spec = MEDICAL_DATASET_SPECS["medmcqa"]
    pmoa_like = generic_row_to_pmoa_like(row, spec, 1)
    runtime_text = " ".join(item["event"] for item in pmoa_like["textual_timeseries"])
    assert "A. Mite" in runtime_text and "D. Bacterium" in runtime_text
    assert "Ans." not in runtime_text
    assert "Ref:" not in runtime_text
    assert "Assessment entity candidate" not in runtime_text


def test_medmcqa_primary_keeps_full_correct_option_text():
    row = {
        "id": "demo-medmcqa-3",
        "question": "Splenomegaly may be a feature of:",
        "opa": "Megaloblastic anemia",
        "opb": "Sickle cell anemia",
        "opc": "Thalassemia",
        "opd": "G6PD deficiency",
        "cop": 1,
        "exp": "Ans. B i.e. Sickle cell anemia.",
    }
    pmoa_like = generic_row_to_pmoa_like(row, MEDICAL_DATASET_SPECS["medmcqa"], 1)
    assert pmoa_like["diagnoses"] == ["Sickle cell anemia"]


def test_chatdoctor_response_is_compacted_to_assessment_label():
    row = {
        "id": "chatdoctor-demo-1",
        "instruction": "If you are a doctor, please answer the medical questions based on the patient's description.",
        "input": "I feel spinning when I turn in bed. What is this?",
        "output": (
            "Hi, Thank you for posting your query. The most likely cause for your symptoms is "
            "benign paroxysmal positional vertigo (BPPV). Please consult your doctor."
        ),
    }
    pmoa_like = generic_row_to_pmoa_like(row, MEDICAL_DATASET_SPECS["chatdoctor_healthcaremagic"], 1)
    assert pmoa_like["diagnoses"][0].lower() == "benign paroxysmal positional vertigo"
    assert len(pmoa_like["diagnoses"][0].split()) <= 8
    runtime_text = " ".join(item["event"] for item in pmoa_like["textual_timeseries"])
    assert "Doctor assessment:" not in runtime_text
    assert "benign paroxysmal positional vertigo" not in runtime_text.lower()
    cases = source_rows_to_cases(
        "chatdoctor_healthcaremagic",
        [row],
        [{"patient_uid": "pmc-demo", "patient": "Auxiliary context."}],
    )
    validate_medical_source_cases("chatdoctor_healthcaremagic", cases)
    assert cases[0]["labels"]["primary_diagnosis"].lower() == "benign paroxysmal positional vertigo"


def test_wikidoc_response_uses_short_entity_not_answer_paragraph():
    row = {
        "id": "wikidoc-demo-1",
        "instruction": "Answer this question truthfully",
        "input": "How is squamous cell carcinoma of the lung classified?",
        "output": (
            "Squamous cell carcinoma of the lung may be classified according to the WHO histological "
            "classification system into 4 main variants with distinct morphology."
        ),
    }
    pmoa_like = generic_row_to_pmoa_like(row, MEDICAL_DATASET_SPECS["medical_meadow_wikidoc"], 1)
    assert pmoa_like["diagnoses"][0].lower() == "squamous cell carcinoma of the lung"
    assert len(pmoa_like["diagnoses"][0]) < 80
    runtime_text = " ".join(item["event"] for item in pmoa_like["textual_timeseries"])
    assert "Reference answer evidence" not in runtime_text
    assert "WHO histological classification system" not in runtime_text
    cases = source_rows_to_cases(
        "medical_meadow_wikidoc",
        [row],
        [{"patient_uid": "pmc-demo", "patient": "Auxiliary context."}],
    )
    validate_medical_source_cases("medical_meadow_wikidoc", cases)


def test_wikidoc_question_topic_is_not_full_question_label():
    row = {
        "id": "wikidoc-demo-2",
        "instruction": "Answer this question truthfully",
        "input": "What information is there about measles?",
        "output": "Measles is a viral infection with fever, cough, conjunctivitis, and rash.",
    }
    pmoa_like = generic_row_to_pmoa_like(row, MEDICAL_DATASET_SPECS["medical_meadow_wikidoc"], 1)
    assert pmoa_like["diagnoses"][0].lower() == "measles"
    assert not pmoa_like["diagnoses"][0].lower().startswith("what information")
    cases = source_rows_to_cases(
        "medical_meadow_wikidoc",
        [row],
        [{"patient_uid": "pmc-demo", "patient": "Auxiliary context."}],
    )
    validate_medical_source_cases("medical_meadow_wikidoc", cases)


def test_medical_source_gate_rejects_placeholder_labels_and_dataset_answers():
    case = {
        "case_id": "bad_0001",
        "task_profile": "",
        "labels": {"primary_diagnosis": "medqa sample 1", "diagnosis_list": ["medqa sample 1"]},
        "qa_tasks": [{"qa_id": "bad_idr", "type": "IDR", "answer": "Dataset: medqa"}],
        "data_quality_flags": {},
    }
    try:
        validate_medical_source_cases("medqa", [case])
    except RuntimeError as exc:
        assert "data-quality gate failed" in str(exc)
        assert "bad primary labels" in str(exc)
        assert "placeholder qa answers" in str(exc)
    else:
        raise AssertionError("expected data-quality gate failure")


def test_medqa_nested_data_schema_uses_correct_answer_and_options():
    row = {
        "id": "medqa-demo-1",
        "data": {
            "Question": "A pregnant patient has cystitis. Which antibiotic is appropriate?",
            "Options": {
                "A": "Ampicillin",
                "B": "Ceftriaxone",
                "C": "Doxycycline",
                "D": "Nitrofurantoin",
            },
            "Correct Option": "D",
            "Correct Answer": "Nitrofurantoin",
        },
        "subject_name": "Medicine",
    }
    pmoa_like = generic_row_to_pmoa_like(row, MEDICAL_DATASET_SPECS["medqa"], 1)
    assert pmoa_like["diagnoses"] == ["Nitrofurantoin"]
    assert pmoa_like["_answer_options"][3] == {"label": "D", "text": "Nitrofurantoin"}
    runtime_text = " ".join(item["event"] for item in pmoa_like["textual_timeseries"])
    assert "Correct Option" not in runtime_text
    assert "Correct Answer" not in runtime_text
    assert "Assessment entity candidate" not in runtime_text
    cases = source_rows_to_cases(
        "medqa",
        [row],
        [{"patient_uid": "pmc-demo", "patient": "Auxiliary context."}],
    )
    validate_medical_source_cases("medqa", cases)
    assert cases[0]["labels"]["primary_diagnosis"] == "Nitrofurantoin"
    assert "D" in cases[0]["labels"]["label_aliases"]
