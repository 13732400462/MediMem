from mem_ehr_agent.expanded_data import (
    MEDICAL_DATASET_SPECS,
    generic_row_to_pmoa_like,
    parse_source_names,
    source_rows_to_cases,
)


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
    cases = source_rows_to_cases(
        "medmcqa",
        [row],
        [{"patient_uid": "pmc-demo", "patient": "A similar patient had diagnostic revision after follow-up."}],
    )
    case = cases[0]
    assert case["case_id"] == "medmcqa_0001"
    assert case["source_refs"][0]["dataset"] == "medmcqa"
    assert case["labels"]["primary_diagnosis"]
    for poison in case["poison_records"]:
        assert "expected_op" not in poison
        assert "revised_claim" not in poison

