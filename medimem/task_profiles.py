from __future__ import annotations

from typing import Any


LONGITUDINAL_DIAGNOSIS = "longitudinal_diagnosis"
MEDICAL_ANSWER_ENTITY = "medical_answer_entity"
CLINICAL_ASSESSMENT_ENTITY = "clinical_assessment_entity"

DEFAULT_TASK_PROFILE = LONGITUDINAL_DIAGNOSIS

SOURCE_TASK_PROFILES = {
    "pmoa_tts": LONGITUDINAL_DIAGNOSIS,
    "pmc_patients": LONGITUDINAL_DIAGNOSIS,
    "medmcqa": MEDICAL_ANSWER_ENTITY,
    "medqa": MEDICAL_ANSWER_ENTITY,
    "medical_meadow_wikidoc": MEDICAL_ANSWER_ENTITY,
    "chatdoctor_healthcaremagic": CLINICAL_ASSESSMENT_ENTITY,
    "medical_dialogue_to_soap": CLINICAL_ASSESSMENT_ENTITY,
}


def task_profile_for_source(source_name: str) -> str:
    return SOURCE_TASK_PROFILES.get(str(source_name), DEFAULT_TASK_PROFILE)


def case_task_profile(case: dict[str, Any]) -> str:
    explicit = str(case.get("task_profile") or "").strip()
    if explicit:
        return explicit
    flags = case.get("data_quality_flags") or {}
    return str(flags.get("task_profile") or DEFAULT_TASK_PROFILE)


def diagnosis_list_budget(profile: str) -> tuple[int, int]:
    if profile == MEDICAL_ANSWER_ENTITY:
        return 1, 3
    if profile == CLINICAL_ASSESSMENT_ENTITY:
        return 3, 5
    return 5, 8


def task_profile_prompt_policy(profile: str) -> str:
    if profile == MEDICAL_ANSWER_ENTITY:
        return (
            "Task profile: medical_answer_entity. The primary_diagnosis field should contain the answer entity "
            "at the same granularity as the correct option or short-answer label. It may be a drug, organism, "
            "vitamin, mechanism, sign, test finding, or disease. Do not force the answer into a disease name. "
            "diagnosis_list should contain 1 to 3 compact answer-relevant entities only."
        )
    if profile == CLINICAL_ASSESSMENT_ENTITY:
        return (
            "Task profile: clinical_assessment_entity. The primary_diagnosis field should contain the doctor's "
            "assessment or diagnosis entity, not a broad symptom dump. diagnosis_list should contain 3 to 5 "
            "assessment-relevant entities when available."
        )
    return (
        "Task profile: longitudinal_diagnosis. The primary_diagnosis field should contain the main disease, "
        "admission diagnosis, or case-title diagnosis supported by the longitudinal timeline. Complications, "
        "mechanisms, later events, anatomy/pathology entities, and comorbidities belong in diagnosis_list unless "
        "they are clearly the main diagnosis. diagnosis_list should contain 5 to 8 evidence-supported entities "
        "when available."
    )
