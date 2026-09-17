from __future__ import annotations

from typing import Any


REQUIRED_CASE_KEYS = {
    "case_id",
    "source_refs",
    "demographics",
    "encounters",
    "events",
    "memory_seed",
    "poison_records",
    "labels",
    "qa_tasks",
    "expected_memory_ops",
    "counterfactuals",
}


def validate_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_CASE_KEYS - set(case))
    if missing:
        errors.append(f"missing keys: {missing}")
    labels = case.get("labels") or {}
    if not labels.get("primary_diagnosis"):
        errors.append("labels.primary_diagnosis is empty")
    if not labels.get("diagnosis_list"):
        errors.append("labels.diagnosis_list is empty")
    if len(case.get("events") or []) < 3:
        errors.append("case must contain at least 3 events")
    if not any((q.get("type") == "CDR") for q in case.get("qa_tasks") or []):
        errors.append("case must contain at least one CDR qa_task")
    if not case.get("expected_memory_ops"):
        errors.append("case must contain at least one expected_memory_op")
    return errors


def validate_prediction(pred: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["case_id", "method", "primary_diagnosis", "diagnosis_list", "confidence", "evidence"]:
        if key not in pred:
            errors.append(f"missing prediction key: {key}")
    if pred.get("confidence") is not None:
        try:
            value = float(pred["confidence"])
        except (TypeError, ValueError):
            errors.append("confidence must be numeric")
        else:
            if value < 0 or value > 1:
                errors.append("confidence must be in [0, 1]")
    return errors

