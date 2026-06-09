from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .task_profiles import CLINICAL_ASSESSMENT_ENTITY, MEDICAL_ANSWER_ENTITY


def compact_style_label(text: Any) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;()[]")
    if not cleaned:
        return ""
    return cleaned[:120]


def _bucket_label(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(?:administer|start|give|treat|therapy|surgery|supplement|mg|antibiotic)\b", low):
        return "treatment_or_action"
    if re.search(r"\b(?:vitamin|drug|enzyme|receptor|organism|bacteria|virus|mite|test|sign|finding)\b", low):
        return "medical_answer_entity"
    if re.search(r"\b(?:disease|syndrome|infection|cancer|carcinoma|diabetes|asthma|anemia|vertigo)\b", low):
        return "diagnosis_entity"
    return "short_answer_entity"


def learn_source_style_policy(
    source_name: str,
    task_profile: str,
    gold_labels: list[str],
    *,
    train_dev_count: int,
) -> dict[str, Any]:
    labels = [compact_style_label(item) for item in gold_labels]
    labels = [item for item in labels if item]
    word_counts = [len(item.split()) for item in labels]
    buckets = Counter(_bucket_label(item) for item in labels)
    median_words = sorted(word_counts)[len(word_counts) // 2] if word_counts else None
    if task_profile == MEDICAL_ANSWER_ENTITY:
        instruction = (
            "Return one compact answer entity at the same granularity as the visible question/options. "
            "For MCQA, choose the option text that best follows from the question; do not invent a diagnosis label."
        )
    elif task_profile == CLINICAL_ASSESSMENT_ENTITY:
        instruction = (
            "Return the most likely assessment entity inferred from patient-side evidence only. "
            "Use a short clinical entity, not a long advice paragraph."
        )
    else:
        instruction = (
            "Return the main longitudinal diagnosis supported by the visible timeline, with complications in the list."
        )
    return {
        "source": source_name,
        "task_profile": task_profile,
        "protocol": "no_test_gold_visible",
        "train_dev_count": int(train_dev_count),
        "median_label_words": median_words,
        "label_style_buckets": dict(sorted(buckets.items())),
        "prompt_policy": instruction,
    }


def style_policy_prompt(policy: dict[str, Any] | None) -> str:
    if not policy:
        return ""
    prompt_policy = str(policy.get("prompt_policy") or "").strip()
    if not prompt_policy:
        return ""
    buckets = policy.get("label_style_buckets") or {}
    bucket_text = ", ".join(f"{key}={value}" for key, value in sorted(buckets.items()))
    extra = f" Learned source label styles from disjoint train/dev rows: {bucket_text}." if bucket_text else ""
    return f"Source style policy: {prompt_policy}{extra}"
