from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_builder import build_case, write_prefix_slices
from .data_sources import fetch_hf_rows, fetch_pmc_patient_rows, fetch_pmoa_rows
from .io_utils import ensure_dir, write_jsonl, write_text
from .medical_terms import canonicalize_diagnosis
from .style_policy import learn_source_style_policy
from .task_profiles import task_profile_for_source


GOLD_ONLY_EVIDENCE_PREFIXES = (
    "Assessment entity candidate",
    "Reference answer evidence",
    "Doctor assessment",
    "Correct answer",
    "Correct option",
)


@dataclass(frozen=True)
class MedicalDatasetSpec:
    name: str
    hf_dataset: str | None
    preferred_splits: tuple[str, ...]
    source_type: str
    url: str


MEDICAL_DATASET_SPECS: dict[str, MedicalDatasetSpec] = {
    "pmoa_tts": MedicalDatasetSpec(
        name="pmoa_tts",
        hf_dataset="snoroozi/pmoa-tts",
        preferred_splits=("train_DSR1", "train", "case_study_25k_DSR1", "case_study_25k_L33"),
        source_type="longitudinal_case",
        url="https://huggingface.co/datasets/snoroozi/pmoa-tts",
    ),
    "pmc_patients": MedicalDatasetSpec(
        name="pmc_patients",
        hf_dataset="aisc-team-b1/PMC-Patients",
        preferred_splits=("train",),
        source_type="patient_summary",
        url="https://huggingface.co/datasets/aisc-team-b1/PMC-Patients",
    ),
    "medical_meadow_wikidoc": MedicalDatasetSpec(
        name="medical_meadow_wikidoc",
        hf_dataset="medalpaca/medical_meadow_wikidoc",
        preferred_splits=("train",),
        source_type="medical_instruction",
        url="https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc",
    ),
    "medmcqa": MedicalDatasetSpec(
        name="medmcqa",
        hf_dataset="openlifescienceai/medmcqa",
        preferred_splits=("train", "validation", "test"),
        source_type="medical_mcqa",
        url="https://huggingface.co/datasets/openlifescienceai/medmcqa",
    ),
    "medqa": MedicalDatasetSpec(
        name="medqa",
        hf_dataset="openlifescienceai/medqa",
        preferred_splits=("train", "dev", "test"),
        source_type="medical_mcqa",
        url="https://huggingface.co/datasets/openlifescienceai/medqa",
    ),
    "chatdoctor_healthcaremagic": MedicalDatasetSpec(
        name="chatdoctor_healthcaremagic",
        hf_dataset="lavita/ChatDoctor-HealthCareMagic-100k",
        preferred_splits=("train",),
        source_type="medical_dialogue",
        url="https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k",
    ),
    "medical_dialogue_to_soap": MedicalDatasetSpec(
        name="medical_dialogue_to_soap",
        hf_dataset="omi-health/medical-dialogue-to-soap-summary",
        preferred_splits=("train", "validation", "test"),
        source_type="medical_dialogue_summary",
        url="https://huggingface.co/datasets/omi-health/medical-dialogue-to-soap-summary",
    ),
}


EXCLUDED_FORMAL_POOL_SOURCES = frozenset({"medical_dialogue_to_soap"})
DEFAULT_MEDICAL_POOL_SOURCES = tuple(
    name for name in MEDICAL_DATASET_SPECS if name not in EXCLUDED_FORMAL_POOL_SOURCES
)
DEFAULT_PREFIX_SIZES = (50, 100, 250, 500, 1000)


def parse_source_names(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_MEDICAL_POOL_SOURCES)
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in MEDICAL_DATASET_SPECS]
    if unknown:
        raise ValueError(f"Unknown medical data sources: {unknown}. Known: {sorted(MEDICAL_DATASET_SPECS)}")
    return names


def stable_source_id(source_name: str, row: dict[str, Any], idx: int) -> str:
    for key in ("id", "case_report_id", "pmc_id", "patient_uid", "patient_id"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    digest = hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return f"{source_name}-{idx:06d}-{digest}"


def source_row_identity(row: dict[str, Any]) -> str:
    for key in ("_source_id", "id", "case_report_id", "pmc_id", "patient_uid", "patient_id"):
        value = row.get(key)
        if value not in {None, ""}:
            return f"{key}:{value}"
    return "sha1:" + hashlib.sha1(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def source_row_dedupe_key(source_name: str, row: dict[str, Any]) -> str:
    identity = source_row_identity(row)
    if not identity.startswith("sha1:"):
        return f"{source_name}:{identity}"
    stable_fields = []
    for key in (
        "title",
        "question",
        "input",
        "instruction",
        "output",
        "patient",
        "summary",
        "text",
        "answer",
        "target",
    ):
        value = row_field_value(row, key)
        if isinstance(value, str) and value.strip():
            stable_fields.append(re.sub(r"\s+", " ", value).strip().lower())
        elif isinstance(value, (int, float)):
            stable_fields.append(str(value))
    if stable_fields:
        digest = hashlib.sha1("\n".join(stable_fields).encode("utf-8")).hexdigest()
        return f"{source_name}:stable:{digest}"
    return f"{source_name}:{identity}"


def dedupe_source_rows(source_name: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = source_row_dedupe_key(source_name, row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, len(rows) - len(deduped)


def case_source_identity(case: dict[str, Any]) -> str:
    flags = case.get("data_quality_flags") or {}
    dataset = str(flags.get("source_dataset") or "")
    source_id = str(flags.get("source_id") or case.get("case_id") or "")
    return f"{dataset}:{source_id}"


def dedupe_cases_by_source_id(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for case in cases:
        key = case_source_identity(case)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped, len(cases) - len(deduped)


def row_field_value(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    data = row.get("data")
    if not isinstance(data, dict):
        return None
    aliases = {
        "question": ("Question",),
        "answer": ("Correct Answer", "answer"),
        "target": ("Correct Answer", "target"),
        "cop": ("Correct Option",),
        "answer_idx": ("Correct Option",),
        "opa": ("A",),
        "opb": ("B",),
        "opc": ("C",),
        "opd": ("D",),
    }
    options = data.get("Options") if isinstance(data.get("Options"), dict) else {}
    for alias in aliases.get(key, (key,)):
        if alias in data:
            return data.get(alias)
        if alias in options:
            return options.get(alias)
    return None


def text_value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row_field_value(row, key)
        if isinstance(value, str) and value.strip():
            return re.sub(r"\s+", " ", value).strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def option_answer(row: dict[str, Any]) -> str:
    options = [
        text_value(row, "opa", "A"),
        text_value(row, "opb", "B"),
        text_value(row, "opc", "C"),
        text_value(row, "opd", "D"),
    ]
    raw = row_field_value(row, "cop") or row_field_value(row, "answer_idx") or row_field_value(row, "answer")
    if isinstance(raw, int) and 0 <= raw < len(options) and options[raw]:
        return options[raw]
    if isinstance(raw, str):
        low = raw.strip().lower()
        label_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3, "0": 0, "1": 1, "2": 2, "3": 3}
        if low in label_to_idx and options[label_to_idx[low]]:
            return options[label_to_idx[low]]
        if raw.strip():
            return raw.strip()
    return next((option for option in options if option), text_value(row, "output", "answer", "target"))


def answer_options(row: dict[str, Any]) -> list[dict[str, str]]:
    options = []
    for label, keys in (
        ("A", ("opa", "A")),
        ("B", ("opb", "B")),
        ("C", ("opc", "C")),
        ("D", ("opd", "D")),
        ("E", ("ope", "E")),
    ):
        value = text_value(row, *keys)
        if value:
            options.append({"label": label, "text": value})
    return options


def option_aliases(row: dict[str, Any], answer: str) -> list[str]:
    aliases = [answer] if answer else []
    options = answer_options(row)
    raw = row_field_value(row, "cop") or row_field_value(row, "answer_idx") or row_field_value(row, "answer")
    label_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
    if isinstance(raw, int) and 0 <= raw < len(options):
        aliases.append(options[raw]["label"])
    elif isinstance(raw, str):
        low = raw.strip().lower()
        if low in label_to_idx and label_to_idx[low] < len(options):
            aliases.append(options[label_to_idx[low]]["label"])
        elif raw.strip() and raw.strip() != answer:
            aliases.append(raw.strip())
    out = []
    seen = set()
    for alias in aliases:
        key = re.sub(r"\s+", " ", str(alias).lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(str(alias))
    return out


def scrub_answer_explanation(text: str, answer: str = "") -> str:
    cleaned = re.sub(r"\bRef\s*:.*$", "", str(text or ""), flags=re.I).strip()
    if not cleaned:
        return ""
    sentences = split_sentences(cleaned, limit=12)
    kept = []
    answer_norm = re.escape(str(answer or "").strip())
    marker_re = re.compile(
        r"\b(?:correct\s+answer|answer|ans\.?|option)\b\s*(?:is|:|=|-)?\s*(?:['\"]?[a-e]['\"]?|['\"]?[^.;]{1,80}['\"]?)",
        flags=re.I,
    )
    for sentence in sentences:
        low = sentence.lower()
        if marker_re.search(sentence):
            continue
        if answer and re.search(r"\b(?:is|was|are|were)\s+['\"]?" + answer_norm + r"['\"]?", sentence, flags=re.I):
            continue
        if "correct answer" in low or "ans." in low:
            continue
        kept.append(sentence)
    return " ".join(kept[:4]).strip()


def split_sentences(text: str, *, limit: int = 8) -> list[str]:
    chunks = [item.strip() for item in re.split(r"(?<=[.;?!])\s+|\n+", text) if item.strip()]
    return chunks[:limit] if chunks else ([text.strip()] if text.strip() else [])


PLACEHOLDER_LABEL_RE = re.compile(
    r"^(?:"
    r"(?:medical_)?(?:answer entity|instruction|dialogue|dialogue summary)|"
    r"(?:medqa|medmcqa|medical_meadow_wikidoc|chatdoctor_healthcaremagic|medical_dialogue_to_soap)\s+sample\s+\d+|"
    r"(?:this\s+)?question truthfully|medical questions based on the patient|"
    r"dataset:\s*|source type:"
    r")",
    flags=re.I,
)


def compact_entity(text: str, *, max_words: int = 10, max_chars: int = 90) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;\"'")
    cleaned = re.sub(r"^(?:the|a|an|your|patient'?s)\s+", "", cleaned, flags=re.I)
    cleaned = re.split(
        r"\s+(?:because|which|that|when|while|although|however|therefore|with evidence of|may be|can be|is classified|are classified|classified according)\s+",
        cleaned,
        maxsplit=1,
        flags=re.I,
    )[0]
    cleaned = re.split(
        r"\s+(?:is used to|are used to|is part of|are part of|is associated with|are associated with|accounting for|should be|needs to be)\s+",
        cleaned,
        maxsplit=1,
        flags=re.I,
    )[0]
    cleaned = re.sub(r"\s*\([A-Z0-9 -]{2,12}\)\s*$", "", cleaned).strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return canonicalize_diagnosis(cleaned).strip(" .,:;\"'")


def is_bad_label(text: str) -> bool:
    label = str(text or "").strip()
    if not label:
        return True
    if label.lower() in {"there", "this", "that", "these", "those", "it", "hi dr age", "bites"}:
        return True
    if PLACEHOLDER_LABEL_RE.search(label):
        return True
    if len(label) > 120 or len(label.split()) > 16:
        return True
    return False


def is_question_like_label(text: str) -> bool:
    return bool(re.match(r"^\s*(?:what|how|which|when|where|why|can you|could you|provide|describe)\b", str(text or ""), flags=re.I))


def diagnosis_from_text(text: str, fallback: str, *, allow_long_fallback: bool = False) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip(" .,:;")
    if not cleaned:
        return [fallback]
    candidates: list[str] = []
    patterns = [
        r"(?:diagnosis|diagnosed with|diagnosed as|assessment|impression)[:\s]+([^.;\n]+)",
        r"(?:answer|correct answer)[:\s]+([^.;\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.I)
        if match:
            candidates.append(compact_entity(match.group(1)))
    if not candidates:
        candidates.append(cleaned[:120] if allow_long_fallback else compact_entity(cleaned))
    usable = [candidate for candidate in candidates if candidate and not is_bad_label(candidate)]
    fallback_compact = compact_entity(fallback)
    return usable or ([fallback_compact] if fallback_compact and not is_bad_label(fallback_compact) else ["Medical answer entity"])


def clean_entity_candidate(text: str, *, max_words: int = 10, max_chars: int = 90) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;")
    cleaned = re.split(
        r"\b(?:which|that|but|although|while|whereas|and antibiotic|and was|and were|at our|on a|on an|based on)\b",
        cleaned,
        maxsplit=1,
        flags=re.I,
    )[0]
    cleaned = re.sub(r"^(?:the\s+)?(?:presence of|diagnosis of|diagnosed with|diagnosed as)\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.I)
    return compact_entity(cleaned, max_words=max_words, max_chars=max_chars)


def pmc_diagnosis_entities(title: str, patient: str) -> list[str]:
    text = re.sub(r"\s+", " ", " ".join(part for part in [title, patient] if part)).strip()
    patterns = [
        r"(?:diagnosis of|diagnosed with|diagnosed as|consistent with|compatible with)\s+([^.;]+)",
        r"(?:secondary to|caused by|due to)\s+([^.;]+)",
        r"(?:presence of)\s+([^.;]+)",
        r"\b(wet\s+AMD|dry\s+AMD|low-grade\s+glioma|septic shock|epilepsy)\b",
    ]
    out: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            entity = clean_entity_candidate(match.group(1))
            if entity and not is_bad_label(entity):
                out.append(entity)
        if out:
            break
    if not out:
        out.extend(diagnosis_from_text(title, fallback="Complex patient summary diagnosis"))
    return dedupe_text(out)[:3]


def chatdoctor_assessment_entities(response: str, question: str = "") -> list[str]:
    text = re.sub(r"\s+", " ", str(response or "")).strip()
    text = re.sub(r"^(?:hi|hello|dear)[,.\s]+(?:thank you[^.]*\.)?", "", text, flags=re.I).strip()
    patterns = [
        r"(?:most likely cause|likely cause|probable cause|possible cause)\s+(?:(?:of|for) [^.;]{1,80}?\s+)?(?:is|would be|could be)\s+([^.;]+)",
        r"(?:diagnosis|impression|assessment)\s+(?:is|would be|could be|:)\s+([^.;]+)",
        r"(?:seems|appears)\s+(?:that\s+)?(?:you|your child|your kid|the patient|he|she|it)?\s*(?:is|are|may be|might be)?\s*(?:having|suffering from|with)?\s+([^.;]+)",
        r"(?:suggestive of|consistent with|due to)\s+([^.;]+)",
        r"(?:may be having|might be having|could be having|having)\s+([^.;]+)",
    ]
    out: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            entity = compact_entity(match.group(1), max_words=8, max_chars=80)
            if entity and not is_bad_label(entity):
                out.append(entity)
    if not out:
        for sentence in split_sentences(text, limit=10):
            if re.search(r"\b(thank|hello|hi\b|understand|consult|regards)\b", sentence, flags=re.I):
                continue
            if not re.search(
                r"\b(diagnos|lesion|lump|infection|effusion|cancer|metastasis|viral|diarrhea|bronchiolitis|vertigo|syndrome|asthma|ulcer|fracture|tumou?r)\b",
                sentence,
                flags=re.I,
            ):
                continue
            entity = compact_entity(sentence, max_words=8, max_chars=80)
            if entity and not is_bad_label(entity):
                out.append(entity)
                break
    if not out and not is_bad_label(question):
        out.extend(diagnosis_from_text(question, fallback="clinical assessment"))
    return dedupe_text(out)[:4]


def soap_assessment_entities(summary: str, dialogue: str = "") -> list[str]:
    text = str(summary or "")
    if not text.strip():
        return []
    section_patterns = [
        r"\b(?:assessment\s*/\s*plan|assessment\s+and\s+plan|assessment|diagnosis|diagnoses|impression)\s*[:\-]\s*(.+?)(?=\s+(?:subjective|objective|plan|assessment|diagnosis|diagnoses|impression)\s*[:\-]|\Z)",
        r"(?:^|\n)\s*(?:assessment\s*/\s*plan|assessment\s+and\s+plan|assessment|diagnosis|diagnoses|impression)\s*[:\-]\s*(.+?)(?=\n\s*(?:subjective|objective|plan|assessment|diagnosis|diagnoses|impression)\s*[:\-]|\Z)",
        r"\b(?:assessment|diagnosis|impression)\s+(?:is|was|:)\s+([^.\n;]+)",
    ]
    candidates: list[str] = []
    for pattern in section_patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            section = re.sub(r"\s+", " ", match.group(1)).strip()
            for part in re.split(r"\s*(?:;|\||, and |\band\b|\d+\.)\s*", section):
                part = re.sub(
                    r"^(?:the\s+)?(?:patient\s+)?(?:has|with|is|was|likely|probably|possible|suspected|assessment of|diagnosis of)\s+",
                    "",
                    part.strip(" .,:;-"),
                    flags=re.I,
                )
                entity = compact_entity(part, max_words=8, max_chars=80)
                if entity and not is_bad_label(entity) and not is_question_like_label(entity):
                    candidates.append(entity)
            if candidates:
                return dedupe_text(candidates)[:4]
    combined = " ".join(part for part in [text, dialogue] if part)
    for pattern in (
        r"\b(?:consistent with|suggestive of|concerning for|due to|secondary to)\s+([^.;\n]+)",
        r"\b(?:acute|chronic|recurrent)\s+([A-Za-z][A-Za-z -]{3,80}(?:infection|syndrome|disease|pain|injury|failure|exacerbation))\b",
    ):
        for match in re.finditer(pattern, combined, flags=re.I):
            entity = compact_entity(match.group(1), max_words=8, max_chars=80)
            if entity and not is_bad_label(entity) and not is_question_like_label(entity):
                candidates.append(entity)
        if candidates:
            break
    return dedupe_text(candidates)[:4]


def wikidoc_answer_entities(answer: str, question: str = "", subject: str = "") -> list[str]:
    text = re.sub(r"\s+", " ", str(answer or "")).strip()
    patterns = [
        r"^([A-Z][A-Za-z0-9 -]+(?:cancer|carcinoma|tumou?r|tumors?|disease|syndrome|infection|deficiency|hypotension|vulvovaginitis|asthma|diabetes|hepatitis|anemia|inhibitors?)(?: of the [A-Za-z -]+)?)\b",
        r"^(?:The\s+)?([^.;]{3,80}?)\s+(?:is|are|may be|can be|refers to|accounting for)\b",
        r"(?:called|known as|termed)\s+([^.;]{3,80})",
    ]
    topic = wikidoc_question_topic_entity(question)
    out: list[str] = [topic] if topic else []
    for pattern in patterns:
        if out:
            break
        match = re.search(pattern, text, flags=re.I)
        if match:
            entity = compact_entity(match.group(1), max_words=10, max_chars=90)
            if entity and not is_bad_label(entity):
                out.append(entity)
                break
    if not out:
        topic = wikidoc_question_topic_entity(question)
        if topic:
            out.append(topic)
    if not out and subject and not is_bad_label(subject) and not is_question_like_label(subject):
        out.append(compact_entity(subject, max_words=8, max_chars=80))
    if out == ["Medical answer entity"] and text:
        out = [compact_entity(text, max_words=10, max_chars=90)]
    cleaned = [item for item in out if item and not is_question_like_label(item)]
    if not cleaned and text:
        entity = compact_entity(text, max_words=10, max_chars=90)
        if entity and not is_question_like_label(entity) and not is_bad_label(entity):
            cleaned.append(entity)
    return dedupe_text(cleaned or ["Medical answer entity"])[:3]


def wikidoc_question_topic_entity(question: str) -> str:
    q = re.sub(r"\s+", " ", str(question or "")).strip(" .?")
    if not q:
        return ""
    patterns = [
        r"(?:how is|how are)\s+(.+?)\s+(?:classified|diagnosed|treated|managed)$",
        r"(?:meaning|definition|summary|overview|information|history|symptoms|treatment|management|screening|physical examination|medical treatment|recommended medical treatment)\s+(?:of|for|about|on)\s+(.+)$",
        r"(?:what information (?:is there|is available)|can you provide information|could you provide information)\s+(?:about|on)\s+(.+)$",
        r"(?:what is|what are|could you explain|can you explain|could you tell me what|what does)\s+(.+?)\s+(?:mean|means|entail|refer to)$",
        r"(?:what is|what are)\s+(.+)$",
        r"(?:for|of|about|on)\s+([A-Za-z][A-Za-z0-9' -]{3,120})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.I)
        if not match:
            continue
        entity = match.group(1)
        entity = re.sub(r"^(?:the|a|an)\s+", "", entity, flags=re.I)
        entity = re.split(r"\s+(?:and how|and what|using|in physiology|in medicine|currently accessible)\b", entity, maxsplit=1, flags=re.I)[0]
        entity = re.sub(r"\s+(?:classified|diagnosed|treated|managed)$", "", entity, flags=re.I)
        entity = compact_entity(entity, max_words=10, max_chars=90)
        if entity and not is_bad_label(entity) and not is_question_like_label(entity):
            return entity
    return ""


def dedupe_text(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()
        if text and key and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def clamp_event_text(text: str, *, max_chars: int = 900) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..."


def generic_row_to_pmoa_like(row: dict[str, Any], spec: MedicalDatasetSpec, idx: int) -> dict[str, Any]:
    source_id = stable_source_id(spec.name, row, idx)
    title = text_value(row, "title", "question", "input", "instruction", "chief_complaint") or f"{spec.name} sample {idx}"
    question = text_value(row, "question", "input", "dialogue", "dialog", "instruction")
    context = text_value(row, "context", "patient", "note", "conversation", "dialogue", "dialog")
    answer = option_answer(row)
    options = answer_options(row)
    if spec.source_type != "medical_mcqa":
        answer = text_value(row, "answer", "target")
    explanation = text_value(row, "exp", "explanation", "output", "response", "soap_summary", "summary")
    safe_explanation = scrub_answer_explanation(explanation, answer) if spec.source_type == "medical_mcqa" else explanation
    subject = text_value(row, "subject_name", "topic_name", "category", "department")
    if spec.name == "chatdoctor_healthcaremagic":
        diagnoses = chatdoctor_assessment_entities(explanation or answer, question or context or title)
    elif spec.name == "medical_dialogue_to_soap":
        diagnoses = soap_assessment_entities(explanation or answer, question or context or title)
        if not diagnoses:
            diagnoses = ["Medical dialogue SOAP assessment unavailable"]
    elif spec.name == "medical_meadow_wikidoc":
        diagnoses = wikidoc_answer_entities(explanation or answer, question or title, subject)
    elif spec.source_type == "medical_mcqa" and answer:
        diagnoses = [clamp_event_text(answer, max_chars=140).strip(" .,:;")]
    else:
        diagnosis_seed = answer or explanation or subject or title
        diagnoses = diagnosis_from_text(diagnosis_seed, fallback=subject or "Medical answer entity")
    diagnosis_metric_applicable = not any(is_bad_label(item) or is_question_like_label(item) for item in diagnoses)
    if spec.name == "medical_dialogue_to_soap" and diagnoses == ["Medical dialogue SOAP assessment unavailable"]:
        diagnosis_metric_applicable = False

    event_texts = []
    if title and not re.fullmatch(r"(?:answer this question truthfully|if you are a doctor.*)", title, flags=re.I):
        event_texts.append(clamp_event_text(f"Initial clinical task/source title: {title}"))
    if context and context != question:
        event_texts.extend(clamp_event_text(item) for item in split_sentences(context, limit=4))
    if question:
        event_texts.append(clamp_event_text(f"Clinical question or presentation: {question}"))
    if options:
        option_text = "; ".join(f"{item['label']}. {item['text']}" for item in options)
        event_texts.append(clamp_event_text(f"Answer options: {option_text}"))
    # Gold/reference answer text is label-only under the no-leak protocol. Keep it
    # out of runtime events even when it would be useful extraction evidence.
    if len(event_texts) < 3:
        event_texts.extend([f"Source type: {spec.source_type}", "No additional patient-side source text was available."])

    return {
        "case_report_id": source_id,
        "pmc_id": source_id,
        "title": title,
        "patient": context or question or title,
        "diagnoses": diagnoses,
        "death_info": {},
        "textual_timeseries": [{"time": i * 7, "event": event} for i, event in enumerate(event_texts[:12])],
        "_source_dataset": spec.name,
        "_source_type": spec.source_type,
        "_source_url": spec.url,
        "_source_id": source_id,
        "_task_profile": task_profile_for_source(spec.name),
        "_answer_options": options,
        "_label_aliases": option_aliases(row, answer),
        "_diagnosis_metric_applicable": diagnosis_metric_applicable,
        "_gold_only": {
            "answer": answer,
            "explanation": explanation,
            "safe_explanation": safe_explanation,
            "diagnoses": diagnoses,
        },
    }


def pmc_row_to_pmoa_like(row: dict[str, Any], spec: MedicalDatasetSpec, idx: int) -> dict[str, Any]:
    source_id = stable_source_id(spec.name, row, idx)
    patient = text_value(row, "patient", "summary", "title") or f"PMC patient sample {idx}"
    title = text_value(row, "title") or patient[:120]
    diagnoses = pmc_diagnosis_entities(title, patient)
    events = [clamp_event_text(item) for item in split_sentences(patient, limit=10)]
    if len(events) < 3:
        events.extend([title, "Longitudinal patient summary context"])
    return {
        "case_report_id": source_id,
        "pmc_id": source_id,
        "title": title,
        "patient": patient,
        "diagnoses": diagnoses,
        "death_info": {},
        "textual_timeseries": [{"time": i * 14, "event": event} for i, event in enumerate(events[:12])],
        "_source_dataset": spec.name,
        "_source_type": spec.source_type,
        "_source_url": spec.url,
        "_source_id": source_id,
        "_task_profile": task_profile_for_source(spec.name),
        "_answer_options": [],
        "_label_aliases": diagnoses,
        "_prefer_explicit_primary": True,
    }


def rows_for_medical_source(
    source_name: str,
    n: int,
    *,
    cache_dir: str | Path | None = None,
    require_real_data: bool = False,
    notes: list[str] | None = None,
    random_seed: int | None = None,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    notes = notes if notes is not None else []
    spec = MEDICAL_DATASET_SPECS[source_name]
    pool_multiplier = max(1, int(os.getenv("MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER", "5") or 5))
    request_n = n * pool_multiplier if random_seed is not None else n
    if source_name == "pmoa_tts":
        rows = fetch_pmoa_rows(request_n, notes, cache_dir=cache_dir)
    elif source_name == "pmc_patients":
        rows = fetch_pmc_patient_rows(request_n, notes, cache_dir=cache_dir)
    else:
        if not spec.hf_dataset:
            rows = []
        else:
            rows = fetch_hf_rows(
                spec.hf_dataset,
                preferred_splits=list(spec.preferred_splits),
                n=request_n,
                notes=notes,
                cache_dir=cache_dir,
            )
    raw_loaded_count = len(rows)
    rows, duplicate_count = dedupe_source_rows(source_name, rows)
    if stats is not None:
        stats["raw_loaded_count"] = raw_loaded_count
        stats["deduped_candidate_count"] = len(rows)
        stats["duplicate_row_count"] = duplicate_count
    if duplicate_count:
        notes.append(
            f"{source_name}: deduped source rows raw_loaded={raw_loaded_count} "
            f"deduped={len(rows)} duplicates={duplicate_count}."
        )
    if require_real_data and len(rows) < n:
        raise RuntimeError(f"{source_name} requires {n} real rows but only {len(rows)} were loaded.")
    if random_seed is not None and len(rows) > n:
        pool_size = len(rows)
        rng = random.Random(f"{random_seed}:{source_name}")
        rows = rng.sample(rows, n)
        notes.append(f"{source_name}: sampled {n} rows from pool={pool_size} with seed={random_seed}.")
    return rows[:n]


def rows_for_medical_source_split(
    source_name: str,
    n: int,
    *,
    cache_dir: str | Path | None = None,
    require_real_data: bool = False,
    notes: list[str] | None = None,
    random_seed: int | None = None,
    style_n: int | None = None,
    stats: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    notes = notes if notes is not None else []
    spec = MEDICAL_DATASET_SPECS[source_name]
    pool_multiplier = max(2, int(os.getenv("MEDICAL_SOURCE_SAMPLE_POOL_MULTIPLIER", "5") or 5))
    style_target = max(0, int(style_n if style_n is not None else min(max(n, 50), 500)))
    request_n = max(n + style_target, n * pool_multiplier) if random_seed is not None else n + style_target
    if source_name == "pmoa_tts":
        pool = fetch_pmoa_rows(request_n, notes, cache_dir=cache_dir)
    elif source_name == "pmc_patients":
        pool = fetch_pmc_patient_rows(request_n, notes, cache_dir=cache_dir)
    else:
        pool = (
            fetch_hf_rows(
                spec.hf_dataset,
                preferred_splits=list(spec.preferred_splits),
                n=request_n,
                notes=notes,
                cache_dir=cache_dir,
            )
            if spec.hf_dataset
            else []
        )
    raw_loaded_count = len(pool)
    pool, duplicate_count = dedupe_source_rows(source_name, pool)
    if stats is not None:
        stats["raw_loaded_count"] = raw_loaded_count
        stats["deduped_candidate_count"] = len(pool)
        stats["duplicate_row_count"] = duplicate_count
    if duplicate_count:
        notes.append(
            f"{source_name}: deduped source split pool raw_loaded={raw_loaded_count} "
            f"deduped={len(pool)} duplicates={duplicate_count}."
        )
    if require_real_data and len(pool) < n:
        raise RuntimeError(f"{source_name} requires {n} real rows but only {len(pool)} were loaded.")
    if random_seed is None:
        return pool[:n], pool[n : n + style_target]
    rng = random.Random(f"{random_seed}:{source_name}:noleak_split")
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    test_indices = set(indices[: min(n, len(indices))])
    test_rows = [pool[i] for i in indices[: min(n, len(indices))]]
    test_keys = {source_row_identity(row) for row in test_rows}
    style_rows = [
        pool[i]
        for i in indices[min(n, len(indices)) :]
        if i not in test_indices and source_row_identity(pool[i]) not in test_keys
    ][:style_target]
    notes.append(
        f"{source_name}: sampled test={len(test_rows)} style_train_dev={len(style_rows)} "
        f"from pool={len(pool)} with seed={random_seed} under no-leak split."
    )
    return test_rows, style_rows


def source_rows_to_cases(
    source_name: str,
    rows: list[dict[str, Any]],
    pmc_context_rows: list[dict[str, Any]],
    *,
    start_idx: int = 1,
    style_policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    spec = MEDICAL_DATASET_SPECS[source_name]
    cases: list[dict[str, Any]] = []
    if not pmc_context_rows:
        pmc_context_rows = [{"patient_uid": "missing-pmc-context", "patient": "No auxiliary patient context was available."}]
    for offset, row in enumerate(rows):
        idx = start_idx + offset
        if source_name == "pmoa_tts":
            pmoa_like = dict(row)
            pmoa_like.update(
                {
                    "_source_dataset": spec.name,
                    "_source_type": spec.source_type,
                    "_source_url": spec.url,
                    "_source_id": stable_source_id(spec.name, row, idx),
                    "_task_profile": task_profile_for_source(spec.name),
                    "_answer_options": [],
                    "_label_aliases": row.get("diagnoses") or row.get("diagnosis") or [],
                }
            )
        elif source_name == "pmc_patients":
            pmoa_like = pmc_row_to_pmoa_like(row, spec, idx)
        else:
            pmoa_like = generic_row_to_pmoa_like(row, spec, idx)
        pmc_row = pmc_context_rows[offset % len(pmc_context_rows)]
        case = build_case(pmoa_like, pmc_row, idx)
        source_id = str(pmoa_like.get("_source_id") or stable_source_id(source_name, row, idx))
        case["case_id"] = f"{source_name}_{offset + 1:04d}"
        for qa in case.get("qa_tasks", []):
            qa_type = str(qa.get("type") or "qa").lower()
            qa["qa_id"] = f"{case['case_id']}_{qa_type}"
        case["source_refs"] = [
            {"dataset": spec.name, "id": source_id, "url": spec.url, "role": "primary_source"},
            {
                "dataset": "PMC-Patients",
                "id": str(pmc_row.get("patient_uid") or pmc_row.get("patient_id") or f"pmc_context_{offset + 1:04d}"),
                "role": "auxiliary_context",
            },
        ]
        flags = case.setdefault("data_quality_flags", {})
        flags["source_dataset"] = spec.name
        flags["source_type"] = spec.source_type
        flags["source_id"] = source_id
        flags["source_url"] = spec.url
        flags["task_profile"] = task_profile_for_source(spec.name)
        flags["diagnosis_metric_applicable"] = bool(pmoa_like.get("_diagnosis_metric_applicable", True))
        flags["no_leak_protocol"] = True
        if style_policy:
            flags["style_policy"] = style_policy
        case["task_profile"] = flags["task_profile"]
        cases.append(case)
    return cases


def gold_labels_for_style_learning(source_name: str, rows: list[dict[str, Any]]) -> list[str]:
    spec = MEDICAL_DATASET_SPECS[source_name]
    labels: list[str] = []
    for idx, row in enumerate(rows, start=1):
        if source_name == "pmoa_tts":
            raw = row.get("diagnoses") or row.get("diagnosis") or []
            labels.extend([str(item) for item in raw] if isinstance(raw, list) else [str(raw)])
        elif source_name == "pmc_patients":
            labels.extend(pmc_row_to_pmoa_like(row, spec, idx).get("diagnoses") or [])
        else:
            labels.extend(generic_row_to_pmoa_like(row, spec, idx).get("diagnoses") or [])
    return [item for item in labels if str(item or "").strip()]


def normalized_leak_text(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def visible_case_payload_for_leakage(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "events": [{"type": event.get("type"), "text": event.get("text")} for event in case.get("events", [])],
        "memory_seed": case.get("memory_seed", []),
        "poison_records": [
            {
                "text": poison.get("text"),
                "supporting_evidence": poison.get("supporting_evidence", []),
            }
            for poison in case.get("poison_records", [])
        ],
        "counterfactuals": [
            {"intervention": item.get("intervention")}
            for item in case.get("counterfactuals", [])
        ],
    }


def leakage_findings_for_case(case: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    visible_json = json.dumps(visible_case_payload_for_leakage(case), ensure_ascii=False)
    visible_norm = normalized_leak_text(visible_json)
    marker_text = visible_json.lower()
    for prefix in GOLD_ONLY_EVIDENCE_PREFIXES:
        if prefix.lower() in marker_text:
            findings.append({"case_id": str(case.get("case_id")), "kind": "gold_marker_visible", "value": prefix})
    labels = case.get("labels") or {}
    aliases = [labels.get("primary_diagnosis")] + list(labels.get("label_aliases") or [])
    flags = case.get("data_quality_flags") or {}
    source_type = str(flags.get("source_type") or "").strip()
    strict_alias_sources = {"medical_mcqa", "medical_instruction", "medical_dialogue", "medical_dialogue_summary"}
    if source_type not in strict_alias_sources:
        return findings
    option_texts = {normalized_leak_text(item.get("text")) for item in case.get("answer_options") or []}
    option_labels = {normalized_leak_text(item.get("label")) for item in case.get("answer_options") or []}
    question_or_title_text = " ".join(
        str(event.get("text") or "")
        for event in case.get("events", [])
        if re.search(r"\b(?:Clinical question or presentation|Initial clinical task/source title)\b", str(event.get("text") or ""), flags=re.I)
    )
    question_or_title_norm = normalized_leak_text(question_or_title_text)
    for alias in aliases:
        alias_text = str(alias or "").strip()
        alias_norm = normalized_leak_text(alias_text)
        if len(alias_norm) < 4:
            continue
        allowed_option = alias_norm in option_texts or alias_norm in option_labels
        if allowed_option and (case.get("data_quality_flags") or {}).get("source_type") == "medical_mcqa":
            continue
        if source_type == "medical_instruction" and alias_norm in question_or_title_norm:
            continue
        if alias_norm in visible_norm:
            findings.append({"case_id": str(case.get("case_id")), "kind": "gold_alias_visible", "value": alias_text[:120]})
    return findings


def validate_medical_source_cases(source_name: str, cases: list[dict[str, Any]]) -> None:
    if not cases:
        raise RuntimeError(f"{source_name}: no cases were built.")
    bad_labels = []
    placeholder_answers = []
    missing_profile = []
    leakage_findings = []
    for case in cases:
        flags = case.get("data_quality_flags") or {}
        profile = str(case.get("task_profile") or flags.get("task_profile") or "").strip()
        if not profile:
            missing_profile.append(case.get("case_id"))
        primary = str((case.get("labels") or {}).get("primary_diagnosis") or "")
        diagnosis_applicable = (case.get("data_quality_flags") or {}).get("diagnosis_metric_applicable") is not False
        if is_bad_label(primary) and (diagnosis_applicable or PLACEHOLDER_LABEL_RE.search(primary)):
            bad_labels.append({"case_id": case.get("case_id"), "primary_diagnosis": primary[:160]})
        for qa in case.get("qa_tasks", []):
            answer = str(qa.get("answer") or "")
            if re.match(r"^\s*Dataset:\s*", answer, flags=re.I):
                placeholder_answers.append({"case_id": case.get("case_id"), "qa_id": qa.get("qa_id"), "answer": answer})
        leakage_findings.extend(
            finding for finding in leakage_findings_for_case(case) if finding.get("kind") == "gold_marker_visible"
        )
    messages = []
    if missing_profile:
        messages.append(f"missing task_profile: {missing_profile[:5]}")
    if placeholder_answers:
        messages.append(f"placeholder qa answers: {placeholder_answers[:5]}")
    if bad_labels:
        messages.append(f"bad primary labels: {bad_labels[:5]}")
    if leakage_findings:
        messages.append(f"gold leakage in visible runtime fields: {leakage_findings[:5]}")
    if messages:
        raise RuntimeError(f"{source_name} data-quality gate failed; " + " | ".join(messages))


def renumber_source_cases(source_name: str, cases: list[dict[str, Any]]) -> None:
    for offset, case in enumerate(cases):
        case["case_id"] = f"{source_name}_{offset + 1:04d}"
        for qa in case.get("qa_tasks", []):
            qa_type = str(qa.get("type") or "qa").lower()
            qa["qa_id"] = f"{case['case_id']}_{qa_type}"


def build_medical_ehr_pool(
    *,
    per_source_n: int,
    output_dir: str | Path = "data/processed/medical_ehr_pool",
    sources: list[str] | None = None,
    require_real_data: bool = False,
    cache_dir: str | Path | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    source_names = sources or list(DEFAULT_MEDICAL_POOL_SOURCES)
    output = ensure_dir(output_dir)
    notes: list[str] = [
        "medical_ehr_pool build",
        f"per_source_n={per_source_n}",
        f"sources={','.join(source_names)}",
    ]
    if random_seed is not None:
        notes.append(f"random_seed={random_seed}")
    strict_no_leak_filter = str(os.getenv("MEDICAL_STRICT_NO_LEAK_FILTER", "0")).lower() in {"1", "true", "yes"}
    candidate_n = per_source_n
    if strict_no_leak_filter:
        candidate_n = int(os.getenv("MEDICAL_STRICT_NO_LEAK_CANDIDATE_N", str(max(per_source_n + 100, per_source_n * 2))))
        notes.append(f"strict_no_leak_filter=true candidate_n={candidate_n}")
    pmc_context = rows_for_medical_source(
        "pmc_patients",
        per_source_n,
        cache_dir=cache_dir,
        require_real_data=require_real_data,
        notes=notes,
        random_seed=random_seed,
    )
    all_cases: list[dict[str, Any]] = []
    per_source_counts: dict[str, int] = {}
    per_source_stats: dict[str, dict[str, int]] = {}
    style_policies: dict[str, dict[str, Any]] = {}
    start_idx = 1
    for source_name in source_names:
        source_load_stats: dict[str, int] = {}
        rows, style_rows = rows_for_medical_source_split(
            source_name,
            candidate_n,
            cache_dir=cache_dir,
            require_real_data=require_real_data,
            notes=notes,
            random_seed=random_seed,
            style_n=min(max(per_source_n, 50), 500),
            stats=source_load_stats,
        )
        raw_loaded_count = int(source_load_stats.get("raw_loaded_count", len(rows)))
        deduped_candidate_count = int(source_load_stats.get("deduped_candidate_count", len(rows)))
        style_policy = learn_source_style_policy(
            source_name,
            task_profile_for_source(source_name),
            gold_labels_for_style_learning(source_name, style_rows),
            train_dev_count=len(style_rows),
        )
        style_policies[source_name] = style_policy
        cases = source_rows_to_cases(source_name, rows, pmc_context, start_idx=start_idx, style_policy=style_policy)
        built_count = len(cases)
        cases, case_duplicate_count = dedupe_cases_by_source_id(cases)
        if case_duplicate_count:
            notes.append(
                f"{source_name}: case source_id dedupe dropped={case_duplicate_count} "
                f"kept={len(cases)} from built={built_count}."
            )
        if strict_no_leak_filter:
            before = len(cases)
            clean_cases = [case for case in cases if not leakage_findings_for_case(case)]
            dropped = before - len(clean_cases)
            notes.append(f"{source_name}: strict no-leak filter dropped={dropped} kept={len(clean_cases)} from candidates={before}.")
            clean_cases, post_filter_duplicate_count = dedupe_cases_by_source_id(clean_cases)
            if post_filter_duplicate_count:
                notes.append(
                    f"{source_name}: post-filter source_id dedupe dropped={post_filter_duplicate_count} "
                    f"kept={len(clean_cases)}."
                )
            post_filter_count = len(clean_cases)
            cases = clean_cases[:per_source_n]
            if len(cases) < per_source_n:
                raise RuntimeError(
                    f"{source_name}: strict no-leak filter and dedupe kept only {len(cases)} clean unique cases "
                    f"from {before} candidates; need {per_source_n}."
                )
            renumber_source_cases(source_name, cases)
        else:
            post_filter_count = len(cases)
            cases = cases[:per_source_n]
            if require_real_data and len(cases) < per_source_n:
                raise RuntimeError(
                    f"{source_name}: source row/case dedupe kept only {len(cases)} unique cases; need {per_source_n}."
                )
        validate_medical_source_cases(source_name, cases)
        start_idx += len(cases)
        per_source_counts[source_name] = len(cases)
        per_source_stats[source_name] = {
            "raw_loaded_count": raw_loaded_count,
            "deduped_candidate_count": deduped_candidate_count,
            "post_filter_count": post_filter_count,
            "selected_count": len(cases),
        }
        write_jsonl(output / f"{source_name}_{len(cases)}.jsonl", cases)
        write_prefix_slices(cases, output_dir=output / "slices" / source_name, stem=source_name, sizes=list(DEFAULT_PREFIX_SIZES))
        all_cases.extend(cases)
    pooled_path = output / f"medical_ehr_pool_{len(all_cases)}.jsonl"
    write_jsonl(pooled_path, all_cases)
    pooled_sizes = [len(source_names) * size for size in DEFAULT_PREFIX_SIZES if len(source_names) * size <= len(all_cases)]
    write_prefix_slices(all_cases, output_dir=output / "slices" / "pooled", stem="medical_ehr_pool", sizes=pooled_sizes)
    manifest = {
        "track": "medical_ehr_pool",
        "per_source_n": per_source_n,
        "random_seed": random_seed,
        "pooled_path": str(pooled_path),
        "sources": {
            name: {
                "count": per_source_counts.get(name, 0),
                "source_type": MEDICAL_DATASET_SPECS[name].source_type,
                "url": MEDICAL_DATASET_SPECS[name].url,
                "style_policy": style_policies.get(name, {}),
                **per_source_stats.get(name, {}),
            }
            for name in source_names
        },
        "no_leak_protocol": True,
        "style_policies": style_policies,
        "pooled_prefix_sizes": pooled_sizes,
    }
    write_text(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_text(output / "build.notes.txt", "\n".join(notes) + "\n")
    return manifest
