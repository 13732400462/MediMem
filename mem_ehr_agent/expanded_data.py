from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_builder import build_case, write_prefix_slices
from .data_sources import fetch_hf_rows, fetch_pmc_patient_rows, fetch_pmoa_rows
from .io_utils import ensure_dir, write_jsonl, write_text
from .task_profiles import task_profile_for_source


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


DEFAULT_MEDICAL_POOL_SOURCES = tuple(MEDICAL_DATASET_SPECS)
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


def text_value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
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
    raw = row.get("cop", row.get("answer_idx", row.get("answer")))
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
    raw = row.get("cop", row.get("answer_idx", row.get("answer")))
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


def diagnosis_from_text(text: str, fallback: str) -> list[str]:
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
            candidates.append(match.group(1).strip())
    if not candidates:
        candidates.append(cleaned[:120])
    return [candidate for candidate in candidates if candidate] or [fallback]


def generic_row_to_pmoa_like(row: dict[str, Any], spec: MedicalDatasetSpec, idx: int) -> dict[str, Any]:
    source_id = stable_source_id(spec.name, row, idx)
    title = text_value(row, "title", "question", "instruction", "input", "chief_complaint") or f"{spec.name} sample {idx}"
    question = text_value(row, "question", "instruction", "input", "dialogue", "dialog")
    context = text_value(row, "context", "patient", "note", "conversation", "dialogue", "dialog")
    answer = option_answer(row)
    options = answer_options(row)
    explanation = text_value(row, "exp", "explanation", "output", "response", "soap_summary", "summary")
    safe_explanation = scrub_answer_explanation(explanation, answer) if spec.source_type == "medical_mcqa" else explanation
    subject = text_value(row, "subject_name", "topic_name", "category", "department")
    diagnosis_seed = answer or explanation or subject or title
    diagnoses = diagnosis_from_text(diagnosis_seed, fallback=subject or "Medical answer entity")

    event_texts = []
    if title:
        event_texts.append(f"Initial clinical task/source title: {title}")
    if context and context != question:
        event_texts.extend(split_sentences(context, limit=4))
    if question:
        event_texts.append(f"Clinical question or presentation: {question}")
    if options:
        option_text = "; ".join(f"{item['label']}. {item['text']}" for item in options)
        event_texts.append(f"Answer options: {option_text}")
    if safe_explanation and safe_explanation != answer:
        event_texts.extend(split_sentences(safe_explanation, limit=3))
    if len(event_texts) < 3:
        event_texts.extend([f"Source type: {spec.source_type}", f"Dataset: {spec.name}"])

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
    }


def pmc_row_to_pmoa_like(row: dict[str, Any], spec: MedicalDatasetSpec, idx: int) -> dict[str, Any]:
    source_id = stable_source_id(spec.name, row, idx)
    patient = text_value(row, "patient", "summary", "title") or f"PMC patient sample {idx}"
    title = text_value(row, "title") or patient[:120]
    diagnoses = diagnosis_from_text(title, fallback="Complex patient summary diagnosis")
    events = split_sentences(patient, limit=10)
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
    }


def rows_for_medical_source(
    source_name: str,
    n: int,
    *,
    cache_dir: str | Path | None = None,
    require_real_data: bool = False,
    notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    notes = notes if notes is not None else []
    spec = MEDICAL_DATASET_SPECS[source_name]
    if source_name == "pmoa_tts":
        rows = fetch_pmoa_rows(n, notes, cache_dir=cache_dir)
    elif source_name == "pmc_patients":
        rows = fetch_pmc_patient_rows(n, notes, cache_dir=cache_dir)
    else:
        if not spec.hf_dataset:
            rows = []
        else:
            rows = fetch_hf_rows(spec.hf_dataset, preferred_splits=list(spec.preferred_splits), n=n, notes=notes)
    if require_real_data and len(rows) < n:
        raise RuntimeError(f"{source_name} requires {n} real rows but only {len(rows)} were loaded.")
    return rows[:n]


def source_rows_to_cases(
    source_name: str,
    rows: list[dict[str, Any]],
    pmc_context_rows: list[dict[str, Any]],
    *,
    start_idx: int = 1,
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
        case["task_profile"] = flags["task_profile"]
        cases.append(case)
    return cases


def build_medical_ehr_pool(
    *,
    per_source_n: int,
    output_dir: str | Path = "data/processed/medical_ehr_pool",
    sources: list[str] | None = None,
    require_real_data: bool = False,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    source_names = sources or list(DEFAULT_MEDICAL_POOL_SOURCES)
    output = ensure_dir(output_dir)
    notes: list[str] = [
        "medical_ehr_pool build",
        f"per_source_n={per_source_n}",
        f"sources={','.join(source_names)}",
    ]
    pmc_context = rows_for_medical_source(
        "pmc_patients",
        per_source_n,
        cache_dir=cache_dir,
        require_real_data=require_real_data,
        notes=notes,
    )
    all_cases: list[dict[str, Any]] = []
    per_source_counts: dict[str, int] = {}
    start_idx = 1
    for source_name in source_names:
        rows = rows_for_medical_source(
            source_name,
            per_source_n,
            cache_dir=cache_dir,
            require_real_data=require_real_data,
            notes=notes,
        )
        cases = source_rows_to_cases(source_name, rows, pmc_context, start_idx=start_idx)
        start_idx += len(cases)
        per_source_counts[source_name] = len(cases)
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
        "pooled_path": str(pooled_path),
        "sources": {
            name: {
                "count": per_source_counts.get(name, 0),
                "source_type": MEDICAL_DATASET_SPECS[name].source_type,
                "url": MEDICAL_DATASET_SPECS[name].url,
            }
            for name in source_names
        },
        "pooled_prefix_sizes": pooled_sizes,
    }
    write_text(output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_text(output / "build.notes.txt", "\n".join(notes) + "\n")
    return manifest
