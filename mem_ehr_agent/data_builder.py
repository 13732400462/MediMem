from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .data_sources import paired_source_rows
from .io_utils import write_jsonl, write_text
from .schemas import validate_case
from .task_profiles import DEFAULT_TASK_PROFILE, LONGITUDINAL_DIAGNOSIS


def _as_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("textual_timeseries") or row.get("events") or []
    events = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            text = str(item.get("event") or item.get("text") or "").strip()
            time = item.get("time", idx)
        else:
            text = str(item).strip()
            time = idx
        if not text:
            continue
        try:
            time_i = int(float(time))
        except (TypeError, ValueError):
            time_i = idx
        events.append(
            {
                "event_id": f"ev_{idx:03d}",
                "time": time_i,
                "text": text,
                "type": infer_event_type(text),
            }
        )
    if not events:
        patient_text = str(row.get("patient") or row.get("summary") or row.get("title") or "clinical case")
        chunks = [c.strip() for c in re.split(r"(?<=[.;])\s+", patient_text) if c.strip()]
        for idx, text in enumerate(chunks[:8] or [patient_text]):
            events.append({"event_id": f"ev_{idx:03d}", "time": idx * 30, "text": text, "type": infer_event_type(text)})
    events.sort(key=lambda x: (x["time"], x["event_id"]))
    return events


def infer_event_type(text: str) -> str:
    low = text.lower()
    diagnosis_terms = [
        "diagnos",
        "confirmed",
        "pathology",
        "carcinoma",
        "cancer",
        "leukemia",
        "leukaemia",
        "syndrome",
        "disease",
        "asthma",
        "herniation",
        "lupus",
        "sle",
        "nxg",
        "rdd",
        "echinococ",
        "injury",
    ]
    if any(x in low for x in diagnosis_terms):
        return "diagnosis"
    if any(x in low for x in ["ct", "mri", "scan", "x-ray", "imaging"]):
        return "imaging"
    if any(x in low for x in ["treated", "therapy", "surgery", "antibiotic", "drug"]):
        return "treatment"
    if any(x in low for x in ["lab", "creatinine", "hemoglobin", "platelet", "blast"]):
        return "lab"
    return "clinical"


def _diagnoses(row: dict[str, Any]) -> list[str]:
    raw = row.get("diagnoses") or row.get("diagnosis") or []
    if isinstance(raw, str):
        parts = [p.strip() for p in re.split(r"[,;/|]", raw) if p.strip()]
    elif isinstance(raw, list):
        parts = [str(p).strip() for p in raw if str(p).strip()]
    else:
        parts = []
    title = str(row.get("title") or "")
    if not parts and title:
        match = re.search(r"(?:case report of|case report:)\s+(.+)", title, flags=re.I)
        if match:
            parts.append(match.group(1)[:120])
    return dedupe(parts) or ["Unspecified complex clinical diagnosis"]


def dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        norm = normalize_text(item)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(item)
    return out


def as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _encounters(events: list[dict[str, Any]], labs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []
    groups: list[list[dict[str, Any]]] = []
    for event in events:
        if not groups or len(groups[-1]) >= max(1, len(events) // 3):
            groups.append([])
        groups[-1].append(event)
    encounters = []
    for idx, group in enumerate(groups[:5]):
        time = group[0]["time"]
        lab_subset = [lab for lab in labs if abs(int(lab["time"]) - int(time)) <= 60]
        summary = "; ".join(e["text"] for e in group[:3])
        encounters.append(
            {
                "encounter_id": f"enc_{idx:02d}",
                "time": time,
                "summary": summary,
                "event_ids": [e["event_id"] for e in group],
                "labs": lab_subset,
            }
        )
    return encounters


def _synthetic_labs(diagnoses: list[str], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = " ".join(diagnoses + [e["text"] for e in events]).lower()
    base_time = events[0]["time"] if events else 0
    labs: list[dict[str, Any]] = []
    if any(k in text for k in ["renal", "kidney", "septic", "urinary"]):
        labs += [
            {"time": base_time, "name": "Creatinine", "value": 3.1, "unit": "mg/dL", "flag": "high"},
            {"time": base_time + 14, "name": "Creatinine", "value": 1.9, "unit": "mg/dL", "flag": "improved"},
            {"time": base_time, "name": "White blood cell count", "value": 18.4, "unit": "10^9/L", "flag": "high"},
        ]
    if any(k in text for k in ["leukemia", "aml", "blast", "myeloid"]):
        labs += [
            {"time": base_time, "name": "Hemoglobin", "value": 8.2, "unit": "g/dL", "flag": "low"},
            {"time": base_time, "name": "Bone marrow blasts", "value": 50, "unit": "%", "flag": "high"},
        ]
    if any(k in text for k in ["lung", "cancer", "metastatic", "malign"]):
        labs += [
            {"time": base_time, "name": "LDH", "value": 380, "unit": "U/L", "flag": "high"},
            {"time": base_time + 7, "name": "CEA", "value": 14.2, "unit": "ng/mL", "flag": "high"},
        ]
    if not labs:
        labs += [
            {"time": base_time, "name": "C-reactive protein", "value": 22, "unit": "mg/L", "flag": "high"},
            {"time": base_time + 7, "name": "C-reactive protein", "value": 8, "unit": "mg/L", "flag": "improved"},
        ]
    return labs


def _early_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    return events[0] if events else {"event_id": "ev_000", "time": 0, "text": "Initial clinical presentation.", "type": "clinical"}


def _superseding_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    diagnosis_events = [event for event in events if event.get("type") == "diagnosis"]
    return diagnosis_events[-1] if diagnosis_events else (events[-1] if events else _early_event(events))


def _supporting_fact(event: dict[str, Any], fallback: str) -> str:
    text = str(event.get("text") or "").strip()
    if not text:
        return fallback
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned[:180].strip(" .,:;")


def redact_label_mentions(text: str, diagnoses: list[str]) -> str:
    redacted = str(text)
    stop_words = {"with", "without", "after", "before", "from", "into", "test", "screening", "diagnosed", "diagnosis"}
    candidates: list[str] = []
    for diagnosis in diagnoses:
        diag = str(diagnosis or "").strip()
        if not diag:
            continue
        candidates.append(diag)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]*", diag):
            if token.lower() in stop_words:
                continue
            if len(token) >= 4 or token.isupper():
                candidates.append(token)
    for candidate in sorted(set(candidates), key=len, reverse=True):
        redacted = re.sub(re.escape(candidate), "[diagnosis-redacted]", redacted, flags=re.I)
    return redacted


def build_stale_memory_artifacts(
    *,
    idx: int,
    primary: str,
    events: list[dict[str, Any]],
    labs: list[dict[str, Any]],
    pmc_row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    early = _early_event(events)
    final = _superseding_event(events)
    old_hypothesis = "viral syndrome" if "viral" not in primary.lower() else "nonspecific inflammatory illness"
    early_fact = _supporting_fact(early, "early symptoms and inflammatory markers")
    final_fact = _supporting_fact(final, f"later evidence supported {primary}")
    runtime_early_fact = redact_label_mentions(early_fact, [primary])
    runtime_final_fact = redact_label_mentions(final_fact, [primary])
    first_lab = labs[0] if labs else {"time": early.get("time", 0), "name": "C-reactive protein", "value": 22, "unit": "mg/L", "flag": "high"}
    later_lab = labs[-1] if labs else first_lab
    similar_context = str(pmc_row.get("patient") or pmc_row.get("title") or "A similar case context").strip()
    similar_snippet = re.sub(r"\s+", " ", similar_context)[:160].strip(" .,:;") or "A similar case context"
    early_time = int(early.get("time", 0) or 0)
    final_time = int(final.get("time", early_time) or early_time)

    memory_seed = [
        {
            "summary": f"Early working impression: {old_hypothesis} was plausible because {runtime_early_fact}.",
            "time_scope": {"start": early_time, "end": early_time},
            "valid_time_scope": {"start": early_time, "end": final_time},
            "status": "active",
            "confidence": 0.35,
            "tags": ["initial_hypothesis", "stale_candidate"],
            "claim_type": "interpretation",
            "should_preserve_fact": True,
            "source_event_id": early.get("event_id"),
        }
    ]
    if labs:
        memory_seed.append(
            {
                "summary": (
                    f"Early lab state: {first_lab.get('name')} was {first_lab.get('value')} "
                    f"{first_lab.get('unit')} ({first_lab.get('flag')})."
                ),
                "time_scope": {"start": first_lab.get("time"), "end": first_lab.get("time")},
                "valid_time_scope": {"start": first_lab.get("time"), "end": later_lab.get("time")},
                "status": "active",
                "confidence": 0.55,
                "tags": ["early_lab", "stale_candidate"],
                "claim_type": "fact",
                "should_preserve_fact": True,
                "source_event_id": early.get("event_id"),
            }
        )

    private_poison_specs = [
        {
            "poison_id": f"poison_{idx:04d}_initial_interpretation",
            "text": f"Early working impression: {old_hypothesis} was plausible because {runtime_early_fact}.",
            "source_event_id": early.get("event_id"),
            "source_time": early_time,
            "supporting_evidence": [runtime_early_fact],
            "superseded_by_event_id": final.get("event_id"),
            "superseding_evidence": final_fact,
            "valid_time_scope": {"start": early_time, "end": final_time},
            "status": "active",
            "claim_type": "interpretation",
            "should_preserve_fact": True,
            "preserved_facts": [runtime_early_fact],
            "revised_claim": f"Early {old_hypothesis} impression is time-limited and superseded by later evidence for {primary}.",
            "pollution_type": "outdated_initial_diagnosis",
            "staleness_type": "superseded_by_later_diagnosis",
            "expected_op": "Revise",
        },
        {
            "poison_id": f"poison_{idx:04d}_partial_truth_misleading",
            "text": (
                f"Partially true memory: {runtime_final_fact}. Misleading conclusion: this evidence is being used "
                f"to keep the earlier {old_hypothesis} impression active."
            ),
            "source_event_id": final.get("event_id"),
            "source_time": final_time,
            "supporting_evidence": [runtime_final_fact],
            "superseded_by_event_id": final.get("event_id"),
            "superseding_evidence": final_fact,
            "valid_time_scope": {"start": final_time, "end": final_time},
            "status": "active",
            "claim_type": "mixed_fact_interpretation",
            "should_preserve_fact": True,
            "preserved_facts": [runtime_final_fact],
            "revised_claim": f"{final_fact} should be preserved, but it supports {primary} rather than {old_hypothesis}.",
            "pollution_type": "partial_truth_misleading_memory",
            "staleness_type": "true_fact_wrong_interpretation",
            "expected_op": "Revise",
        },
        {
            "poison_id": f"poison_{idx:04d}_severity_or_lab",
            "text": (
                f"Early status appeared stable with {first_lab.get('name')}={first_lab.get('value')} "
                f"{first_lab.get('unit')} ({first_lab.get('flag')}); this early state can guide later decisions."
            ),
            "source_event_id": early.get("event_id"),
            "source_time": int(first_lab.get("time", early_time) or early_time),
            "supporting_evidence": [
                f"{first_lab.get('name')}={first_lab.get('value')} {first_lab.get('unit')} ({first_lab.get('flag')})"
            ],
            "superseded_by_event_id": final.get("event_id"),
            "superseding_evidence": final_fact,
            "valid_time_scope": {"start": first_lab.get("time"), "end": final_time},
            "status": "active",
            "claim_type": "severity",
            "should_preserve_fact": True,
            "preserved_facts": [
                f"{first_lab.get('name')} was {first_lab.get('value')} {first_lab.get('unit')} early"
            ],
            "revised_claim": "Early severity/lab interpretation should not be treated as the current state after later diagnostic evidence.",
            "pollution_type": "outdated_severity_or_lab_state",
            "staleness_type": "invalidated_by_later_course",
            "expected_op": "Invalidate",
        },
        {
            "poison_id": f"poison_{idx:04d}_similar_case_transfer",
            "text": f"Similar case memory: {similar_snippet}; transfer this prior case context to the current patient.",
            "source_event_id": str(pmc_row.get("patient_uid") or pmc_row.get("patient_id") or f"pmc_{idx:04d}"),
            "source_time": early_time,
            "supporting_evidence": [similar_snippet],
            "superseded_by_event_id": final.get("event_id"),
            "superseding_evidence": "This memory comes from a different or only loosely similar patient context.",
            "valid_time_scope": {"start": early_time, "end": early_time},
            "status": "active",
            "claim_type": "interpretation",
            "should_preserve_fact": False,
            "preserved_facts": [],
            "revised_claim": "",
            "pollution_type": "similar_case_mistransfer",
            "staleness_type": "cross_patient_context_mismatch",
            "expected_op": "Discard",
        },
    ]
    if idx % 2 == 0:
        private_poison_specs.append(
            {
                "poison_id": f"poison_{idx:04d}_valid_historical_fact",
                "text": f"Historical fact memory: {runtime_early_fact}. Keep as source evidence, but do not over-weight it.",
                "source_event_id": early.get("event_id"),
                "source_time": early_time,
                "supporting_evidence": [runtime_early_fact],
                "superseded_by_event_id": None,
                "superseding_evidence": "",
                "valid_time_scope": {"start": early_time, "end": final_time},
                "status": "active",
                "claim_type": "fact",
                "should_preserve_fact": True,
                "preserved_facts": [runtime_early_fact],
                "revised_claim": "",
                "pollution_type": "valid_historical_fact",
                "staleness_type": "valid_but_low_priority",
                "expected_op": "Keep",
            }
        )
    if idx % 3 == 0:
        private_poison_specs.append(
            {
                "poison_id": f"poison_{idx:04d}_ambiguous_low_confidence",
                "text": (
                    f"Ambiguous memory: {runtime_early_fact}; possible relation to the later course is uncertain "
                    "and should be reviewed before use."
                ),
                "source_event_id": early.get("event_id"),
                "source_time": early_time,
                "supporting_evidence": [runtime_early_fact],
                "superseded_by_event_id": final.get("event_id"),
                "superseding_evidence": final_fact,
                "valid_time_scope": {"start": early_time, "end": final_time},
                "status": "active",
                "claim_type": "ambiguous_fact",
                "should_preserve_fact": True,
                "preserved_facts": [runtime_early_fact],
                "revised_claim": "",
                "pollution_type": "ambiguous_low_confidence_memory",
                "staleness_type": "uncertain_relevance",
                "expected_op": "Flag",
            }
        )
    if idx % 5 == 0:
        private_poison_specs.append(
            {
                "poison_id": f"poison_{idx:04d}_partial_transfer",
                "text": (
                    f"Similar case memory: {similar_snippet}; preserve any general context but do not treat "
                    "the prior patient as the current patient."
                ),
                "source_event_id": str(pmc_row.get("patient_uid") or pmc_row.get("patient_id") or f"pmc_{idx:04d}"),
                "source_time": early_time,
                "supporting_evidence": [similar_snippet],
                "superseded_by_event_id": final.get("event_id"),
                "superseding_evidence": "The source is a different patient and only transferable as general context.",
                "valid_time_scope": {"start": early_time, "end": early_time},
                "status": "active",
                "claim_type": "mixed_transfer",
                "should_preserve_fact": True,
                "preserved_facts": [similar_snippet],
                "revised_claim": "Only general context may be retained; cross-patient details must not drive current diagnosis.",
                "pollution_type": "similar_case_partial_transfer",
                "staleness_type": "partial_cross_patient_transfer",
                "expected_op": "Revise",
            }
        )
    runtime_keys = {
        "poison_id",
        "text",
        "source_event_id",
        "source_time",
        "supporting_evidence",
        "valid_time_scope",
        "status",
        "claim_type",
        "pollution_type",
        "staleness_type",
    }
    poison_records = [{key: poison[key] for key in runtime_keys if key in poison} for poison in private_poison_specs]
    expected_ops = [
        {
            "op": poison["expected_op"],
            "target": poison["poison_id"],
            "reason": f"{poison['pollution_type']} should be handled as {poison['expected_op']}.",
            "preserved_facts": poison.get("preserved_facts", []),
            "revised_claim": poison.get("revised_claim", ""),
            "should_preserve_fact": poison.get("should_preserve_fact", False),
            "quality_checks": {
                "preserve_true_facts": bool(poison.get("should_preserve_fact")),
                "remove_wrong_interpretation": poison["expected_op"] == "Revise",
                "respect_time_scope": poison["expected_op"] in {"Revise", "Invalidate", "Flag"},
                "avoid_diagnosis_leakage": True,
            },
        }
        for poison in private_poison_specs
    ]
    expected_ops.append(
        {
            "op": "Write",
            "target": final.get("event_id"),
            "reason": "Confirmed or late diagnostic evidence should be stored as active memory.",
        }
    )
    return memory_seed, poison_records, expected_ops


def build_case(pmoa_row: dict[str, Any], pmc_row: dict[str, Any], idx: int) -> dict[str, Any]:
    events = _as_events(pmoa_row)
    diagnoses = _diagnoses(pmoa_row)
    labs = _synthetic_labs(diagnoses, events)
    task_profile = str(pmoa_row.get("_task_profile") or DEFAULT_TASK_PROFILE)
    if pmoa_row.get("_prefer_explicit_primary"):
        primary = diagnoses[0] if diagnoses else "Medical answer entity"
    else:
        primary = choose_primary_diagnosis(events, diagnoses) if task_profile == LONGITUDINAL_DIAGNOSIS else (diagnoses[0] if diagnoses else "Medical answer entity")
    label_aliases = as_text_list(pmoa_row.get("_label_aliases")) or [primary]
    if primary and not any(normalize_text(primary) == normalize_text(d) for d in diagnoses):
        diagnoses = [primary] + diagnoses
    elif primary:
        diagnoses = [primary] + [d for d in diagnoses if normalize_text(d) != normalize_text(primary)]
    key_event = _superseding_event(events)
    memory_seed, poison_records, expected_memory_ops = build_stale_memory_artifacts(
        idx=idx,
        primary=primary,
        events=events,
        labs=labs,
        pmc_row=pmc_row,
    )
    case_id = f"case_{idx:04d}"
    case = {
        "case_id": case_id,
        "source_refs": [
            {
                "dataset": "PMOA-TTS",
                "id": str(pmoa_row.get("case_report_id") or pmoa_row.get("pmc_id") or case_id),
            },
            {
                "dataset": "PMC-Patients",
                "id": str(pmc_row.get("patient_uid") or pmc_row.get("patient_id") or f"pmc_{idx:04d}"),
            },
        ],
        "demographics": pmoa_row.get("demographics") or {
            "age": pmoa_row.get("age", "Not Specified"),
            "sex": pmoa_row.get("sex") or pmoa_row.get("gender") or "Not Specified",
            "ethnicity": "Not Specified",
        },
        "events": events,
        "synthetic_labs": labs,
        "encounters": _encounters(events, labs),
        "memory_seed": memory_seed,
        "poison_records": poison_records,
        "labels": {
            "primary_diagnosis": primary,
            "diagnosis_list": diagnoses,
            "label_aliases": label_aliases,
            "outcome": pmoa_row.get("death_info", {}),
        },
        "answer_options": pmoa_row.get("_answer_options") or [],
        "task_profile": task_profile,
        "qa_tasks": [
            {
                "qa_id": f"{case_id}_idr",
                "type": "IDR",
                "question": f"What happened around time {key_event['time']}?",
                "answer": key_event["text"],
            },
            {
                "qa_id": f"{case_id}_cdr",
                "type": "CDR",
                "question": "Which final diagnosis best explains the longitudinal timeline?",
                "answer": primary,
            },
            {
                "qa_id": f"{case_id}_sr",
                "type": "SR",
                "question": "What is the most likely primary diagnosis after all encounters?",
                "answer": primary,
            },
        ],
        "expected_memory_ops": expected_memory_ops,
        "counterfactuals": [
            {
                "intervention": f"Remove or negate this evidence: {key_event['text']}",
                "target_diagnosis": primary,
                "expected_effect": "confidence_drop",
                "minimum_drop": 0.40,
            }
        ],
        "pmc_context": str(pmc_row.get("patient") or pmc_row.get("title") or "")[:1200],
        "data_quality_flags": {
            "pmoa_real_source": str(pmoa_row.get("pmc_id") or "").lower() != "fallback",
            "pmc_real_source": not str(pmc_row.get("patient_uid") or "").startswith("fallback"),
            "event_count": len(events),
            "diagnosis_event_count": sum(1 for event in events if event.get("type") == "diagnosis"),
            "has_follow_up": any(int(event.get("time", 0) or 0) > 0 for event in events),
            "label_fragment_like": is_fragment_like_label(primary),
            "species_context": infer_species_context(pmoa_row, events),
            "task_profile": task_profile,
            "diagnosis_metric_applicable": bool(pmoa_row.get("_diagnosis_metric_applicable", True)),
        },
    }
    return case


def infer_species_context(row: dict[str, Any], events: list[dict[str, Any]]) -> str:
    text = " ".join(
        [
            str(row.get("species") or ""),
            str(row.get("title") or ""),
            str(row.get("patient") or ""),
            " ".join(str(event.get("text") or "") for event in events[:20]),
        ]
    ).lower()
    species_terms = {
        "dog": ("dog", "canine"),
        "cat": ("cat", "feline"),
        "horse": ("horse", "equine"),
        "cow": ("cow", "bovine", "cattle"),
        "sheep": ("sheep", "ovine"),
        "goat": ("goat", "caprine"),
        "pig": ("pig", "porcine", "swine"),
        "mouse": ("mouse", "murine"),
        "rat": ("rat",),
    }
    for species, terms in species_terms.items():
        if any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms):
            return species
    return "human"


def is_fragment_like_label(label: str) -> bool:
    low = normalize_text(label)
    if not low:
        return True
    diagnosis_terms = {
        "cancer",
        "carcinoma",
        "disease",
        "syndrome",
        "infection",
        "injury",
        "fracture",
        "metastasis",
        "metastases",
        "pneumonia",
        "anemia",
        "leukemia",
        "lymphoma",
        "tumor",
        "tumour",
        "hernia",
        "meningitis",
        "diabetes",
        "lupus",
        "leiomyoma",
        "osteomalacia",
        "hyperplasia",
    }
    fragment_markers = {
        "by",
        "associated",
        "history",
        "confirmed",
        "suspected",
        "screening",
        "elevated",
        "personal",
    }
    tokens = low.split()
    if len(tokens) <= 2 and not any(term in low for term in diagnosis_terms):
        return True
    if tokens and tokens[0] in fragment_markers and not any(term in low for term in diagnosis_terms):
        return True
    if " no " in f" {low} " and "history" in low:
        return True
    return False


def choose_primary_diagnosis(events: list[dict[str, Any]], diagnoses: list[str]) -> str:
    joined_norm = normalize_text(" ".join(e["text"] for e in events))
    if "e granulosus" in joined_norm or "echinococ" in joined_norm:
        return "Echinococcosis"
    explicit_patterns = [
        r"(?:diagnosed with|diagnosed as|diagnosis of|confirmed)\s+([^.;]+)",
        r"pathology (?:diagnosed|confirmed|showed)\s+([^.;]+)",
        r"\b([A-Z][A-Za-z -]+ carcinoma)\b",
        r"\b([A-Z][A-Za-z -]+ cancer)\b",
    ]
    for event in events:
        text = str(event.get("text", ""))
        for pattern in explicit_patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                candidate = cleanup_primary(match.group(1))
                if candidate:
                    return canonical_diagnosis(candidate)
    joined = " ".join(e["text"] for e in events).lower()
    for diagnosis in diagnoses:
        if normalize_text(diagnosis) and normalize_text(diagnosis) in normalize_text(joined):
            return canonical_diagnosis(diagnosis)
    # Disease-specific cues when case reports list a background condition before the actual target.
    cue_map = [
        ("invasive ductal carcinoma", "Breast cancer"),
        ("mastectomy", "Breast cancer"),
        ("i-131", "Papillary thyroid cancer"),
        ("thyroidectomy", "Papillary thyroid cancer"),
        ("small bowel herniation", "Small bowel herniation"),
        ("systemic lupus", "Systemic lupus erythematosus"),
        ("sle", "Systemic lupus erythematosus"),
    ]
    for cue, diagnosis in cue_map:
        if cue in joined:
            return diagnosis
    return canonical_diagnosis(diagnoses[0] if diagnoses else "Unspecified complex clinical diagnosis")


def cleanup_primary(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,:;()[]")
    text = re.sub(r"\b(with|after|following|indicating|and underwent)\b.*$", "", text, flags=re.I).strip(" .,:;")
    return text[:120].strip()


def canonical_diagnosis(text: str) -> str:
    low = normalize_text(text)
    mapping = {
        "sle": "Systemic lupus erythematosus",
        "nxg": "Necrobiotic xanthogranuloma",
        "extra nodal rdd": "Rosai-Dorfman-Destombes disease",
        "rdd": "Rosai-Dorfman-Destombes disease",
        "e granulosus": "Echinococcosis",
        "invasive ductal carcinoma": "Breast cancer",
        "bilateral poland syndrome": "Poland syndrome",
    }
    for key, value in mapping.items():
        if key == low or key in low:
            return value
    return text


def build_dataset(
    n: int,
    output_path: str,
    notes_path: str | None = None,
    *,
    require_real_data: bool = False,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    pmoa_rows, pmc_rows, notes = paired_source_rows(n, require_real_data=require_real_data, cache_dir=cache_dir)
    cases = [build_case(pmoa_rows[i], pmc_rows[i], i + 1) for i in range(n)]
    errors = {case["case_id"]: validate_case(case) for case in cases}
    failed = {case_id: errs for case_id, errs in errors.items() if errs}
    if failed:
        raise ValueError(f"Generated invalid cases: {failed}")
    write_jsonl(output_path, cases)
    if notes_path:
        distribution: dict[str, dict[str, int]] = {}
        for case in cases:
            poison_by_id = {str(p.get("poison_id")): p for p in case.get("poison_records", [])}
            for expected in case.get("expected_memory_ops", []):
                op = str(expected.get("op") or "")
                if op == "Write":
                    continue
                poison = poison_by_id.get(str(expected.get("target")), {})
                pollution_type = str(poison.get("pollution_type") or "unknown")
                distribution.setdefault(pollution_type, {})
                distribution[pollution_type][op] = distribution[pollution_type].get(op, 0) + 1
        audit_lines = ["expected_memory_ops distribution:"]
        for pollution_type, counts in sorted(distribution.items()):
            summary = ", ".join(f"{op}={count}" for op, count in sorted(counts.items()))
            audit_lines.append(f"- {pollution_type}: {summary}")
        write_text(notes_path, "\n".join(notes + audit_lines) + "\n")
    return cases


def write_prefix_slices(
    cases: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    stem: str,
    sizes: list[int],
    notes: str = "",
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for size in sizes:
        if size <= 0:
            continue
        prefix_cases = cases[: min(size, len(cases))]
        path = output / f"{stem}_{size}.jsonl"
        write_jsonl(path, prefix_cases)
        note_lines = [
            notes.strip(),
            f"prefix_size={len(prefix_cases)}",
            f"source_pool_size={len(cases)}",
            "prefix_stable=true",
        ]
        write_text(path.with_suffix(".notes.txt"), "\n".join(line for line in note_lines if line) + "\n")
        written.append(path)
    return written
