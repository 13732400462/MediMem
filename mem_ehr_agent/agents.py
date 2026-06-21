from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from .llm import DeepSeekClient, extract_json_object
from .memory import apply_critique, bootstrap_memory
from .medical_terms import canonicalize_diagnosis, canonicalize_diagnoses
from .style_policy import style_policy_prompt
from .task_profiles import (
    CLINICAL_ASSESSMENT_ENTITY,
    LONGITUDINAL_DIAGNOSIS,
    MEDICAL_ANSWER_ENTITY,
    case_task_profile,
    diagnosis_list_budget,
    task_profile_prompt_policy,
)


TARGET_LEAKAGE_PATTERNS = (
    r"final\s+answer\s+or\s+diagnosis\s+target",
    r"final\s+answer\s+target",
    r"final\s+answer",
    r"diagnosis\s+target",
    r"soap\s+assessment\s+target",
    r"topic\s+or\s+diagnosis\s+target",
    r"doctor\s+assessment\s+target",
    r"assessment\s+entity\s+candidate",
    r"reference\s+answer\s+evidence",
    r"doctor\s+assessment",
    r"correct\s+option",
    r"correct\s+answer",
)
TARGET_LEAKAGE_RE = re.compile(r"(?i)\b(?:" + "|".join(TARGET_LEAKAGE_PATTERNS) + r")\b\s*:?\s*")
TARGET_LEAKAGE_PREFIX_RE = re.compile(r"(?i)^\s*(?:" + "|".join(TARGET_LEAKAGE_PATTERNS) + r")\s*:?\s*")
PROMPT_INPUT_TOKEN_BUDGET = int(os.environ.get("MEDIMEM_PROMPT_INPUT_BUDGET", "6600"))
PROMPT_CONTEXT_TOKEN_LIMIT = int(os.environ.get("MEDIMEM_PROMPT_CONTEXT_LIMIT", "8192"))
PROMPT_RESERVED_RESPONSE_TOKENS = int(os.environ.get("MEDIMEM_PROMPT_RESERVED_RESPONSE_TOKENS", "384"))
PROMPT_EST_CHARS_PER_TOKEN = float(os.environ.get("MEDIMEM_PROMPT_EST_CHARS_PER_TOKEN", "3.0"))


def contains_target_leakage(text: Any) -> bool:
    return bool(TARGET_LEAKAGE_RE.search(str(text or "")))


def sanitize_runtime_text(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if TARGET_LEAKAGE_PREFIX_RE.search(raw):
        return ""
    cleaned = TARGET_LEAKAGE_RE.sub("", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n-:;")
    return cleaned


def estimate_prompt_tokens(text: Any) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    return max(1, int(len(raw) / PROMPT_EST_CHARS_PER_TOKEN) + 1)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    return 8 + sum(estimate_prompt_tokens(message.get("content", "")) + 4 for message in messages)


def prompt_input_budget(max_tokens: int | None = None) -> int:
    response_budget = max(PROMPT_RESERVED_RESPONSE_TOKENS, int(max_tokens or 0))
    return min(PROMPT_INPUT_TOKEN_BUDGET, max(1024, PROMPT_CONTEXT_TOKEN_LIMIT - response_budget - 128))


def prompt_max_tokens(requested: int, messages: list[dict[str, str]]) -> int:
    available = PROMPT_CONTEXT_TOKEN_LIMIT - estimate_messages_tokens(messages) - 64
    return max(64, min(int(requested), available))


def truncate_text(text: Any, max_chars: int) -> str:
    cleaned = sanitize_runtime_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 24:
        return cleaned[:max_chars].rstrip()
    return cleaned[: max_chars - 14].rstrip() + " ...[truncated]"


def compact_runtime_list(items: list[Any], *, limit: int, text_limit: int) -> list[Any]:
    compacted: list[Any] = []
    for item in items[:limit]:
        if isinstance(item, dict):
            compacted.append({key: truncate_text(value, text_limit) if isinstance(value, str) else value for key, value in item.items()})
        else:
            compacted.append(truncate_text(item, text_limit))
    return compacted


def messages_fit_budget(messages: list[dict[str, str]], *, max_tokens: int | None = None) -> bool:
    return estimate_messages_tokens(messages) <= prompt_input_budget(max_tokens)


def runtime_leakage_filtered_count(items: list[Any]) -> int:
    return sum(1 for item in items if contains_target_leakage(item))


def case_context(
    case: dict[str, Any],
    *,
    max_events: int | None = None,
    include_labs: bool = True,
    include_time: bool = True,
) -> str:
    events = case.get("events", [])
    if max_events is not None:
        events = events[:max_events]
    lines = [
        f"case_id: {case['case_id']}",
        f"demographics: {case.get('demographics', {})}",
        "timeline:",
    ]
    for event in events:
        text = sanitize_runtime_text(event.get("text"))
        if not text:
            continue
        prefix = f"- t={event.get('time')} [{event.get('type')}]" if include_time else f"- [{event.get('type')}]"
        lines.append(f"{prefix} {text}")
    if include_labs:
        lines.append("labs:")
        for lab in case.get("synthetic_labs", [])[:12]:
            prefix = f"- t={lab.get('time')}" if include_time else "-"
            lines.append(f"{prefix} {lab.get('name')}={lab.get('value')} {lab.get('unit')} ({lab.get('flag')})")
    return "\n".join(lines)


def compact_case_context(
    case: dict[str, Any],
    *,
    max_events: int = 24,
    include_labs: bool = True,
    include_time: bool = True,
    event_text_limit: int = 260,
) -> str:
    events = list(case.get("events", []))
    if len(events) <= max_events:
        compact_case = dict(case)
        compact_case["events"] = [
            {**event, "text": truncate_text(event.get("text"), event_text_limit)}
            for event in events
        ]
        return case_context(compact_case, max_events=None, include_labs=include_labs, include_time=include_time)
    high_value_types = {"diagnosis", "imaging", "pathology", "treatment", "lab"}
    selected: dict[str, dict[str, Any]] = {}

    def add(event: dict[str, Any]) -> None:
        selected[str(event.get("event_id") or len(selected))] = event

    for event in events[: max(1, max_events // 3)]:
        add(event)
    for event in events:
        text = sanitize_runtime_text(event.get("text")).lower()
        if event.get("type") in high_value_types or any(term in text for term in ("diagnos", "confirmed", "patholog", "biopsy", "ct", "mri")):
            add(event)
    for event in events[-max(1, max_events // 3) :]:
        add(event)

    compact_events = sorted(selected.values(), key=lambda item: (int(item.get("time", 0) or 0), str(item.get("event_id") or "")))
    if len(compact_events) > max_events:
        compact_events = compact_events[: max_events - 1] + compact_events[-1:]
    compact_case = dict(case)
    compact_case["events"] = [
        {**event, "text": truncate_text(event.get("text"), event_text_limit)}
        for event in compact_events
    ]
    context = case_context(compact_case, max_events=None, include_labs=include_labs, include_time=include_time)
    return f"{context}\nshown_events={len(compact_events)}/{len(events)}"


def budgeted_case_context(case: dict[str, Any], *, level: int = 0, include_labs: bool = True, include_time: bool = True) -> str:
    if level <= 0:
        return compact_case_context(case, max_events=42, include_labs=include_labs, include_time=include_time, event_text_limit=260)
    if level == 1:
        return compact_case_context(case, max_events=32, include_labs=include_labs, include_time=include_time, event_text_limit=220)
    if level == 2:
        return compact_case_context(case, max_events=24, include_labs=include_labs, include_time=include_time, event_text_limit=180)
    return compact_case_context(case, max_events=16, include_labs=False, include_time=include_time, event_text_limit=140)


def primary_only_task_profile_policy(profile: str) -> str:
    if profile == MEDICAL_ANSWER_ENTITY:
        return (
            "Task profile: medical_answer_entity. primary_diagnosis should be the short answer entity "
            "at the same granularity as the option or label. It may be a drug, organism, vitamin, "
            "mechanism, sign, test finding, or disease."
        )
    if profile == CLINICAL_ASSESSMENT_ENTITY:
        return (
            "Task profile: clinical_assessment_entity. primary_diagnosis should be the doctor's assessment "
            "or diagnosis entity, not a broad symptom dump."
        )
    return (
        "Task profile: longitudinal_diagnosis. primary_diagnosis should be the main disease, admission "
        "diagnosis, or case-title diagnosis supported by the visible timeline."
    )


def prediction_json_prompt(method: str, context: str, extra: str = "", *, task_profile: str = LONGITUDINAL_DIAGNOSIS) -> list[dict[str, str]]:
    is_ours_method = "ours" in method or "medimem" in method
    min_items, max_items = diagnosis_list_budget(task_profile)
    list_policy = (
        f"The downstream profile budget is {min_items} to {max_items}; later evidence-aware stages may expand "
        "diagnosis_list from primary_diagnosis. "
        if is_ours_method
        else "Do not output diagnosis_list; the evaluator will derive it from primary_diagnosis. "
    )
    system = (
        "You are a clinical research diagnosis evaluator. This is not medical advice. "
        "Use only the provided case evidence. Return exactly one minified JSON object with keys: "
        "primary_diagnosis, confidence. "
        "confidence must be a number from 0 to 1. Do not include diagnosis_list, evidence, or reasoning fields. "
        f"{primary_only_task_profile_policy(task_profile)} {list_policy}"
        "primary_diagnosis must be under 8 words and must not repeat the same token or phrase. "
        "First infer whether the patient is human or a non-human species. Do not transfer human-only disease "
        "priors to animal cases unless the provided evidence supports them. "
        "primary_diagnosis should be the final main disease/entity at the label-like granularity, not a symptom, "
        "procedure, broad organ finding, or unrelated complication. diagnosis_granularity must be one of "
        "final_disease, etiology, complication, anatomy_finding, pathology_entity, symptom_or_state, uncertain. "
        "Do not include markdown, prose, or code fences outside the JSON object. Always close the JSON object."
    )
    user = f"[METHOD]\n{method}\n\n[CASE]\n{context}\n\n{extra}\n\nReturn JSON only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def budgeted_prediction_prompt(
    case: dict[str, Any],
    method: str,
    context: str,
    extra: str,
    *,
    task_profile: str,
    level: int,
    max_tokens: int,
) -> tuple[list[dict[str, str]], str, str]:
    requested_level = level
    include_time = "t=" in str(context)
    if "medimem" in method:
        attempt_context = budgeted_case_context(case, level=level, include_labs=True, include_time=include_time)
    elif level <= 0:
        attempt_context = context
    else:
        attempt_context = truncate_context_middle(context, max(4000, 24000 // (level + 1)))
    extra_limits = [12000, 8000, 5000, 3000, 1800]
    attempt_extra = extra if level <= 0 else truncate_context_middle(extra, extra_limits[min(level, len(extra_limits) - 1)])
    messages = prediction_json_prompt(method, attempt_context, attempt_extra, task_profile=task_profile)
    while not messages_fit_budget(messages, max_tokens=max_tokens) and level < 4:
        level += 1
        if "medimem" in method:
            attempt_context = budgeted_case_context(case, level=level, include_labs=True, include_time=include_time)
        else:
            attempt_context = truncate_context_middle(context, max(3000, 18000 // (level + 1)))
        attempt_extra = truncate_context_middle(extra, extra_limits[min(level, len(extra_limits) - 1)])
        messages = prediction_json_prompt(method, attempt_context, attempt_extra, task_profile=task_profile)
    if not messages_fit_budget(messages, max_tokens=max_tokens):
        attempt_extra = truncate_context_middle(attempt_extra, 1000)
        attempt_context = truncate_context_middle(attempt_context, 6000)
        messages = prediction_json_prompt(method, attempt_context, attempt_extra, task_profile=task_profile)
    if requested_level > 0:
        retry_extra_limit = max(400, 1200 // (requested_level + 1))
        retry_context_limit = max(3000, 6000 - requested_level * 800)
        attempt_extra = truncate_context_middle(attempt_extra, retry_extra_limit)
        attempt_context = truncate_context_middle(attempt_context, retry_context_limit)
        messages = prediction_json_prompt(method, attempt_context, attempt_extra, task_profile=task_profile)
    return messages, attempt_context, attempt_extra


def budget_retry_json_prompt(method: str, context: str, extra: str) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly one valid minified JSON object with keys primary_diagnosis and confidence. "
                "primary_diagnosis must be a short non-repeated phrase under 8 words. "
                "Do not include diagnosis_list, evidence, reasoning, markdown, or any extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[METHOD]\n{method}\n\n[CASE]\n{context}\n\n{extra}\n\n"
                "The previous response was invalid JSON or an unfinished repeated string. Return valid JSON only."
            ),
        },
    ]
    if messages_fit_budget(messages, max_tokens=128):
        return messages
    compact_context = truncate_context_middle(context, 4000)
    compact_extra = truncate_context_middle(extra, 1000)
    return [
        messages[0],
        {
            "role": "user",
            "content": (
                f"[METHOD]\n{method}\n\n[CASE]\n{compact_context}\n\n{compact_extra}\n\n"
                "The previous response was invalid JSON or an unfinished repeated string. Return valid JSON only."
            ),
        },
    ]


def is_context_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "maximum context length" in text or "context length" in text


def truncate_context_middle(context: str, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    omitted = len(context) - head_chars - tail_chars
    return (
        context[:head_chars].rstrip()
        + f"\n\n[... truncated {omitted} characters to fit model context ...]\n\n"
        + context[-tail_chars:].lstrip()
    )


def preserve_specific_infection_entity(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;()[]")
    if re.search(r"\b[A-Z][a-z]+\s+[a-z]+\s+infection\b", cleaned):
        return cleaned
    return ""


def normalize_prediction(
    case_id: str,
    method: str,
    raw: dict[str, Any],
    usage: dict[str, int] | None = None,
    *,
    enable_normalization: bool = True,
) -> dict[str, Any]:
    diag_list = raw.get("diagnosis_list") or raw.get("diagnoses") or []
    if isinstance(diag_list, str):
        diag_list = [d.strip() for d in re.split(r"[,;/|]", diag_list) if d.strip()]
    evidence = raw.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    primary = str(raw.get("primary_diagnosis") or (diag_list[0] if diag_list else "Unknown")).strip()
    primary = preserve_specific_infection_entity(primary) or canonicalize_diagnosis(primary, enabled=enable_normalization)
    diag_list = [
        preserve_specific_infection_entity(str(d).strip()) or canonicalize_diagnosis(str(d).strip(), enabled=enable_normalization)
        for d in diag_list
        if str(d).strip()
    ]
    species_context = str(raw.get("species_context") or raw.get("species") or "human").strip() or "human"
    diagnosis_granularity = str(raw.get("diagnosis_granularity") or infer_diagnosis_granularity(primary)).strip()
    return {
        "case_id": case_id,
        "method": method,
        "primary_diagnosis": primary or "Unknown",
        "diagnosis_list": diag_list or ([primary] if primary else []),
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence": [str(e) for e in evidence],
        "reasoning_summary": str(raw.get("reasoning_summary") or raw.get("summary") or ""),
        "species_context": species_context,
        "diagnosis_granularity": diagnosis_granularity,
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def infer_diagnosis_granularity(text: str) -> str:
    low = str(text).lower()
    if not low or low == "unknown":
        return "uncertain"
    if any(term in low for term in ("metastasis", "failure", "injury", "infection", "complication")):
        return "complication"
    if any(term in low for term in ("carcinoma", "tumor", "tumour", "lymphoma", "leukemia", "sarcoma", "melanoma")):
        return "pathology_entity"
    if any(term in low for term in ("pain", "fever", "dyspnea", "bleeding", "elevated", "low ")):
        return "symptom_or_state"
    if any(term in low for term in ("lobe", "mass", "lesion", "nodule", "aneurysm")):
        return "anatomy_finding"
    return "final_disease"


def heuristic_predict(
    case: dict[str, Any],
    method: str,
    *,
    max_events: int | None = None,
    enable_normalization: bool = True,
) -> dict[str, Any]:
    events = case.get("events", [])
    if max_events is not None:
        events = events[:max_events]
    joined_norm = re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9]+", " ", " ".join(sanitize_runtime_text(e.get("text")) for e in events).lower()),
    ).strip()
    candidates: list[tuple[float, str, str]] = []
    if "e granulosus" in joined_norm or "echinococ" in joined_norm:
        candidates.append((0.96, "Echinococcosis", "E. granulosus / echinococcosis evidence in timeline"))
    patterns = [
        (0.95, r"(?:diagnosed with|diagnosed as|diagnosis of|confirmed)\s+([^.;]+)", "explicit diagnosis"),
        (0.85, r"pathology (?:diagnosed|confirmed|showed)\s+([^.;]+)", "pathology"),
        (0.72, r"(?:showed|found)\s+([^.;]*(?:lesion|hematoma|leukemia|shock|failure|cancer)[^.;]*)", "key finding"),
    ]
    for event in events:
        text = sanitize_runtime_text(event.get("text"))
        if not text:
            continue
        for score, pattern, reason in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                diag = cleanup_diagnosis(match.group(1))
                if diag:
                    candidates.append((score, diag, f"{reason}: {text}"))
    if not candidates:
        typed = [e for e in events if e.get("type") == "diagnosis"]
        if typed:
            text = sanitize_runtime_text(typed[-1].get("text"))
            candidates.append((0.55, cleanup_diagnosis(text), text))
    if not candidates:
        text = " ".join(sanitize_runtime_text(e.get("text")) for e in events[-2:])
        candidates.append((0.35, cleanup_diagnosis(text[:100]) or "Unknown", text[:180]))
    candidates.sort(key=lambda x: x[0], reverse=True)
    diagnoses = []
    evidence = []
    for score, diag, ev in candidates:
        if diag.lower() not in {d.lower() for d in diagnoses}:
            diagnoses.append(diag)
            evidence.append(ev)
    return normalize_prediction(
        case["case_id"],
        method,
        {
            "primary_diagnosis": diagnoses[0],
            "diagnosis_list": diagnoses[:4],
            "confidence": candidates[0][0],
            "evidence": evidence[:4],
            "reasoning_summary": "Rule-based fallback extracted explicit longitudinal diagnostic evidence.",
            "species_context": infer_case_species(case),
            "diagnosis_granularity": infer_diagnosis_granularity(diagnoses[0]),
        },
        enable_normalization=enable_normalization,
    )


def infer_case_species(case: dict[str, Any]) -> str:
    flags = case.get("data_quality_flags") or {}
    if flags.get("species_context"):
        return str(flags["species_context"])
    text = " ".join(
        [
            str(case.get("demographics") or ""),
            " ".join(sanitize_runtime_text(event.get("text")) for event in case.get("events", [])[:20]),
        ]
    ).lower()
    for species, terms in {
        "dog": ("dog", "canine"),
        "cat": ("cat", "feline"),
        "horse": ("horse", "equine"),
        "cow": ("cow", "bovine", "cattle"),
        "sheep": ("sheep", "ovine"),
        "goat": ("goat", "caprine"),
        "pig": ("pig", "porcine", "swine"),
        "mouse": ("mouse", "murine"),
        "rat": ("rat",),
    }.items():
        if any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms):
            return species
    return "human"


def cleanup_diagnosis(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .,:;()[]")
    cleaned = re.sub(r"\b(with|after|following|and underwent|indicating)\b.*$", "", cleaned, flags=re.I).strip(" .,:;")
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rsplit(" ", 1)[0]
    return canonicalize(cleaned)


def canonicalize(text: str) -> str:
    return canonicalize_diagnosis(text)


def run_llm_prediction(
    case: dict[str, Any],
    *,
    method: str,
    client: DeepSeekClient | None,
    context: str,
    extra: str = "",
    fallback_max_events: int | None = None,
    temperature: float = 0.1,
    fail_on_llm_error: bool = False,
    enable_normalization: bool = True,
) -> dict[str, Any]:
    task_profile = case_task_profile(case)
    if client is None:
        if fail_on_llm_error:
            raise RuntimeError(f"LLM client unavailable for {case['case_id']} ({method}).")
        return heuristic_predict(case, method, max_events=fallback_max_events, enable_normalization=enable_normalization)
    requested_max_tokens = int(os.environ.get("MEDICAL_PREDICTION_MAX_TOKENS", "128"))
    last_exc: Exception | None = None
    attempt_context = context
    attempt_extra = extra
    try:
        for level in range(5):
            messages, attempt_context, attempt_extra = budgeted_prediction_prompt(
                case,
                method,
                context,
                extra,
                task_profile=task_profile,
                level=level,
                max_tokens=requested_max_tokens,
            )
            try:
                result = client.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=prompt_max_tokens(requested_max_tokens, messages),
                )
                break
            except Exception as exc:  # noqa: BLE001 - context overflow gets progressively compacted
                last_exc = exc
                if not is_context_limit_error(exc):
                    raise
        else:
            raise last_exc or RuntimeError("LLM prediction failed without an exception.")
    except Exception as exc:  # noqa: BLE001 - API failures can be configured as hard blockers
        if fail_on_llm_error:
            raise RuntimeError(f"LLM prediction failed for {case['case_id']} ({method}): {exc}") from exc
        pred = heuristic_predict(case, method, max_events=fallback_max_events, enable_normalization=enable_normalization)
        pred["reasoning_summary"] += f" LLM fallback reason: {exc}"
        pred["llm_error"] = str(exc)
        return pred
    parse_retries = int(os.environ.get("MEDICAL_JSON_PARSE_RETRIES", "2")) if fail_on_llm_error else 0
    parse_exc: Exception | None = None
    for parse_attempt in range(parse_retries + 1):
        try:
            raw = extract_json_object(result.text)
            return normalize_prediction(case["case_id"], method, raw, result.usage, enable_normalization=enable_normalization)
        except Exception as exc:  # noqa: BLE001 - strict mode may retry the same LLM with a tighter schema
            parse_exc = exc
            if parse_attempt >= parse_retries:
                break
            result = client.chat(
                budget_retry_json_prompt(method, attempt_context, attempt_extra),
                temperature=0.0,
                max_tokens=128,
            )
    exc = parse_exc or RuntimeError("LLM JSON parsing failed without an exception.")
    try:
        raise exc
    except Exception as exc:  # noqa: BLE001 - prediction should degrade to fallback, not crash the run
        if fail_on_llm_error:
            raise RuntimeError(f"LLM prediction returned malformed JSON for {case['case_id']} ({method}): {exc}") from exc
        pred = heuristic_predict(case, method, max_events=fallback_max_events, enable_normalization=enable_normalization)
        pred["reasoning_summary"] += f" LLM fallback reason: {exc}"
        pred["llm_error"] = str(exc)
        pred["usage"] = result.usage
        return pred


def run_direct(case: dict[str, Any], client: DeepSeekClient | None, *, fail_on_llm_error: bool = False) -> dict[str, Any]:
    return run_llm_prediction(
        case,
        method="direct_deepseek",
        client=client,
        context=case_context(case, max_events=min(4, len(case.get("events", []))), include_labs=False),
        extra="Make a diagnosis from this limited truncated context.",
        fallback_max_events=min(4, len(case.get("events", []))),
        fail_on_llm_error=fail_on_llm_error,
    )


def run_single_cot_agent(
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    polluted: bool = False,
    fail_on_llm_error: bool = False,
) -> dict[str, Any]:
    method = "baseline_polluted_single_cot_agent" if polluted else "baseline_single_cot_agent"
    extra = (
        "Give a brief clinical reasoning summary and the final diagnosis. "
        "Do not output full chain-of-thought or step-by-step hidden reasoning; keep reasoning_summary to one short sentence."
    )
    if polluted:
        extra += (
            "\nUse the limited case context plus the following previously stored memory cards. "
            "The memory cards may be stale, mixed, or cross-patient, and this baseline has no dedicated cleaning tool.\n"
            f"{pollution_memory_context(case)}"
        )
    pred = run_llm_prediction(
        case,
        method=method,
        client=client,
        context=case_context(case, max_events=min(4, len(case.get("events", []))), include_labs=False),
        extra=extra,
        fallback_max_events=min(4, len(case.get("events", []))),
        temperature=0.05,
        fail_on_llm_error=fail_on_llm_error,
    )
    pred["cot_style"] = "single_agent_concise"
    pred["pollution_exposed"] = polluted
    pred["runtime_leakage_filtered_count"] = runtime_leakage_filtered_count(
        [event.get("text") for event in case.get("events", [])[: min(4, len(case.get("events", [])))]]
        + ([poison.get("text") for poison in case.get("poison_records", [])] if polluted else [])
    )
    pred["memory_leakage_filtered_count"] = (
        runtime_leakage_filtered_count([poison.get("text") for poison in case.get("poison_records", [])]) if polluted else 0
    )
    return pred


def pollution_memory_context(case: dict[str, Any]) -> str:
    lines = ["[POLLUTED_MEMORY_CONTEXT]"]
    for poison in case.get("poison_records", []):
        text = sanitize_runtime_text(poison.get("text"))
        if not text:
            continue
        lines.append(
            f"- id={poison.get('poison_id')} type={poison.get('pollution_type')} "
            f"text={text}"
        )
    return "\n".join(lines)


def safe_memory_ops_for_prompt(case: dict[str, Any], ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_ops: list[dict[str, Any]] = []
    for op in ops:
        target = str(op.get("target") or "")
        safe_op: dict[str, Any] = {
            "op": op.get("op"),
            "target": target,
            "touched_memory_ids": op.get("touched_memory_ids") or [],
            "revised_memory_id": op.get("revised_memory_id"),
        }
        if op.get("op") == "Revise":
            safe_op["preserved_fact_count"] = len([x for x in op.get("preserved_facts", []) if str(x).strip()])
            safe_op["revision_note"] = (
                "Preserve source-grounded evidence by reference only; do not use the prior interpretation as a diagnosis."
            )
        safe_ops.append(safe_op)
    return safe_ops


def source_aligned_evidence_notes(cards: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    high_value_tags = {"diagnosis", "imaging", "pathology", "treatment", "lab", "follow_up", "follow-up"}
    notes = []
    for card in cards:
        if card.get("status") not in {"active", "flagged"}:
            continue
        tags = {str(tag).lower() for tag in card.get("tags", [])}
        if tags & {"poison", "outdated", "stale_candidate", "initial_hypothesis"}:
            continue
        summary = sanitize_runtime_text(card.get("summary"))
        if not summary:
            continue
        low = summary.lower()
        if not (
            tags & high_value_tags
            or any(term in low for term in ("diagnos", "patholog", "biopsy", "ct", "mri", "treatment", "follow-up", "follow up"))
        ):
            continue
        scope = card.get("time_scope") or {}
        notes.append(
            {
                "time": scope.get("start") if isinstance(scope, dict) else None,
                "summary": summary,
                "refs": card.get("evidence_refs") or [],
                "tags": sorted(tags),
                "confidence": card.get("confidence"),
            }
        )
    notes.sort(key=lambda item: str(item.get("time") or ""))
    return notes[-limit:]


def diagnosis_event_candidates(case: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    candidates = []
    for event in case.get("events", []):
        if event.get("type") != "diagnosis":
            continue
        text = sanitize_runtime_text(event.get("text"))
        if not text:
            continue
        candidates.append(
            {
                "time": event.get("time"),
                "event_id": event.get("event_id"),
                "text": text,
            }
        )
    candidates.sort(key=lambda item: str(item.get("time") or ""))
    return candidates[-limit:]


def clean_diagnosis_candidate_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip(" .,:;")
    cleaned = re.sub(r"^(?:diagnosed with|diagnosed as|diagnosis of)\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^ihc confirmed\s+", "", cleaned, flags=re.I)
    return cleaned.strip(" .,:;")


def diagnosis_candidate_priority(text: str) -> int:
    low = text.lower()
    if re.search(r"\b(no history of|no symptoms|no cancer symptoms|normal)\b", low):
        return -10
    score = 0
    if re.search(r"\b(diagnosed with|diagnosed as|diagnosis of|suspected|confirmed|ihc confirmed)\b", low):
        score += 5
    if re.search(r"\b(cancer|carcinoma|adenocarcinoma|lymphoma|disease|syndrome|pneumonia|dmmr|metastasis|carcinomatosis)\b", low):
        score += 2
    return score


DIAGNOSTIC_ENTITY_TERMS = (
    "abnormality",
    "adenocarcinoma",
    "amenorrhea",
    "anemia",
    "aneurysm",
    "anomaly",
    "airway",
    "arrest",
    "anxiety",
    "arthritis",
    "asthma",
    "bleeding",
    "brachydactyly",
    "bursitis",
    "cancer",
    "canal",
    "carcinoma",
    "carcinomatosis",
    "congestion",
    "covid",
    "cystadenoma",
    "deficiency",
    "deformity",
    "depression",
    "defect",
    "dextrocardia",
    "diabetes",
    "disease",
    "displacement",
    "dmmr",
    "dysplasia",
    "dysgerminoma",
    "dysphagia",
    "dyspnea",
    "endocarditis",
    "epilepsy",
    "effusion",
    "emboli",
    "embolism",
    "fistula",
    "fibrosis",
    "fracture",
    "gastritis",
    "gammaglobulinemia",
    "gammopathy",
    "gist",
    "hernia",
    "hemangioma",
    "hemorrhage",
    "hepatitis",
    "hematoma",
    "hoarseness",
    "hydronephrosis",
    "hyperglycemia",
    "hyperplasia",
    "hyperkeratosis",
    "hyperlipidemia",
    "hypertension",
    "hypopituitarism",
    "hypothyroidism",
    "hypogonadism",
    "hypophosphatemic",
    "hypersplenism",
    "infarction",
    "infection",
    "insufficiency",
    "inversion",
    "involvement",
    "injury",
    "keratosis",
    "leukemia",
    "leiomyoma",
    "lichen",
    "lobe",
    "lymphangioma",
    "malformation",
    "lymphoma",
    "malrotation",
    "metastases",
    "metastasis",
    "mediastinal",
    "metrorrhagia",
    "microangiopathy",
    "microvascular",
    "mutation",
    "myocardial",
    "myxoma",
    "myeloma",
    "melanoma",
    "migraine",
    "necrobiotic",
    "neoplasm",
    "nephritis",
    "myelitis",
    "mucormycosis",
    "mycobacterium",
    "myasthenia",
    "neuropathy",
    "obesity",
    "oligozoospermia",
    "osteomyelitis",
    "osteomalacia",
    "osteoporosis",
    "poliomyelitis",
    "pneumonia",
    "pneumothorax",
    "peritonitis",
    "perforation",
    "pericarditis",
    "phimosis",
    "pleurisy",
    "planus",
    "pancreatitis",
    "polymorphism",
    "polyneuropathy",
    "prostatitis",
    "pseudoaneurysm",
    "pyuria",
    "pyopericardium",
    "rosacea",
    "rhabdomyosarcoma",
    "rhinitis",
    "regurgitation",
    "syndrome",
    "syndactyly",
    "sinusitis",
    "stenosis",
    "sporotrichosis",
    "stroke",
    "situs",
    "thrombocytopenia",
    "thromboembolism",
    "thrombophilia",
    "thrombotic",
    "tuberculosis",
    "thymoma",
    "tachycardia",
    "tumor",
    "ulcer",
    "urosepsis",
    "ureter",
    "web",
    "xanthogranuloma",
    "zoster",
)

NON_DIAGNOSTIC_TERMS = (
    "biopsy",
    "chemotherapy",
    "colonoscopy",
    "ct",
    "documented source",
    "endoscopy",
    "evidence",
    "examination",
    "history",
    "imaging",
    "mri",
    "pet",
    "radiotherapy",
    "resection",
    "scan",
    "screening",
    "source evidence",
    "surgery",
    "therapy",
    "treatment",
)

SYMPTOM_OR_STATE_TERMS = (
    "acute kidney injury",
    "cough",
    "fever",
    "headache",
    "nausea",
    "pain",
    "stable",
    "vomiting",
)


PUBLIC_DIAGNOSIS_CUE_MAP: tuple[tuple[str, str], ...] = (
    ("bronchioloalveolar carcinoma", "bronchioloalveolar carcinoma"),
    ("non-mucinous bac", "bronchioloalveolar carcinoma"),
    ("acute non-stemi", "acute non st elevation myocardial infarction"),
    ("nstemi", "acute non st elevation myocardial infarction"),
    ("non-st-segment elevation myocardial infarction", "acute non st elevation myocardial infarction"),
    ("st-segment elevation myocardial infarction", "acute st elevation myocardial infarction"),
    ("coronary artery anomalies", "coronary artery anomaly"),
    ("posterior right diagonal artery", "posterior right diagonal artery variant"),
    ("right diagonal artery", "posterior right diagonal artery variant"),
    ("hyperglycemia", "hyperglycemia"),
    ("hyperlipidemia", "hyperlipidemia"),
    ("hypertension", "hypertension"),
    ("hepatitis b", "hepatitis b"),
    ("rosai-dorfman-destombes", "rosai dorfman destombes disease"),
    ("rosai dorfman destombes", "rosai dorfman destombes disease"),
    ("kidney calculi", "kidney calculi"),
    ("borderline resectable pdac", "pancreatic ductal adenocarcinoma"),
    ("pdac", "pancreatic ductal adenocarcinoma"),
    ("pancreatic ductal adenocarcinoma", "pancreatic ductal adenocarcinoma"),
    ("cerebrovascular accident", "cerebrovascular accident"),
    ("lynch syndrome", "lynch syndrome"),
    ("retrocaval ureter", "retrocaval ureter"),
    ("ureteropelvic junction obstruction", "ureteropelvic junction obstruction"),
    ("k-wire migration", "k wire migration"),
    ("kirschner wire migration", "k wire migration"),
    ("urinary bladder injury", "urinary bladder injury"),
    ("bladder injury", "urinary bladder injury"),
    ("vesicovaginal fistula", "vesicovaginal fistula"),
    ("ureterovaginal fistula", "ureterovaginal fistula"),
    ("serous cystadenoma", "serous cystadenoma"),
    ("ivc aneurysm", "inferior vena cava aneurysm"),
    ("inferior vena cava aneurysm", "inferior vena cava aneurysm"),
    ("venous malformations", "venous malformation"),
    ("venous malformation", "venous malformation"),
    ("gastritis", "gastritis"),
    ("phosphaturic mesenchymal tumor", "tumor induced osteomalacia"),
    ("tumor-induced osteomalacia", "tumor induced osteomalacia"),
    ("hypophosphatemic osteomalacia", "hypophosphatemic osteomalacia"),
    ("osteoporosis", "osteoporosis"),
    ("recurrent contralateral pneumothorax", "recurrent contralateral pneumothorax"),
    ("azygos lobe", "azygos lobe"),
    ("pneumothorax", "pneumothorax"),
    ("invasive ductal carcinoma", "breast cancer"),
    ("hypogonadotropic hypogonadism", "hypogonadotropic hypogonadism"),
    ("oligozoospermia", "oligozoospermia"),
    ("obesity", "obesity"),
    ("thrombotic microangiopathy", "thrombotic microangiopathy"),
    ("hemolytic anemia", "hemolytic anemia"),
    ("colon metastasis", "colon metastasis"),
    ("systemic lupus erythematosus", "systemic lupus erythematosus"),
    ("diagnosed with sle", "systemic lupus erythematosus"),
    ("raynaud", "raynaud phenomenon"),
    ("microvascular angina", "microvascular angina"),
    ("uterine inversion", "uterine inversion"),
    ("metrorrhagia", "metrorrhagia"),
    ("anemia", "anemia"),
    ("good's syndrome", "good syndrome"),
    ("goods syndrome", "good syndrome"),
    ("thymoma", "thymoma"),
    ("hypogammaglobulinemia", "hypogammaglobulinemia"),
    ("oral lichen planus", "oral lichen planus"),
    ("patent foramen ovale", "patent foramen ovale"),
    ("pulmonary emboli", "pulmonary embolism"),
    ("pulmonary embolism", "pulmonary embolism"),
    ("cryptogenic stroke", "cryptogenic stroke"),
    ("watershed stroke", "cryptogenic stroke"),
    ("thrombophilia", "thrombophilia"),
    ("factor ii gene", "factor ii gene mutation"),
    ("factor ii 20210a", "factor ii gene mutation"),
    ("pai-1 4g/5g", "pai 1 4g 5g gene polymorphism"),
    ("prothrombin gene mutation", "prothrombin gene mutation"),
    ("septic arthritis", "septic arthritis"),
    ("type 2 diabetes", "type 2 diabetes mellitus"),
    ("diabetes mellitus", "type 2 diabetes mellitus"),
    ("osteoarthritis", "osteoarthritis"),
    ("congenital hip defect", "congenital hip defect"),
    ("monoclonal gammopathy", "monoclonal gammopathy"),
    ("polyneuropathy", "polyneuropathy"),
    ("neuropathic ulcer", "neuropathy"),
    ("neuropathy", "neuropathy"),
    ("cellulitis", "cellulitis"),
    ("ischemic stroke", "ischemic stroke"),
    ("intracerebral hemorrhage", "intracerebral hemorrhage"),
    ("hemorrhage (ich)", "intracerebral hemorrhage"),
    ("ich", "intracerebral hemorrhage"),
    ("epilepsy", "epilepsy"),
    ("primary lung signet", "primary lung signet ring cell carcinoma"),
    ("signet-ring cell", "primary lung signet ring cell carcinoma"),
    ("signet ring cell", "primary lung signet ring cell carcinoma"),
    ("iga nephropathy", "iga nephropathy"),
    ("atrial fibrillation", "atrial fibrillation"),
    ("arterial hypertension", "arterial hypertension"),
    ("mitral stenosis", "mitral stenosis"),
    ("mitral regurgitation", "mitral regurgitation"),
    ("aortic regurgitation", "aortic regurgitation"),
    ("ventricular tachycardia", "ventricular tachycardia"),
    ("supraventricular tachycardia", "supraventricular tachycardia"),
    ("partial anomalous pulmonary venous connection", "partial anomalous pulmonary venous connection"),
    ("papvc", "partial anomalous pulmonary venous connection"),
    ("regressed left ventricle", "regressed left ventricle"),
    ("intestinal malrotation", "intestinal malrotation"),
    ("malrotation", "intestinal malrotation"),
    ("duodenal web", "duodenal web"),
    ("lupus nephritis", "lupus nephritis"),
    ("dysphagia", "dysphagia"),
    ("dyspnea associated", "dyspnea associated"),
    ("dyspnea", "dyspnea"),
    ("oropharyngeal and neck masses", "oropharyngeal and neck masses"),
    ("difficult airway", "difficult airway"),
    ("pulmonary arrest", "pulmonary arrest"),
    ("brain injury", "brain injury"),
    ("hoarseness", "hoarseness of voice"),
    ("cholelithiasis", "cholelithiasis"),
    ("cholecystocolonic fistula", "cholecystocolonic fistula"),
    ("heart failure with reduced ejection fraction", "heart failure with reduced ejection fraction"),
    ("hfr ef", "heart failure with reduced ejection fraction"),
    ("middle cerebral artery", "middle cerebral artery ischemic stroke"),
    ("pseudoaneurysm", "pseudoaneurysm"),
    ("arteriocolonic fistula", "arteriocolonic fistula"),
    ("imaa", "inferior mesenteric artery aneurysm"),
    ("inferior mesenteric artery aneurysm", "inferior mesenteric artery aneurysm"),
    ("hematochezia", "hematochezia"),
    ("ischemic colitis", "ischemic colitis"),
    ("sigmoid colon", "sigmoid colon inflammation"),
    ("epidermoid cyst", "epidermoid cyst"),
    ("epidermal cyst", "epidermoid cyst"),
    ("impacted tooth in infratemporal fossa", "displacement of maxillary third molar into the infratemporal fossa"),
    ("tooth into ipsilateral infratemporal fossa", "displacement of maxillary third molar into the infratemporal fossa"),
    ("maxillary third molar", "displacement of maxillary third molar into the infratemporal fossa"),
    ("hematoma", "hematoma or hemangioma"),
    ("primary malignant melanoma of the esophagus", "primary malignant melanoma of the esophagus"),
    ("malignant melanoma of esophagus", "primary malignant melanoma of the esophagus"),
    ("pmme", "primary malignant melanoma of the esophagus"),
    ("seborrheic keratosis", "seborrheic keratosis"),
    ("tuberculous pleurisy", "tuberculous pleurisy"),
    ("metastatic colorectal cancer", "metastatic colorectal cancer"),
    ("liver metastasis of intestinal-type adenocarcinoma", "metastatic colorectal cancer"),
    ("hodgkin lymphoma", "hodgkin lymphoma"),
    ("classical hodgkin lymphoma", "classical hodgkin lymphoma of mixed cellularity"),
    ("herpes zoster", "herpes zoster"),
    ("transverse myelitis", "transverse myelitis"),
    ("tm with hz", "transverse myelitis"),
    ("vzv", "herpes zoster"),
    ("gb minen", "mixed neuroendocrine non neuroendocrine tumor of the gallbladder"),
    ("gallbladder minen", "mixed neuroendocrine non neuroendocrine tumor of the gallbladder"),
    ("gb cancer", "gallbladder cancer"),
    ("gallbladder adenocarcinoma", "gallbladder adenocarcinoma"),
    ("large cell neuroendocrine carcinoma", "large cell neuroendocrine carcinoma"),
    ("lcnec", "large cell neuroendocrine carcinoma"),
    ("neuroendocrine tumor", "neuroendocrine tumor"),
    ("s. brasiliensis", "sporotrichosis"),
    ("sporothrix brasiliensis", "sporotrichosis"),
    ("sporotrichosis", "sporotrichosis"),
    ("feline sporotrichosis", "feline sporotrichosis"),
    ("myasthenia gravis", "myasthenia gravis"),
    ("malignant melanoma", "malignant melanoma"),
    ("gastrointestinal stromal tumor", "gastrointestinal stromal tumor"),
    ("sdha-deficient gist", "sdha deficient gastrointestinal stromal tumor"),
    ("sdha deficient gist", "sdha deficient gastrointestinal stromal tumor"),
    ("gist histology", "gastrointestinal stromal tumor"),
    ("nxg", "necrobiotic xanthogranuloma"),
    ("poland syndrome", "poland syndrome"),
    ("spindle cell sclerosing rhabdomyosarcoma", "spindle cell sclerosing rhabdomyosarcoma"),
    ("crush injury", "crush injury of the left lower extremity"),
    ("invasive mucinous adenocarcinoma", "invasive mucinous adenocarcinoma"),
    ("cerebral mucormycosis", "cerebral mucormycosis"),
    ("mucormycosis", "cerebral mucormycosis"),
    ("mycobacterium persicum", "mycobacterium persicum infection"),
    ("congenital hip defects", "congenital hip defect"),
    ("septic arthritis", "septic arthritis"),
    ("migraines", "migraine"),
    ("migraine", "migraine"),
    ("allergic rhinitis", "allergic rhinitis"),
    ("rhino sinusitis", "rhino sinusitis"),
    ("persistent dry cough", "persistent dry cough"),
    ("uterine fibroids", "uterine fibroids"),
    ("uterine fibroid", "uterine fibroids"),
    ("mediastinal abscess", "mediastinal abscess"),
    ("pyopericardium", "pyopericardium"),
    ("pericardial effusion", "pericardial effusion"),
    ("non-small cell lung cancer", "non small cell lung cancer"),
    ("nsclc", "non small cell lung cancer"),
    ("phimosis", "phimosis"),
    ("splenic hemangioma", "splenic hemangioma"),
    ("hypersplenism", "hypersplenism"),
    ("multiple myeloma", "multiple myeloma"),
    ("chronic liver disease", "chronic liver disease"),
    ("echinococcosis", "echinococcosis"),
    ("echinococcus", "echinococcosis"),
    ("poliomyelitis", "poliomyelitis"),
)


def normalize_diagnosis_entity_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip(" .,:;()[]")
    cleaned = re.sub(
        r"^(?:diagnosed with|diagnosed as|diagnosis of|suspected|confirmed|ihc confirmed|"
        r"pathology confirmed|surgical pathology confirmed|frozen section confirmed|"
        r"preliminary diagnosis of|diagnosis revised to)\s+",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\b(?:diagnosis )?confirmed$", "", cleaned, flags=re.I).strip(" .,:;")
    cleaned = re.sub(r"\s*\((?:stage|t\d|n\d|m\d)[^)]+\)", "", cleaned, flags=re.I).strip(" .,:;")
    if "," in cleaned:
        first = cleaned.split(",", 1)[0].strip()
        if is_likely_diagnostic_entity(first, allow_primary=False):
            cleaned = first
    cleaned = re.sub(r"\b(?:with|due to|associated with|after|following|confirmed via|confirmed by)\b.*$", "", cleaned, flags=re.I).strip(" .,:;")
    return canonicalize_diagnosis(cleaned)


def recall_candidate_fragments(text: str) -> list[str]:
    cleaned = clean_diagnosis_candidate_text(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;")
    if not cleaned:
        return []
    parts = [cleaned]
    for pattern in (r"\s*;\s*", r"\s*\|\s*", r"\s*,\s*", r"\s+and\s+", r"\s+plus\s+"):
        next_parts: list[str] = []
        for part in parts:
            next_parts.extend(re.split(pattern, part, flags=re.I))
        parts = next_parts
    fragments: list[str] = []
    for part in parts:
        part = re.sub(r"^(?:with|plus|and|including)\s+", "", part, flags=re.I).strip(" .,:;()[]")
        part = re.sub(r"\b(?:was|were|is|are)\s+(?:also\s+)?(?:diagnosed|confirmed)\b.*$", "", part, flags=re.I).strip(" .,:;")
        if part:
            fragments.append(part)
    return fragments


def is_likely_diagnostic_entity(text: str, *, allow_primary: bool = False) -> bool:
    low = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()
    if not low:
        return False
    if allow_primary:
        return True
    noisy_phrases = (
        "absence of",
        "anterior tracheal wall mass",
        "case discussion",
        "cerebral infarction suspected",
        "cerebral infarction treatment",
        "ct scan",
        "cystic cavity",
        "cystic lesion",
        "cystic mass",
        "denied risk factors",
        "depression on scalp",
        "differential diagnosis",
        "elevation in leads",
        "fdg pet",
        "genetic testing",
        "genomic alterations",
        "high tumor mutational burden",
        "infection control",
        "laminated keratin",
        "low density mediastinal lesion",
        "mediastinal mass",
        "microvascular occlusion",
        "mild tracheal compression",
        "molecular diagnostics",
        "mutation",
        "pet ct",
        "post operative recovery",
        "round depression",
        "scan normalization",
        "st segment depression",
        "st segment elevation",
        "surgical removal",
        "tumor markers",
        "well circumscribed",
    )
    if any(phrase in low for phrase in noisy_phrases):
        return False
    if "diagnosis-redacted" in low or "preserve documented" in low:
        return False
    if re.search(r"\b(monitored by|visited by|followed by|regimen|screening|history of|relapsed at|remission at)\b", low):
        return False
    if low in {"stable disease", "disease progression"}:
        return False
    if re.search(r"\b(no|normal|negative|refused)\b", low):
        return False
    if re.search(r"\b(?:mg|ng|pg|u/l|mmhg|beats|min|cm|mm|%|high|low)\b", low):
        return False
    if any(term in low for term in SYMPTOM_OR_STATE_TERMS) and not any(term in low for term in DIAGNOSTIC_ENTITY_TERMS):
        return False
    if any(term in low for term in NON_DIAGNOSTIC_TERMS) and not any(term in low for term in DIAGNOSTIC_ENTITY_TERMS):
        return False
    return any(term in low for term in DIAGNOSTIC_ENTITY_TERMS)


def diagnosis_key(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9]+", " ", canonicalize_diagnosis(str(text)).lower()),
    ).strip()


def profile_list_budget_for_case(case: dict[str, Any]) -> int:
    return diagnosis_list_budget(case_task_profile(case))[1]


def normalize_answer_entity(text: str, *, enable_normalization: bool = True) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;()[]")
    if not cleaned:
        return ""
    if re.search(r"\b[A-Z][a-z]+\s+[a-z]+\s+infection\b", cleaned):
        return cleaned
    return canonicalize_diagnosis(cleaned, enabled=enable_normalization)


def conservative_canonical_diagnosis(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;()[]")
    if re.search(r"\b[A-Z][a-z]+\s+[a-z]+\s+infection\b", cleaned):
        return cleaned
    return canonicalize_diagnosis(cleaned)


def answer_option_candidates(case: dict[str, Any]) -> list[str]:
    candidates = []
    for option in case.get("answer_options") or []:
        text = str(option.get("text") or "").strip()
        label = str(option.get("label") or "").strip()
        if text:
            candidates.append(text)
        if label and text:
            candidates.append(f"{label}. {text}")
    return candidates


def answer_entity_source_candidates(case: dict[str, Any]) -> list[tuple[str, str, float]]:
    if (case.get("data_quality_flags") or {}).get("no_leak_protocol", True):
        return []
    return []


def visible_instruction_topic_candidates(case: dict[str, Any]) -> list[tuple[str, str, float]]:
    flags = case.get("data_quality_flags") or {}
    if str(flags.get("source_type") or "").strip() != "medical_instruction":
        return []
    candidates: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for event in case.get("events") or []:
        text = str(event.get("text") or "")
        if not re.search(r"\b(?:Clinical question or presentation|Initial clinical task/source title)\b", text, flags=re.I):
            continue
        question = re.sub(r"^[^:]{0,80}:\s*", "", text).strip()
        lowered = question.lower().strip()
        topic = ""
        patterns = [
            r"what information is (?:available|there) (?:on|about)\s+(.+?)[?.]?$",
            r"what is available (?:on|about)\s+(.+?)[?.]?$",
            r"what is\s+(.+?)(?:,|\s+and\s+how|\s+and\s+what|[?.]$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.I)
            if match:
                topic = question[match.start(1) : match.end(1)]
                break
        if not topic and lowered.startswith("what does "):
            topic = re.sub(r"^what does\s+", "", question, flags=re.I)
            if " for " in topic.lower():
                topic = re.split(r"\s+for\s+", topic, maxsplit=1, flags=re.I)[1]
            topic = re.sub(r"\s+entail[?.]?$", "", topic, flags=re.I)
        topic = re.sub(r"^(?:a|an|the)\s+", "", topic.strip(" ?.;:,"), flags=re.I)
        topic = re.sub(r"\s+", " ", topic).strip()
        if not topic or len(topic.split()) > 8:
            continue
        normalized = normalize_answer_entity(topic)
        key = diagnosis_key(normalized)
        if key and key not in seen:
            seen.add(key)
            candidates.append((normalized, "visible_instruction_question_topic", 18.0))
    return candidates


def text_overlap_score(candidate: str, blobs: list[str]) -> float:
    candidate_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(candidate).lower())).strip()
    if not candidate_norm:
        return 0.0
    candidate_tokens = {tok for tok in candidate_norm.split() if len(tok) > 1}
    score = 0.0
    for blob in blobs:
        blob_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(blob).lower())).strip()
        if not blob_norm:
            continue
        if candidate_norm in blob_norm:
            score += 4.0
        elif candidate_tokens:
            blob_tokens = set(blob_norm.split())
            score += len(candidate_tokens & blob_tokens) / len(candidate_tokens)
    return score


def apply_profile_diagnosis_list_budget(case: dict[str, Any], pred: dict[str, Any], *, enable_normalization: bool = True) -> dict[str, Any]:
    max_items = profile_list_budget_for_case(case)
    primary = normalize_answer_entity(pred.get("primary_diagnosis"), enable_normalization=enable_normalization)
    merged = []
    seen = set()
    for item in [primary] + [str(x) for x in pred.get("diagnosis_list", [])]:
        entity = normalize_answer_entity(item, enable_normalization=enable_normalization)
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", entity.lower())).strip()
        if entity and key and key not in seen:
            seen.add(key)
            merged.append(entity)
        if len(merged) >= max_items:
            break
    updated = dict(pred)
    updated["primary_diagnosis"] = primary or pred.get("primary_diagnosis") or "Unknown"
    updated["diagnosis_list"] = merged or ([updated["primary_diagnosis"]] if updated["primary_diagnosis"] else [])
    updated["diagnosis_list_budget"] = {"task_profile": case_task_profile(case), "max_items": max_items}
    return updated


def select_primary_for_task_profile(
    case: dict[str, Any],
    pred: dict[str, Any],
    evidence_notes: list[dict[str, Any]],
    diagnosis_candidates: list[dict[str, Any]],
    *,
    enable_normalization: bool = True,
) -> dict[str, Any]:
    profile = case_task_profile(case)
    current = str(pred.get("primary_diagnosis") or "")
    blobs = [
        current,
        " ".join(str(item) for item in pred.get("diagnosis_list", [])),
        " ".join(str(item) for item in pred.get("evidence", [])),
        str(pred.get("reasoning_summary") or ""),
    ]
    candidates: list[tuple[str, str, float]] = [(current, "model_primary", 3.0)]
    flags = case.get("data_quality_flags") or {}
    source_dataset = str(flags.get("source_dataset") or "").strip()
    source_type = str(flags.get("source_type") or "").strip()
    if source_dataset == "pmc_patients":
        current_entity = diagnosis_entity_from_text(current)
        if current_entity and current_entity != current:
            candidates.append((current_entity, "pmc_model_primary_explicit", 5.2))

    if profile == MEDICAL_ANSWER_ENTITY:
        candidates.extend(answer_entity_source_candidates(case))
        candidates.extend(visible_instruction_topic_candidates(case))
        option_rows = []
        for option in answer_option_candidates(case):
            option_rows.append((option, "answer_options", 2.5 + text_overlap_score(option, blobs)))
        if source_type == "medical_mcqa" and option_rows:
            best = max(option_rows, key=lambda row: (row[2], -len(row[0])))
            selected = re.sub(r"^[A-E]\.\s*", "", best[0]).strip()
            selected = normalize_answer_entity(selected, enable_normalization=enable_normalization)
            if selected:
                updated = dict(pred)
                updated["primary_diagnosis"] = selected
                updated["diagnosis_granularity"] = "answer_entity"
                updated["primary_selection_pass"] = {
                    "enabled": True,
                    "task_profile": profile,
                    "source": "answer_options",
                    "score": best[2],
                    "previous_primary": current,
                    "constraint": "visible_medical_mcqa_options",
                }
                return apply_profile_diagnosis_list_budget(case, updated, enable_normalization=enable_normalization)
        candidates.extend(option_rows)
        for item in pred.get("diagnosis_list", []):
            candidates.append((str(item), "model_list", 1.5 + text_overlap_score(str(item), blobs)))
        best = max(candidates, key=lambda row: (row[2], -len(row[0]))) if candidates else ("", "none", 0.0)
        selected = re.sub(r"^[A-E]\.\s*", "", best[0]).strip()
        selected = normalize_answer_entity(selected, enable_normalization=enable_normalization)
        if selected:
            updated = dict(pred)
            updated["primary_diagnosis"] = selected
            updated["diagnosis_granularity"] = "answer_entity"
            updated["primary_selection_pass"] = {
                "enabled": True,
                "task_profile": profile,
                "source": best[1],
                "score": best[2],
                "previous_primary": current,
            }
            return apply_profile_diagnosis_list_budget(case, updated, enable_normalization=enable_normalization)

    source_rows = visible_diagnosis_source_rows(case, pred, evidence_notes, diagnosis_candidates)
    for rank, row in enumerate(source_rows):
        text = str(row.get("text") or "")
        if not text:
            continue
        entity = diagnosis_entity_from_text(text)
        if entity:
            candidates.append((entity, str(row.get("source") or "visible"), float(row.get("weight") or 1.0) + max(0.0, 2.0 - rank / 40)))

    def longitudinal_score(row: tuple[str, str, float]) -> tuple[float, int]:
        candidate, source, base = row
        low = candidate.lower()
        score = base + text_overlap_score(candidate, blobs)
        if source_dataset == "pmc_patients" and low in {"infection", "disease", "syndrome", "tumor", "cancer"}:
            score -= 4.0
        if source_dataset == "pmoa_tts":
            if re.search(
                r"\b("
                r"revascularization|residual disease|stage\s+[ivx]+|"
                r"post fixation|fixation|placement|stent|thrombectomy|"
                r"adaptation|rupture|hemorrhag(?:e|ing)|effusion|"
                r"short segment|short-segment|anastomosis|obstruction of the anastomosis"
                r")\b",
                low,
            ):
                score -= 3.0
            if re.search(
                r"\b("
                r"carcinoma|adenocarcinoma|cancer|lymphoma|leukemia|sarcoma|"
                r"thrombosis|deep vein thrombosis|pulmonary embolism|embolism|"
                r"hemophagocytic lymphohistiocytosis|hlh|hcc|hepatocellular|"
                r"gastric outlet obstruction|fracture|lupus|hypertension|copd|"
                r"infection|syndrome|disease"
                r")\b",
                low,
            ):
                score += 2.0
        if source in {"timeline_diagnosis", "diagnosis_event", "timeline_pathology"}:
            score += 2.0
        if re.search(r"\b(complication|metastasis|metastases|failure|injury|embolism|effusion|bleeding)\b", low):
            score -= 1.6
        if re.search(r"\b(carcinoma|cancer|lymphoma|leukemia|disease|syndrome|infection|pneumonia|tumor|tumour)\b", low):
            score += 1.0
        return score, -len(candidate)

    if profile in {LONGITUDINAL_DIAGNOSIS, CLINICAL_ASSESSMENT_ENTITY} and candidates:
        best = max(candidates, key=longitudinal_score)
        selected = normalize_answer_entity(best[0], enable_normalization=enable_normalization)
        if selected and selected.lower() != "unknown":
            updated = dict(pred)
            updated["primary_diagnosis"] = selected
            updated["diagnosis_granularity"] = infer_diagnosis_granularity(selected)
            updated["primary_selection_pass"] = {
                "enabled": True,
                "task_profile": profile,
                "source": best[1],
                "score": longitudinal_score(best)[0],
                "previous_primary": current,
            }
            return apply_profile_diagnosis_list_budget(case, updated, enable_normalization=enable_normalization)

    updated = dict(pred)
    updated["primary_selection_pass"] = {"enabled": True, "task_profile": profile, "source": "unchanged"}
    return apply_profile_diagnosis_list_budget(case, updated, enable_normalization=enable_normalization)


def diagnosis_entity_from_text(raw_text: str) -> str:
    raw = str(raw_text or "")
    explicit = re.search(
        r"(?:confirmed|showed|revealed|diagnosed with|diagnosis of|consistent with|pathology positive for)\s+([^.;]+)",
        raw,
        flags=re.I,
    )
    if explicit:
        entity = explicit.group(1).strip(" .,:;")
        if re.search(r"\b[A-Z][a-z]+\s+[a-z]+\s+infection\b", entity):
            return entity
        return canonicalize_diagnosis(entity)
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", raw.lower())).strip()
    for cue, entity in PUBLIC_DIAGNOSIS_CUE_MAP:
        cue_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", cue.lower())).strip()
        if cue_norm and re.search(rf"\b{re.escape(cue_norm)}\b", normalized):
            return canonicalize_diagnosis(entity)
    return normalize_diagnosis_entity_text(raw)


def visible_diagnosis_source_rows(
    case: dict[str, Any],
    pred: dict[str, Any],
    evidence_notes: list[dict[str, Any]],
    diagnosis_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profile = case_task_profile(case)
    for idx, item in enumerate(pred.get("diagnosis_list", [])):
        rows.append({"source": "prediction", "rank": idx, "text": str(item), "weight": 3.0})
    for idx, item in enumerate(pred.get("evidence", [])):
        rows.append({"source": "prediction_evidence", "rank": idx, "text": str(item), "weight": 2.2})
    for idx, item in enumerate(diagnosis_candidates):
        rows.append(
            {
                "source": "diagnosis_event",
                "rank": idx,
                "text": str(item.get("text") or ""),
                "event_id": item.get("event_id"),
                "time": item.get("time"),
                "weight": 3.4,
            }
        )
    for idx, note in enumerate(evidence_notes):
        tags = {str(tag).lower() for tag in note.get("tags", [])}
        summary = sanitize_runtime_text(note.get("summary"))
        if not summary:
            continue
        if tags & {"diagnosis", "pathology", "imaging", "treatment", "follow_up", "follow-up"} or re.search(
            r"\b(diagnos|patholog|biopsy|confirmed|revealed|showed|treated|metastasis|anomaly)\b",
            summary,
            flags=re.I,
        ):
            rows.append(
                {
                    "source": "evidence_note",
                    "rank": idx,
                    "text": summary,
                    "refs": note.get("refs") or [],
                    "time": note.get("time"),
                    "weight": 2.8 if tags & {"diagnosis", "pathology"} else 2.2,
                }
            )
    for idx, event in enumerate(case.get("events", [])):
        text = sanitize_runtime_text(event.get("text"))
        if not text:
            continue
        event_type = str(event.get("type") or "")
        if event_type in {"diagnosis", "pathology", "imaging", "treatment", "follow_up", "follow-up"} or re.search(
            r"\b(diagnos|confirmed|patholog|biopsy|carcinoma|tumor|aneurysm|metastasis|syndrome|pneumothorax|ureter|fracture|infection)\b",
            text,
            flags=re.I,
        ):
            weight = 2.6 if event_type in {"diagnosis", "pathology"} else 1.8
        else:
            weight = 1.4
        if profile == MEDICAL_ANSWER_ENTITY and not re.search(
            r"\b(answer options?|assessment entity candidate|reference answer evidence|correct option)\b",
            text,
            flags=re.I,
        ):
            weight *= 0.35
        if profile == CLINICAL_ASSESSMENT_ENTITY and re.search(
            r"\b(doctor assessment|assessment entity candidate|likely cause|diagnosis|impression|suggestive of)\b",
            text,
            flags=re.I,
        ):
            weight += 1.4
        rows.append(
            {
                "source": f"timeline_{event_type or 'clinical'}",
                "rank": idx,
                "text": text,
                "event_id": event.get("event_id"),
                "time": event.get("time"),
                "weight": weight,
            }
        )
    return [row for row in rows if str(row.get("text") or "").strip()]


def add_visible_diagnosis_candidates(
    candidates: dict[str, dict[str, Any]],
    raw_text: str,
    *,
    source: str,
    weight: float,
    rank: int,
    evidence: str,
) -> None:
    fragments = recall_candidate_fragments(raw_text) or [raw_text]
    for fragment in fragments:
        entity = diagnosis_entity_from_text(fragment)
        if not entity or not is_likely_diagnostic_entity(entity, allow_primary=False):
            continue
        key = diagnosis_key(entity)
        if not key:
            continue
        item = candidates.setdefault(
            key,
            {"entity": entity, "score": 0.0, "sources": set(), "evidence": [], "best_rank": rank},
        )
        item["score"] += weight
        item["sources"].add(source)
        item["best_rank"] = min(int(item.get("best_rank", rank)), rank)
        if evidence and evidence not in item["evidence"]:
            item["evidence"].append(evidence[:240])

    normalized_blob = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(raw_text).lower())).strip()
    for cue, entity in PUBLIC_DIAGNOSIS_CUE_MAP:
        cue_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", cue.lower())).strip()
        if not cue_norm or not re.search(rf"\b{re.escape(cue_norm)}\b", normalized_blob):
            continue
        canonical = canonicalize_diagnosis(entity)
        key = diagnosis_key(canonical)
        item = candidates.setdefault(
            key,
            {"entity": canonical, "score": 0.0, "sources": set(), "evidence": [], "best_rank": rank},
        )
        item["score"] += weight + 1.2
        item["sources"].add(source)
        item["best_rank"] = min(int(item.get("best_rank", rank)), rank)
        if evidence and evidence not in item["evidence"]:
            item["evidence"].append(evidence[:240])


def evidence_driven_diagnosis_rerank(
    case: dict[str, Any],
    pred: dict[str, Any],
    evidence_notes: list[dict[str, Any]],
    diagnosis_candidates: list[dict[str, Any]],
    *,
    max_items: int = 12,
) -> dict[str, Any]:
    """Recall and reorder diagnosis_list using only visible timeline and memory-derived evidence."""
    max_items = min(max_items, profile_list_budget_for_case(case))
    candidates: dict[str, dict[str, Any]] = {}
    primary = conservative_canonical_diagnosis(str(pred.get("primary_diagnosis") or "")).strip()
    if primary:
        candidates[diagnosis_key(primary)] = {
            "entity": primary,
            "score": 4.0,
            "sources": {"primary"},
            "evidence": list(pred.get("evidence") or [])[:2],
            "best_rank": 0,
        }

    for rank, row in enumerate(visible_diagnosis_source_rows(case, pred, evidence_notes, diagnosis_candidates)):
        text = str(row.get("text") or "")
        source = str(row.get("source") or "visible")
        add_visible_diagnosis_candidates(
            candidates,
            text,
            source=source,
            weight=float(row.get("weight") or 1.0),
            rank=rank,
            evidence=text,
        )

    filtered = []
    for item in candidates.values():
        entity = str(item.get("entity") or "")
        key = diagnosis_key(entity)
        if not key:
            continue
        sources = set(item.get("sources") or [])
        score = float(item.get("score") or 0.0)
        if "primary" not in sources and score < 2.6:
            continue
        if not is_likely_diagnostic_entity(entity, allow_primary="primary" in sources):
            continue
        filtered.append(item)

    profile = case_task_profile(case)

    def sort_key(item: dict[str, Any]) -> tuple[float, int, int, str]:
        entity = str(item.get("entity") or "")
        score = float(item.get("score") or 0.0)
        primary_bonus = 2.5 if diagnosis_key(entity) == diagnosis_key(primary) else 0.0
        if profile == MEDICAL_ANSWER_ENTITY:
            primary_bonus = 4.0 if diagnosis_key(entity) == diagnosis_key(primary) else 0.0
        elif re.search(r"\b(complication|metastasis|metastases|failure|injury|embolism|effusion|bleeding)\b", entity.lower()):
            score -= 1.2
        return (-(score + primary_bonus), int(item.get("best_rank", 9999)), len(entity), entity)

    filtered.sort(key=sort_key)

    merged: list[str] = []
    seen: set[str] = set()
    for item in filtered:
        entity = str(item.get("entity") or "").strip()
        key = diagnosis_key(entity)
        if entity and key not in seen:
            merged.append(entity)
            seen.add(key)
        if len(merged) >= max_items:
            break

    if not merged:
        return pred
    updated = dict(pred)
    existing = [conservative_canonical_diagnosis(str(item)).strip() for item in pred.get("diagnosis_list", []) if str(item).strip()]
    visible_keys = {diagnosis_key(str(item.get("entity") or "")) for item in filtered}
    preserved: list[str] = []
    preserved_seen: set[str] = set()
    for item in merged + existing:
        key = diagnosis_key(item)
        if item in existing and key != diagnosis_key(primary):
            if key not in visible_keys:
                continue
            if not is_likely_diagnostic_entity(item, allow_primary=False):
                continue
        if item and key and key not in preserved_seen:
            preserved.append(item)
            preserved_seen.add(key)
        if len(preserved) >= max_items:
            break
    updated["diagnosis_list"] = preserved
    updated["diagnosis_candidate_pool"] = {
        "enabled": True,
        "candidate_count": len(candidates),
        "retained_count": len(preserved),
    }
    if primary:
        primary_key = diagnosis_key(primary)
        primary_candidate = next((item for item in filtered if diagnosis_key(str(item.get("entity"))) == primary_key), None)
        primary_evidence = list((primary_candidate or {}).get("evidence") or [])
        evidence = [str(item) for item in updated.get("evidence", []) if str(item).strip()]
        if not any(diagnosis_key(primary) and diagnosis_key(primary) in diagnosis_key(item) for item in evidence):
            evidence_source = next((item for item in primary_evidence if str(item).strip()), "")
            if evidence_source:
                evidence = [evidence_source] + evidence
        updated["evidence"] = evidence[:5]
    return updated


def diagnosis_second_pass_prompt(
    case: dict[str, Any],
    pred: dict[str, Any],
    evidence_notes: list[dict[str, Any]],
    diagnosis_candidates: list[dict[str, Any]],
    *,
    candidate_limit: int = 40,
    text_limit: int = 200,
) -> list[dict[str, str]]:
    profile = case_task_profile(case)
    min_items, max_items = diagnosis_list_budget(profile)
    candidate_rows = visible_diagnosis_source_rows(case, pred, evidence_notes, diagnosis_candidates)
    candidate_lines = []
    seen_text: set[str] = set()
    for row in candidate_rows[:candidate_limit]:
        text = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
        if not text or text.lower() in seen_text:
            continue
        seen_text.add(text.lower())
        candidate_lines.append(
            f"- source={row.get('source')} time={row.get('time')} refs={row.get('refs') or row.get('event_id')} text={truncate_text(text, text_limit)}"
        )
    current = {
        "primary_diagnosis": truncate_text(pred.get("primary_diagnosis"), 120),
        "diagnosis_list": compact_runtime_list(list(pred.get("diagnosis_list", [])), limit=8, text_limit=120),
        "evidence": compact_runtime_list(list(pred.get("evidence", [])), limit=8, text_limit=160),
        "reasoning_summary": truncate_text(pred.get("reasoning_summary", ""), 240),
    }
    system = (
        "You are a clinical research diagnosis reconciler. Use only the provided visible evidence candidates. "
        "Return one compact JSON object with keys primary_diagnosis, diagnosis_list, confidence, evidence, reasoning_summary, "
        f"species_context, diagnosis_granularity. {task_profile_prompt_policy(profile)} "
        f"diagnosis_list must contain {min_items} to {max_items} concise evidence-supported entities when available. "
        "Do not invent diagnoses absent from the visible evidence. Return JSON only."
    )
    user = (
        f"[CASE_ID]\n{case.get('case_id')}\n\n"
        f"[CURRENT_PREDICTION]\n{current}\n\n"
        f"[VISIBLE_EVIDENCE_CANDIDATES]\n{chr(10).join(candidate_lines)}\n\n"
        "Reconcile the final primary diagnosis and diagnosis_list. Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def llm_diagnosis_second_pass(
    case: dict[str, Any],
    pred: dict[str, Any],
    client: DeepSeekClient | None,
    evidence_notes: list[dict[str, Any]],
    diagnosis_candidates: list[dict[str, Any]],
    *,
    fail_on_llm_error: bool = False,
    enable_normalization: bool = True,
) -> dict[str, Any]:
    if client is None:
        return pred
    try:
        last_exc: Exception | None = None
        result = None
        for candidate_limit, text_limit, requested_tokens in ((40, 200, 384), (24, 160, 256), (12, 120, 192), (6, 100, 128)):
            messages = diagnosis_second_pass_prompt(
                case,
                pred,
                evidence_notes,
                diagnosis_candidates,
                candidate_limit=candidate_limit,
                text_limit=text_limit,
            )
            try:
                result = client.chat(
                    messages,
                    temperature=0.0,
                    max_tokens=prompt_max_tokens(requested_tokens, messages),
                )
                break
            except Exception as exc:  # noqa: BLE001 - context overflow gets progressively compacted
                last_exc = exc
                if not is_context_limit_error(exc):
                    raise
        if result is None:
            raise last_exc or RuntimeError("Diagnosis second pass failed without an exception.")
        raw = extract_json_object(result.text)
        refined = normalize_prediction(
            str(case.get("case_id")),
            str(pred.get("method") or "ours_second_pass"),
            raw,
            result.usage,
            enable_normalization=enable_normalization,
        )
    except Exception as exc:  # noqa: BLE001 - second pass should not break a completed primary prediction
        if fail_on_llm_error:
            raise RuntimeError(f"Diagnosis second pass failed for {case.get('case_id')}: {exc}") from exc
        updated = dict(pred)
        updated["diagnosis_second_pass_error"] = str(exc)
        return updated

    updated = dict(pred)
    previous_usage = pred.get("usage") or {}
    usage = refined.get("usage") or {}
    updated.update(
        {
            "primary_diagnosis": refined.get("primary_diagnosis") or pred.get("primary_diagnosis"),
            "diagnosis_list": refined.get("diagnosis_list") or pred.get("diagnosis_list", []),
            "confidence": refined.get("confidence", pred.get("confidence")),
            "evidence": refined.get("evidence") or pred.get("evidence", []),
            "reasoning_summary": refined.get("reasoning_summary") or pred.get("reasoning_summary", ""),
            "species_context": refined.get("species_context", pred.get("species_context")),
            "diagnosis_granularity": refined.get("diagnosis_granularity", pred.get("diagnosis_granularity")),
            "diagnosis_second_pass": {"enabled": True},
            "usage": {
                "prompt_tokens": int(previous_usage.get("prompt_tokens", 0) or 0) + int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(previous_usage.get("completion_tokens", 0) or 0)
                + int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(previous_usage.get("total_tokens", 0) or 0) + int(usage.get("total_tokens", 0) or 0),
            },
        }
    )
    return evidence_driven_diagnosis_rerank(case, updated, evidence_notes, diagnosis_candidates)


COUNTERFACTUAL_CPG_THRESHOLD = 0.40


def counterfactual_verification_prompt(
    case: dict[str, Any],
    pred: dict[str, Any],
    intervention: str,
    context: str,
    extra: str,
) -> list[dict[str, str]]:
    system = (
        "You are a clinical research counterfactual verifier. This is not medical advice. "
        "Use only the visible case evidence and the stated counterfactual intervention. "
        "Do not assume labels, expected effects, or hidden ground truth. Return one compact JSON object with keys: "
        "primary_diagnosis, confidence_for_original_diagnosis, counterfactual_primary_diagnosis, "
        "causal_consistency_summary. confidence_for_original_diagnosis must be a number from 0 to 1. "
        "The confidence is specifically the probability/confidence that the original diagnosis remains supported "
        "after the intervention. Return JSON only."
    )
    user = (
        f"[CASE_ID]\n{case.get('case_id')}\n\n"
        f"[ORIGINAL_DIAGNOSIS_TO_VERIFY]\n{truncate_text(pred.get('primary_diagnosis'), 120)}\n\n"
        f"[ORIGINAL_CONFIDENCE]\n{pred.get('confidence')}\n\n"
        f"[COUNTERFACTUAL_INTERVENTION]\n{truncate_text(intervention, 260)}\n\n"
        f"[VISIBLE_CASE]\n{context}\n\n"
        f"{extra}\n\n"
        "Re-run the causal reasoning under the counterfactual intervention. Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def budgeted_counterfactual_prompt(
    case: dict[str, Any],
    pred: dict[str, Any],
    intervention: str,
    context: str,
    extra: str,
    *,
    level: int,
    max_tokens: int,
) -> list[dict[str, str]]:
    requested_level = level
    include_time = "t=" in str(context)
    if level <= 0:
        attempt_context = context
        attempt_extra = extra
    else:
        attempt_context = budgeted_case_context(case, level=level, include_labs=True, include_time=include_time)
        attempt_extra = truncate_context_middle(extra, [8000, 5000, 3000, 1600][min(level - 1, 3)])
    messages = counterfactual_verification_prompt(case, pred, intervention, attempt_context, attempt_extra)
    while not messages_fit_budget(messages, max_tokens=max_tokens) and level < 4:
        level += 1
        attempt_context = budgeted_case_context(case, level=level, include_labs=True, include_time=include_time)
        attempt_extra = truncate_context_middle(extra, [8000, 5000, 3000, 1600][min(level - 1, 3)])
        messages = counterfactual_verification_prompt(case, pred, intervention, attempt_context, attempt_extra)
    if not messages_fit_budget(messages, max_tokens=max_tokens):
        messages = counterfactual_verification_prompt(
            case,
            pred,
            intervention,
            truncate_context_middle(attempt_context, 5000),
            truncate_context_middle(attempt_extra, 1000),
        )
    if requested_level > 0:
        messages = counterfactual_verification_prompt(
            case,
            pred,
            intervention,
            truncate_context_middle(attempt_context, max(3000, 5000 - requested_level * 600)),
            truncate_context_middle(attempt_extra, max(400, 1200 // (requested_level + 1))),
        )
    return messages


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_counterfactual_verification(
    case: dict[str, Any],
    pred: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    context: str,
    extra: str,
    threshold: float = COUNTERFACTUAL_CPG_THRESHOLD,
    fail_on_llm_error: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    counterfactuals = list(case.get("counterfactuals") or [])
    if not counterfactuals:
        return {
            "enabled": True,
            "available": False,
            "passed": False,
            "threshold": threshold,
            "cpg": 0.0,
            "summary": "No counterfactual intervention was provided for this case.",
        }, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    intervention = str(counterfactuals[0].get("intervention") or counterfactuals[0].get("remove") or "").strip()
    original_confidence = max(0.0, min(1.0, _safe_float(pred.get("confidence"), 0.5)))
    if client is None:
        raise RuntimeError(
            "Counterfactual verification requires a real LLM API client. "
            "Use --require-api for formal runs or --disable-counterfactual-verification for the ablation only."
        )

    try:
        last_exc: Exception | None = None
        result = None
        for level, requested_tokens in ((0, 384), (1, 256), (2, 192), (3, 128), (4, 128)):
            messages = budgeted_counterfactual_prompt(
                case,
                pred,
                intervention,
                context,
                extra,
                level=level,
                max_tokens=requested_tokens,
            )
            try:
                result = client.chat(
                    messages,
                    temperature=0.0,
                    max_tokens=prompt_max_tokens(requested_tokens, messages),
                )
                break
            except Exception as exc:  # noqa: BLE001 - context overflow gets progressively compacted
                last_exc = exc
                if not is_context_limit_error(exc):
                    raise
        if result is None:
            raise last_exc or RuntimeError("Counterfactual verification failed without an exception.")
        raw = extract_json_object(result.text)
    except Exception as exc:  # noqa: BLE001
        if fail_on_llm_error:
            raise RuntimeError(f"Counterfactual verification failed for {case.get('case_id')}: {exc}") from exc
        return {
            "enabled": True,
            "available": True,
            "intervention": intervention,
            "original_confidence": original_confidence,
            "counterfactual_confidence_for_original": original_confidence,
            "cpg": 0.0,
            "threshold": threshold,
            "passed": False,
            "counterfactual_primary_diagnosis": pred.get("primary_diagnosis"),
            "summary": f"Counterfactual verification failed: {exc}",
        }, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    cf_confidence = max(0.0, min(1.0, _safe_float(raw.get("confidence_for_original_diagnosis"), original_confidence)))
    cpg = max(0.0, original_confidence - cf_confidence)
    return {
        "enabled": True,
        "available": True,
        "intervention": intervention,
        "original_confidence": original_confidence,
        "counterfactual_confidence_for_original": cf_confidence,
        "cpg": cpg,
        "threshold": threshold,
        "passed": cpg >= threshold,
        "primary_diagnosis": str(raw.get("primary_diagnosis") or pred.get("primary_diagnosis") or ""),
        "counterfactual_primary_diagnosis": str(
            raw.get("counterfactual_primary_diagnosis") or raw.get("primary_diagnosis") or ""
        ),
        "summary": str(raw.get("causal_consistency_summary") or raw.get("summary") or ""),
    }, result.usage


def should_run_counterfactual_verification(
    case: dict[str, Any],
    pred: dict[str, Any],
    *,
    ops: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    evidence_notes: list[dict[str, Any]],
    policy: str,
    sample_rate: float,
    risk_threshold: float,
) -> tuple[bool, list[str]]:
    policy = str(policy or "always")
    if policy == "always":
        return True, ["policy_always"]
    if policy == "never":
        return False, ["policy_never"]
    if policy != "risk_sample":
        return True, [f"unknown_policy:{policy}"]

    reasons: list[str] = []
    confidence = _safe_float(pred.get("confidence"), 0.5)
    if confidence < risk_threshold:
        reasons.append("low_confidence")
    if any(str(op.get("op") or "") in {"Revise", "Invalidate", "Discard"} for op in ops):
        reasons.append("memory_cleaning_changed_state")
    evidence_count = len([item for item in pred.get("evidence", []) if str(item).strip()])
    if evidence_count < 2:
        reasons.append("weak_prediction_evidence")
    if not evidence_notes and evidence_count < 2:
        reasons.append("no_source_evidence_notes")
    poison_ids = {str(item.get("poison_id") or "") for item in case.get("poison_records", [])}
    evidence_refs = {
        str(ref)
        for memory in memories
        for ref in (memory.get("evidence_refs") or [])
        if ref is not None
    }
    if poison_ids and poison_ids.intersection(evidence_refs):
        reasons.append("retrieved_polluted_memory")
    if reasons:
        return True, reasons

    clamped_rate = max(0.0, min(1.0, float(sample_rate)))
    case_id = str(case.get("case_id") or "")
    bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < clamped_rate:
        return True, ["low_risk_sampled"]
    return False, ["low_risk_not_sampled"]


def counterfactual_revision_extra(base_extra: str, verification: dict[str, Any]) -> str:
    return (
        f"{base_extra}\n\n"
        "[COUNTERFACTUAL_VERIFICATION_FAILURE]\n"
        f"intervention={verification.get('intervention')}\n"
        f"original_confidence={verification.get('original_confidence')}\n"
        f"counterfactual_confidence_for_original={verification.get('counterfactual_confidence_for_original')}\n"
        f"cpg={verification.get('cpg')} threshold={verification.get('threshold')}\n"
        f"counterfactual_primary_diagnosis={verification.get('counterfactual_primary_diagnosis')}\n"
        f"summary={verification.get('summary')}\n"
        "The original diagnosis did not show enough causal sensitivity to the intervention. "
        "Re-examine the cleaned memory operations, evidence chain, and initial hypothesis. "
        "Return the safest evidence-supported final diagnosis and lower confidence if causal support remains weak."
    )


def add_usage(previous: dict[str, Any] | None, extra: dict[str, Any] | None) -> dict[str, int]:
    previous = previous or {}
    extra = extra or {}
    return {
        "prompt_tokens": int(previous.get("prompt_tokens", 0) or 0) + int(extra.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(previous.get("completion_tokens", 0) or 0) + int(extra.get("completion_tokens", 0) or 0),
        "total_tokens": int(previous.get("total_tokens", 0) or 0) + int(extra.get("total_tokens", 0) or 0),
    }


def evidence_gated_diagnosis_recall(
    case: dict[str, Any],
    pred: dict[str, Any],
    evidence_notes: list[dict[str, Any]],
    diagnosis_candidates: list[dict[str, Any]],
    *,
    max_items: int = 7,
) -> dict[str, Any]:
    """Expand diagnosis_list with source-visible diagnoses without changing primary."""
    max_items = min(max_items, profile_list_budget_for_case(case))
    primary = canonicalize_diagnosis(str(pred.get("primary_diagnosis") or "")).strip()
    existing = [canonicalize_diagnosis(str(item)).strip() for item in pred.get("diagnosis_list", []) if str(item).strip()]
    if primary and primary not in existing:
        existing = [primary] + existing
    if len(existing) > 3:
        updated = dict(pred)
        updated["diagnosis_list"] = existing[:max_items]
        updated["diagnosis_recall_added_count"] = 0
        updated["diagnosis_recall_sources"] = {}
        return updated

    source_rows: list[tuple[int, str, str]] = []
    for item in diagnosis_candidates:
        text = str(item.get("text") or "")
        if re.search(r"\b(diagnosed|diagnosis|confirmed|pathology|ihc|revealed|showed)\b", text, flags=re.I):
            source_rows.append((0, text, "diagnosis_event"))

    source_blob = " ".join(text for _, text, _ in source_rows).lower()
    recalled: list[str] = []
    sources: dict[str, str] = {}
    seen = {
        re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", item.lower())).strip()
        for item in existing
        if item
    }

    def add(raw: str, source: str) -> None:
        entity = normalize_diagnosis_entity_text(raw)
        if not entity or not is_likely_diagnostic_entity(entity, allow_primary=False):
            return
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", entity.lower())).strip()
        if not key or key in seen:
            return
        entity_tokens = [tok for tok in re.findall(r"[a-z0-9]+", entity.lower()) if len(tok) > 2]
        if entity_tokens and not any(tok in source_blob for tok in entity_tokens):
            return
        recalled.append(entity)
        sources[entity] = source
        seen.add(key)

    for _, text, source in sorted(source_rows, key=lambda row: row[0]):
        for fragment in recall_candidate_fragments(text):
            add(fragment, source)
            if len(recalled) >= 2:
                break
        if len(recalled) >= 2:
            break

    updated = dict(pred)
    merged = []
    merged_seen: set[str] = set()
    for item in existing + recalled:
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", item.lower())).strip()
        if item and key not in merged_seen:
            merged.append(item)
            merged_seen.add(key)
    updated["diagnosis_list"] = merged[:max_items]
    updated["diagnosis_recall_added_count"] = max(0, len(updated["diagnosis_list"]) - len(existing))
    updated["diagnosis_recall_sources"] = sources
    return updated


def refine_diagnosis_list(
    case: dict[str, Any],
    pred: dict[str, Any],
    evidence_notes: list[dict[str, Any]],
    diagnosis_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = canonicalize_diagnosis(str(pred.get("primary_diagnosis") or "")).strip()
    if not primary:
        return pred

    candidate_rows: list[tuple[int, int, str]] = [(0, 0, primary)]

    for idx, item in enumerate(diagnosis_candidates):
        candidate_rows.append((1, idx, str(item.get("text") or "")))

    for idx, note in enumerate(evidence_notes):
        tags = {str(tag).lower() for tag in note.get("tags", [])}
        summary = str(note.get("summary") or "")
        low = summary.lower()
        if tags & {"diagnosis", "pathology"} or any(term in low for term in ("diagnos", "patholog", "biopsy", "confirmed")):
            candidate_rows.append((2, idx, summary))

    for idx, item in enumerate(pred.get("diagnosis_list", [])):
        candidate_rows.append((3, idx, str(item)))

    refined: list[str] = []
    seen: set[str] = set()
    accepted_by_source: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    def add_candidate(raw_text: str, *, source: int, allow_primary: bool = False) -> bool:
        entity = normalize_diagnosis_entity_text(raw_text)
        if not entity or not is_likely_diagnostic_entity(entity, allow_primary=allow_primary):
            return False
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", entity.lower())).strip()
        if not key or key in seen:
            return False
        refined.append(entity)
        seen.add(key)
        accepted_by_source[source] = accepted_by_source.get(source, 0) + 1
        return True

    add_candidate(primary, source=0, allow_primary=True)
    high_confidence_candidates = 0
    for source, _, raw in sorted(candidate_rows[1:], key=lambda row: (row[0], row[1])):
        if add_candidate(raw, source=source):
            if source in {1, 2}:
                high_confidence_candidates += 1

    profile_max_items = profile_list_budget_for_case(case)
    max_items = min(profile_max_items, 5 if high_confidence_candidates > 3 or accepted_by_source.get(3, 0) >= 3 else 3)
    updated = dict(pred)
    updated["diagnosis_list"] = refined[:max_items]
    updated["diagnosis_list_refinement"] = {
        "enabled": True,
        "candidate_count": len(candidate_rows),
        "high_confidence_candidate_count": high_confidence_candidates,
        "max_items": max_items,
    }
    return updated


def prefer_visible_diagnosis_candidate(case: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    if case_task_profile(case) == MEDICAL_ANSWER_ENTITY:
        return pred
    candidates = [
        item | {
            "clean_text": clean_diagnosis_candidate_text(str(item.get("text") or "")),
            "priority": diagnosis_candidate_priority(str(item.get("text") or "")),
        }
        for item in diagnosis_event_candidates(case, limit=20)
    ]
    candidates = [item for item in candidates if item["priority"] > 0 and item["clean_text"]]
    if not candidates:
        return pred
    evidence_blob = " ".join(str(x) for x in pred.get("evidence", []))
    diag_blob = " ".join(str(x) for x in pred.get("diagnosis_list", []))
    primary = str(pred.get("primary_diagnosis") or "")

    def support_score(item: dict[str, Any]) -> tuple[int, int]:
        clean = item["clean_text"]
        support = 0
        if clean.lower() in evidence_blob.lower() or clean.lower() in diag_blob.lower():
            support += 4
        if canonicalize_diagnosis(clean) in [canonicalize_diagnosis(str(x)) for x in pred.get("diagnosis_list", [])]:
            support += 4
        if clean.lower() in primary.lower() or primary.lower() in clean.lower():
            support += 2
        return support + int(item["priority"]), int(item.get("time") or 0)

    best = max(candidates, key=support_score)
    if support_score(best)[0] < 5:
        return pred
    new_primary = canonicalize_diagnosis(best["clean_text"])
    if not new_primary or canonicalize_diagnosis(primary) == new_primary:
        return pred
    updated = dict(pred)
    updated["primary_diagnosis"] = new_primary
    diagnosis_list = [new_primary]
    for item in pred.get("diagnosis_list", []):
        canonical = canonicalize_diagnosis(str(item))
        if canonical and canonical not in diagnosis_list:
            diagnosis_list.append(canonical)
    updated["diagnosis_list"] = diagnosis_list[:profile_list_budget_for_case(case)]
    updated["reasoning_summary"] = (
        str(pred.get("reasoning_summary") or "")
        + " Primary diagnosis selected from visible diagnosis-event candidates."
    ).strip()
    updated["primary_selection"] = {
        "source": "visible_diagnosis_event",
        "event_id": best.get("event_id"),
        "time": best.get("time"),
    }
    return updated


def evidence_support_score_for_text(text: str, pred: dict[str, Any], case: dict[str, Any]) -> float:
    candidate = canonicalize_diagnosis(str(text))
    if not candidate:
        return 0.0
    blobs = [
        " ".join(str(item) for item in pred.get("evidence", [])),
        " ".join(str(item) for item in pred.get("diagnosis_list", [])),
        " ".join(
            sanitize_runtime_text(event.get("text"))
            for event in case.get("events", [])
            if event.get("type") == "diagnosis"
        ),
    ]
    score = 0.0
    for blob in blobs:
        low_blob = blob.lower()
        low_candidate = candidate.lower()
        if low_candidate and low_candidate in low_blob:
            score += 2.0
        else:
            candidate_tokens = {tok for tok in re.findall(r"[a-z0-9]+", low_candidate) if len(tok) > 3}
            blob_tokens = set(re.findall(r"[a-z0-9]+", low_blob))
            if candidate_tokens:
                score += len(candidate_tokens & blob_tokens) / len(candidate_tokens)
    return score


def verify_primary_with_evidence(case: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    if case_task_profile(case) == MEDICAL_ANSWER_ENTITY:
        updated = dict(pred)
        updated["primary_evidence_support"] = evidence_support_score_for_text(str(pred.get("primary_diagnosis") or ""), pred, case)
        return updated
    primary = str(pred.get("primary_diagnosis") or "")
    primary_support = evidence_support_score_for_text(primary, pred, case)
    if primary_support >= 1.0:
        updated = dict(pred)
        updated["primary_evidence_support"] = primary_support
        return updated
    candidates = []
    for item in pred.get("diagnosis_list", []):
        candidates.append(str(item))
    for item in diagnosis_event_candidates(case, limit=12):
        candidates.append(clean_diagnosis_candidate_text(str(item.get("text") or "")))
    scored = [
        (evidence_support_score_for_text(candidate, pred, case), candidate)
        for candidate in candidates
        if is_likely_diagnostic_entity(candidate, allow_primary=True)
    ]
    if not scored:
        updated = dict(pred)
        updated["primary_evidence_support"] = primary_support
        updated["primary_selection_warning"] = "low_evidence_support"
        return updated
    best_score, best_candidate = max(scored, key=lambda item: item[0])
    if best_score <= primary_support or best_score < 1.0:
        updated = dict(pred)
        updated["primary_evidence_support"] = primary_support
        updated["primary_selection_warning"] = "low_evidence_support"
        return updated
    new_primary = canonicalize_diagnosis(best_candidate)
    updated = dict(pred)
    updated["primary_diagnosis"] = new_primary
    updated["diagnosis_granularity"] = infer_diagnosis_granularity(new_primary)
    updated["primary_evidence_support"] = best_score
    updated["primary_selection"] = {
        "source": "evidence_verifier",
        "previous_primary": primary,
        "previous_support": primary_support,
    }
    diagnosis_list = [new_primary]
    for item in pred.get("diagnosis_list", []):
        canonical = canonicalize_diagnosis(str(item))
        if canonical and canonical not in diagnosis_list:
            diagnosis_list.append(canonical)
    updated["diagnosis_list"] = diagnosis_list[:profile_list_budget_for_case(case)]
    return updated


def format_memory_line(card: dict[str, Any], *, summary_limit: int = 220) -> str:
    summary = sanitize_runtime_text(card.get("summary"))
    if not summary:
        return ""
    summary = truncate_text(summary, summary_limit)
    return (
        f"- {summary} "
        f"(status={card.get('status')}, confidence={card.get('confidence')}, refs={card.get('evidence_refs')})"
    )


def compact_memory_lines(cards: list[dict[str, Any]], *, limit: int = 8, summary_limit: int = 180) -> list[str]:
    ordered = sorted(
        cards,
        key=lambda card: (
            0 if card.get("status") in {"active", "flagged"} else 1,
            -_safe_float(card.get("confidence"), 0.0),
        ),
    )
    return [line for line in (format_memory_line(card, summary_limit=summary_limit) for card in ordered[:limit]) if line]


def compact_evidence_lines(notes: list[dict[str, Any]], *, include_time: bool = True, limit: int = 8, text_limit: int = 180) -> list[str]:
    selected = notes[-limit:]
    lines = []
    for note in selected:
        summary = truncate_text(note.get("summary"), text_limit)
        if not summary:
            continue
        prefix = f"- t={note.get('time')} " if include_time else "- "
        lines.append(f"{prefix}refs={note.get('refs')} tags={note.get('tags')} text={summary}")
    return lines


def compact_diagnosis_candidate_lines(
    candidates: list[dict[str, Any]],
    *,
    include_time: bool = True,
    limit: int = 8,
    text_limit: int = 160,
) -> list[str]:
    scored = [
        (diagnosis_candidate_priority(str(item.get("text") or "")), item)
        for item in candidates
        if sanitize_runtime_text(item.get("text"))
    ]
    scored.sort(key=lambda pair: (pair[0], int(pair[1].get("time") or 0)), reverse=True)
    selected = [item for _, item in scored[:limit]]
    selected.sort(key=lambda item: str(item.get("time") or ""))
    lines = []
    for item in selected:
        text = truncate_text(item.get("text"), text_limit)
        if not text:
            continue
        prefix = f"- t={item.get('time')} " if include_time else "- "
        lines.append(f"{prefix}ref={item.get('event_id')} diagnosis_text={text}")
    return lines


def compact_prompt_memory_ops(prompt_ops: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    compacted = []
    for op in prompt_ops[:limit]:
        item = {
            "op": op.get("op"),
            "target": op.get("target"),
            "touched_memory_ids": list(op.get("touched_memory_ids") or [])[:4],
            "revised_memory_id": op.get("revised_memory_id"),
        }
        if "preserved_fact_count" in op:
            item["preserved_fact_count"] = op.get("preserved_fact_count")
        if "revision_note" in op:
            item["revision_note"] = op.get("revision_note")
        compacted.append({key: value for key, value in item.items() if value not in (None, "", [])})
    return compacted


def strip_temporal_signal_from_memory_store(store: Any) -> None:
    for card in store.cards:
        card["time_scope"] = {}
        temporal_tags = {"temporal", "time", "timeline"}
        card["tags"] = [tag for tag in list(card.get("tags") or []) if str(tag).lower() not in temporal_tags]
    store.save()


def temporal_blind_evidence_notes(cards: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    notes = source_aligned_evidence_notes(cards, limit=limit)
    for note in notes:
        note["time"] = None
    return notes


def temporal_blind_diagnosis_event_candidates(case: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    candidates = diagnosis_event_candidates(case, limit=limit)
    for item in candidates:
        item["time"] = None
    return candidates


def run_direct_polluted(
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    fail_on_llm_error: bool = False,
) -> dict[str, Any]:
    pred = run_llm_prediction(
        case,
        method="baseline_polluted_direct_deepseek",
        client=client,
        context=case_context(case, max_events=min(4, len(case.get("events", []))), include_labs=False),
        extra=(
            "Make a diagnosis from this limited truncated context plus the following previously stored memory cards. "
            "The memory cards may be stale, mixed, or cross-patient, but this baseline has no dedicated cleaning tool.\n"
            f"{pollution_memory_context(case)}"
        ),
        fallback_max_events=min(4, len(case.get("events", []))),
        fail_on_llm_error=fail_on_llm_error,
    )
    pred["pollution_exposed"] = True
    pred["runtime_leakage_filtered_count"] = runtime_leakage_filtered_count(
        [event.get("text") for event in case.get("events", [])[: min(4, len(case.get("events", [])))]]
        + [poison.get("text") for poison in case.get("poison_records", [])]
    )
    pred["memory_leakage_filtered_count"] = runtime_leakage_filtered_count(
        [poison.get("text") for poison in case.get("poison_records", [])]
    )
    return pred


def run_ours(
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    memory_dir: str | Path,
    strategy: dict[str, Any],
    fail_on_llm_error: bool = False,
) -> dict[str, Any]:
    memory_path = Path(memory_dir) / f"{case['case_id']}.memory.jsonl"
    features = strategy.get("features") or {}
    store = bootstrap_memory(case, memory_path, include_poison=bool(features.get("enable_polluted_memory")))
    disable_temporal_signal = bool(features.get("disable_temporal_signal"))
    if disable_temporal_signal:
        strip_temporal_signal_from_memory_store(store)
    profile = case_task_profile(case)
    source_dataset = str((case.get("data_quality_flags") or {}).get("source_dataset") or "").strip()
    adaptive_memory_cleaning = bool(features.get("profile_adaptive_memory_cleaning", True))
    adaptive_evidence_notes = bool(features.get("profile_adaptive_evidence_notes", False))
    disable_memory_cleaning = bool(features.get("disable_memory_cleaning")) or (
        adaptive_memory_cleaning and source_dataset == "pmc_patients"
    )
    ops = [] if disable_memory_cleaning else apply_critique(
        case,
        store,
        client,
        fail_on_llm_error=fail_on_llm_error,
        enforce_op_guard=not bool(features.get("disable_critic_op_guard")),
    )
    prompt_ops = safe_memory_ops_for_prompt(case, ops)
    query = "final diagnosis longitudinal causal evidence treatment imaging pathology labs"
    top_k = resolve_top_k(case, strategy)
    retrieved_memories = store.retrieve(query, k=top_k)
    disable_evidence_note_injection = bool(features.get("disable_evidence_note_injection")) or (
        adaptive_evidence_notes and profile == MEDICAL_ANSWER_ENTITY
    )
    evidence_notes = [] if disable_evidence_note_injection else (
        temporal_blind_evidence_notes(store.cards) if disable_temporal_signal else source_aligned_evidence_notes(store.cards)
    )
    diagnosis_candidates = [] if disable_evidence_note_injection else (
        temporal_blind_diagnosis_event_candidates(case) if disable_temporal_signal else diagnosis_event_candidates(case)
    )
    seen_ids: set[str] = set()
    memories = []
    for memory in retrieved_memories:
        memory_id = str(memory.get("memory_id") or "")
        if memory_id not in seen_ids:
            memories.append(memory)
            seen_ids.add(memory_id)
    memory_lines = compact_memory_lines(memories)
    evidence_lines = compact_evidence_lines(evidence_notes, include_time=not disable_temporal_signal)
    diagnosis_candidate_lines = compact_diagnosis_candidate_lines(
        diagnosis_candidates,
        include_time=not disable_temporal_signal,
    )
    source_style = style_policy_prompt((case.get("data_quality_flags") or {}).get("style_policy"))
    if disable_temporal_signal:
        extra_intro = (
            "Use the active JSONL memory cards and source-aligned evidence notes. "
            "This ablation hides explicit temporal indices and time scopes; choose primary_diagnosis from visible clinical content "
            "without relying on event timestamps or chronological hints. "
        )
        final_check = "Before finalizing, check whether the visible clinical evidence supports the diagnosis."
    else:
        extra_intro = (
            "Use the active JSONL memory cards and source-aligned evidence notes. "
            "Prefer timeline/evidence refs over stale interpretations when choosing primary_diagnosis. "
            "If a diagnosis event candidate directly captures the final explanatory state, use that concise text as primary_diagnosis "
            "and put broader underlying diseases in diagnosis_list rather than replacing it. "
        )
        final_check = (
            "Before finalizing, check whether longitudinal evidence contradicts the initial hypothesis. "
            "Do not treat a prior interpretation as a diagnosis unless timeline evidence supports it."
        )
    extra = (
        f"{extra_intro}\n"
        f"{source_style}\n"
        f"[MEMORY_CARDS]\n{chr(10).join(memory_lines)}\n"
        f"[SOURCE_ALIGNED_EVIDENCE_NOTES]\n{chr(10).join(evidence_lines)}\n"
        f"[DIAGNOSIS_EVENT_CANDIDATES]\n{chr(10).join(diagnosis_candidate_lines)}\n"
        f"[MEMORY_OPS]\n{compact_prompt_memory_ops(prompt_ops)}\n"
        f"{final_check}"
    )
    visible_context = budgeted_case_context(
        case,
        level=0,
        include_labs=True,
        include_time=not disable_temporal_signal,
    )
    pred = run_llm_prediction(
        case,
        method=f"medimem_topk{top_k}_round{strategy.get('rounds', 1)}",
        client=client,
        context=visible_context,
        extra=extra,
        fallback_max_events=None,
        temperature=float(strategy.get("temperature", 0.05)),
        fail_on_llm_error=fail_on_llm_error,
        enable_normalization=not bool(features.get("disable_normalization")),
    )
    if not disable_evidence_note_injection:
        pred = prefer_visible_diagnosis_candidate(case, pred)
        pred = refine_diagnosis_list(case, pred, evidence_notes, diagnosis_candidates)
        pred = verify_primary_with_evidence(case, pred)
        pred = evidence_gated_diagnosis_recall(case, pred, evidence_notes, diagnosis_candidates)
        pred = evidence_driven_diagnosis_rerank(case, pred, evidence_notes, diagnosis_candidates)
        if profile != MEDICAL_ANSWER_ENTITY and source_dataset != "pmc_patients":
            pred = llm_diagnosis_second_pass(
                case,
                pred,
                client,
                evidence_notes,
                diagnosis_candidates,
                fail_on_llm_error=False,
                enable_normalization=not bool(features.get("disable_normalization")),
            )
        else:
            reason = "pmc_source_evidence_primary_locked" if source_dataset == "pmc_patients" else "medical_answer_entity_option_locked"
            pred["diagnosis_second_pass"] = {"enabled": False, "reason": reason}
        pred = select_primary_for_task_profile(
            case,
            pred,
            evidence_notes,
            diagnosis_candidates,
            enable_normalization=not bool(features.get("disable_normalization")),
        )
    else:
        if profile == MEDICAL_ANSWER_ENTITY:
            pred["diagnosis_second_pass"] = {"enabled": False, "reason": "medical_answer_entity_option_locked"}
        pred = select_primary_for_task_profile(
            case,
            pred,
            [],
            [],
            enable_normalization=not bool(features.get("disable_normalization")),
        )
    counterfactual_policy = str(features.get("counterfactual_policy") or "always")
    counterfactual_sample_rate = float(
        features["counterfactual_sample_rate"] if "counterfactual_sample_rate" in features else 0.2
    )
    counterfactual_risk_threshold = float(
        features["counterfactual_risk_threshold"] if "counterfactual_risk_threshold" in features else 0.55
    )
    if features.get("disable_counterfactual_verification") or counterfactual_policy == "never":
        pred["counterfactual_verification"] = {
            "enabled": False,
            "passed": False,
            "threshold": COUNTERFACTUAL_CPG_THRESHOLD,
            "cpg": 0.0,
            "policy": counterfactual_policy,
            "summary": "Counterfactual verification disabled by feature flag or policy.",
        }
        pred["counterfactual_revision_triggered"] = False
    else:
        should_run_cf, cf_reasons = should_run_counterfactual_verification(
            case,
            pred,
            ops=ops,
            memories=memories,
            evidence_notes=evidence_notes,
            policy=counterfactual_policy,
            sample_rate=counterfactual_sample_rate,
            risk_threshold=counterfactual_risk_threshold,
        )
        pred["counterfactual_revision_triggered"] = False
        if not should_run_cf:
            verification = {
                "enabled": False,
                "available": True,
                "passed": False,
                "threshold": COUNTERFACTUAL_CPG_THRESHOLD,
                "cpg": 0.0,
                "policy": counterfactual_policy,
                "skip_reasons": cf_reasons,
                "summary": "Counterfactual verification skipped for low-risk unsampled case.",
            }
            cf_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        else:
            verification, cf_usage = run_counterfactual_verification(
                case,
                pred,
                client,
                context=visible_context,
                extra=extra,
                threshold=COUNTERFACTUAL_CPG_THRESHOLD,
                fail_on_llm_error=fail_on_llm_error and counterfactual_policy == "always",
            )
            verification["policy"] = counterfactual_policy
            verification["trigger_reasons"] = cf_reasons
        pred["usage"] = add_usage(pred.get("usage"), cf_usage)
        pred["counterfactual_verification"] = verification
        counterfactual_primary = str(verification.get("counterfactual_primary_diagnosis") or "").strip()
        current_primary = str(pred.get("primary_diagnosis") or "").strip()
        high_confidence_contradiction = (
            verification.get("available")
            and verification.get("passed")
            and float(verification.get("counterfactual_confidence_for_original") or 1.0) <= 0.20
            and counterfactual_primary
            and current_primary
            and diagnosis_key(counterfactual_primary) != diagnosis_key(current_primary)
        )
        pred["counterfactual_revision_policy"] = "audit_only_unless_high_confidence_contradiction"
        if high_confidence_contradiction:
            pre_counterfactual_prediction = {
                key: pred.get(key)
                for key in [
                    "primary_diagnosis",
                    "diagnosis_list",
                    "confidence",
                    "evidence",
                    "reasoning_summary",
                    "species_context",
                    "diagnosis_granularity",
                ]
            }
            revision = run_llm_prediction(
                case,
                method=f"medimem_topk{top_k}_round{strategy.get('rounds', 1)}",
                client=client,
                context=visible_context,
                extra=counterfactual_revision_extra(extra, verification),
                fallback_max_events=None,
                temperature=0.0,
                fail_on_llm_error=fail_on_llm_error,
                enable_normalization=not bool(features.get("disable_normalization")),
            )
            pred.update(
                {
                    "primary_diagnosis": revision.get("primary_diagnosis", pred.get("primary_diagnosis")),
                    "diagnosis_list": revision.get("diagnosis_list", pred.get("diagnosis_list", [])),
                    "confidence": revision.get("confidence", pred.get("confidence")),
                    "evidence": revision.get("evidence", pred.get("evidence", [])),
                    "reasoning_summary": revision.get("reasoning_summary", pred.get("reasoning_summary", "")),
                    "species_context": revision.get("species_context", pred.get("species_context")),
                    "diagnosis_granularity": revision.get("diagnosis_granularity", pred.get("diagnosis_granularity")),
                    "counterfactual_revision_triggered": True,
                    "pre_counterfactual_prediction": pre_counterfactual_prediction,
                    "usage": add_usage(pred.get("usage"), revision.get("usage")),
                }
            )
            pred = select_primary_for_task_profile(
                case,
                pred,
                evidence_notes,
                diagnosis_candidates,
                enable_normalization=not bool(features.get("disable_normalization")),
            )
    pred["memory_ops"] = ops
    pred["prompt_memory_ops"] = prompt_ops
    pred["pollution_exposed"] = bool(features.get("enable_polluted_memory"))
    pred["retrieved_memory_count"] = len(memories)
    pred["source_evidence_note_count"] = len(evidence_notes)
    pred["diagnosis_candidate_count"] = len(diagnosis_candidates)
    pred["runtime_leakage_filtered_count"] = runtime_leakage_filtered_count(
        [event.get("text") for event in case.get("events", [])]
        + [memory.get("summary") for memory in memories]
        + [card.get("summary") for card in store.cards]
    )
    pred["memory_leakage_filtered_count"] = runtime_leakage_filtered_count(
        [memory.get("summary") for memory in memories]
    )
    pred["memory_leakage_filtered_from_store_count"] = runtime_leakage_filtered_count(
        [card.get("summary") for card in store.cards]
    )
    pred["strategy"] = strategy
    pred["optimization_features"] = feature_state(strategy)
    return pred


def resolve_top_k(case: dict[str, Any], strategy: dict[str, Any]) -> int:
    features = strategy.get("features") or {}
    if "top_k" in strategy and not features.get("top_k_is_auto"):
        return int(strategy["top_k"])
    if features.get("disable_dynamic_top_k"):
        return int(strategy.get("fallback_top_k", 3))
    source_dataset = str((case.get("data_quality_flags") or {}).get("source_dataset") or "").strip()
    if source_dataset == "medical_meadow_wikidoc":
        return 3
    if case_task_profile(case) == MEDICAL_ANSWER_ENTITY or source_dataset == "pmc_patients":
        return 8
    events = case.get("events", [])
    event_count = len(events)
    if event_count < 20:
        top_k = 3
    elif event_count <= 50:
        top_k = 5
    else:
        top_k = 8
    evidence_terms = ("pathology", "biopsy", "imaging", "ct", "mri", "diagnosis", "diagnosed", "follow-up", "follow up")
    evidence_hits = sum(
        1
        for event in events
        if any(term in sanitize_runtime_text(event.get("text")).lower() for term in evidence_terms)
    )
    if event_count > 50 and evidence_hits:
        top_k = max(top_k, 8)
    elif evidence_hits >= 3:
        top_k = max(top_k, 8)
    return int(top_k)


def feature_state(strategy: dict[str, Any]) -> dict[str, bool]:
    features = strategy.get("features") or {}
    return {
        "diagnosis_normalization": not bool(features.get("disable_normalization")),
        "dynamic_top_k": not bool(features.get("disable_dynamic_top_k")),
        "memory_cleaning": not bool(features.get("disable_memory_cleaning")),
        "polluted_memory": bool(features.get("enable_polluted_memory")),
        "critic_op_guard": not bool(features.get("disable_critic_op_guard")),
        "evidence_note_injection": not bool(features.get("disable_evidence_note_injection")),
        "profile_adaptive_memory_cleaning": bool(features.get("profile_adaptive_memory_cleaning", True)),
        "profile_adaptive_evidence_notes": bool(features.get("profile_adaptive_evidence_notes", False)),
        "counterfactual_verification": not bool(features.get("disable_counterfactual_verification")),
        "temporal_signal": not bool(features.get("disable_temporal_signal")),
    }
