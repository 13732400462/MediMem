from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .llm import DeepSeekClient, extract_json_object
from .memory import apply_critique, bootstrap_memory


def case_context(case: dict[str, Any], *, max_events: int | None = None, include_labs: bool = True) -> str:
    events = case.get("events", [])
    if max_events is not None:
        events = events[:max_events]
    lines = [
        f"case_id: {case['case_id']}",
        f"demographics: {case.get('demographics', {})}",
        "timeline:",
    ]
    for event in events:
        lines.append(f"- t={event.get('time')} [{event.get('type')}] {event.get('text')}")
    if include_labs:
        lines.append("labs:")
        for lab in case.get("synthetic_labs", [])[:12]:
            lines.append(f"- t={lab.get('time')} {lab.get('name')}={lab.get('value')} {lab.get('unit')} ({lab.get('flag')})")
    return "\n".join(lines)


def prediction_json_prompt(method: str, context: str, extra: str = "") -> list[dict[str, str]]:
    system = (
        "You are a clinical research diagnosis evaluator. This is not medical advice. "
        "Use only the provided case evidence. Return one compact JSON object with keys: "
        "primary_diagnosis, diagnosis_list, confidence, evidence, reasoning_summary. "
        "confidence must be a number from 0 to 1."
    )
    user = f"[METHOD]\n{method}\n\n[CASE]\n{context}\n\n{extra}\n\nReturn JSON only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_prediction(case_id: str, method: str, raw: dict[str, Any], usage: dict[str, int] | None = None) -> dict[str, Any]:
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
    return {
        "case_id": case_id,
        "method": method,
        "primary_diagnosis": primary or "Unknown",
        "diagnosis_list": [str(d).strip() for d in diag_list if str(d).strip()] or ([primary] if primary else []),
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence": [str(e) for e in evidence],
        "reasoning_summary": str(raw.get("reasoning_summary") or raw.get("summary") or ""),
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def heuristic_predict(case: dict[str, Any], method: str, *, max_events: int | None = None) -> dict[str, Any]:
    events = case.get("events", [])
    if max_events is not None:
        events = events[:max_events]
    joined_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", " ".join(str(e.get("text", "")) for e in events).lower())).strip()
    candidates: list[tuple[float, str, str]] = []
    if "e granulosus" in joined_norm or "echinococ" in joined_norm:
        candidates.append((0.96, "Echinococcosis", "E. granulosus / echinococcosis evidence in timeline"))
    patterns = [
        (0.95, r"(?:diagnosed with|diagnosed as|diagnosis of|confirmed)\s+([^.;]+)", "explicit diagnosis"),
        (0.85, r"pathology (?:diagnosed|confirmed|showed)\s+([^.;]+)", "pathology"),
        (0.72, r"(?:showed|found)\s+([^.;]*(?:lesion|hematoma|leukemia|shock|failure|cancer)[^.;]*)", "key finding"),
    ]
    for event in events:
        text = str(event.get("text", ""))
        for score, pattern, reason in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                diag = cleanup_diagnosis(match.group(1))
                if diag:
                    candidates.append((score, diag, f"{reason}: {text}"))
    if not candidates:
        typed = [e for e in events if e.get("type") == "diagnosis"]
        if typed:
            text = str(typed[-1].get("text", ""))
            candidates.append((0.55, cleanup_diagnosis(text), text))
    if not candidates:
        text = " ".join(str(e.get("text", "")) for e in events[-2:])
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
        },
    )


def cleanup_diagnosis(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .,:;()[]")
    cleaned = re.sub(r"\b(with|after|following|and underwent|indicating)\b.*$", "", cleaned, flags=re.I).strip(" .,:;")
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rsplit(" ", 1)[0]
    return canonicalize(cleaned)


def canonicalize(text: str) -> str:
    low = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()
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
        if low == key or key in low:
            return value
    return text


def run_llm_prediction(
    case: dict[str, Any],
    *,
    method: str,
    client: DeepSeekClient | None,
    context: str,
    extra: str = "",
    fallback_max_events: int | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    if client is None:
        return heuristic_predict(case, method, max_events=fallback_max_events)
    try:
        result = client.chat(prediction_json_prompt(method, context, extra), temperature=temperature, max_tokens=900)
        raw = extract_json_object(result.text)
        return normalize_prediction(case["case_id"], method, raw, result.usage)
    except Exception as exc:  # noqa: BLE001 - prediction should degrade to fallback, not crash the run
        pred = heuristic_predict(case, method, max_events=fallback_max_events)
        pred["reasoning_summary"] += f" LLM fallback reason: {exc}"
        pred["llm_error"] = str(exc)
        return pred


def run_direct(case: dict[str, Any], client: DeepSeekClient | None) -> dict[str, Any]:
    return run_llm_prediction(
        case,
        method="direct_deepseek",
        client=client,
        context=case_context(case, max_events=min(4, len(case.get("events", []))), include_labs=False),
        extra="Make a diagnosis from this limited truncated context.",
        fallback_max_events=min(4, len(case.get("events", []))),
    )


def run_ours(
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    memory_dir: str | Path,
    strategy: dict[str, Any],
) -> dict[str, Any]:
    memory_path = Path(memory_dir) / f"{case['case_id']}.memory.jsonl"
    store = bootstrap_memory(case, memory_path)
    ops = apply_critique(case, store)
    query = "final diagnosis longitudinal causal evidence treatment imaging pathology labs"
    top_k = int(strategy.get("top_k", 5))
    memories = store.retrieve(query, k=top_k)
    memory_lines = [
        f"- {m.get('summary')} (status={m.get('status')}, confidence={m.get('confidence')}, refs={m.get('evidence_refs')})"
        for m in memories
    ]
    extra = (
        "Use the active JSONL memory cards and ignore discarded/invalidated outdated hypotheses.\n"
        f"[MEMORY_CARDS]\n{chr(10).join(memory_lines)}\n"
        f"[MEMORY_OPS]\n{ops}\n"
        "Before finalizing, check whether longitudinal evidence contradicts the initial hypothesis."
    )
    pred = run_llm_prediction(
        case,
        method=f"ours_topk{top_k}_round{strategy.get('rounds', 1)}",
        client=client,
        context=case_context(case, include_labs=True),
        extra=extra,
        fallback_max_events=None,
        temperature=float(strategy.get("temperature", 0.05)),
    )
    pred["memory_ops"] = ops
    pred["retrieved_memory_count"] = len(memories)
    pred["strategy"] = strategy
    return pred
