from __future__ import annotations

from typing import Any

from pathlib import Path

from .agents import case_context, compact_memory_lines, pollution_memory_context, run_llm_prediction, sanitize_runtime_text
from .amem_baseline import run_amem_adapter
from .llm import DeepSeekClient
from .memory import bootstrap_memory


BASELINE_SOURCES = {
    "static_rag": {"paper": "Static lexical retrieval baseline", "repo": "local protocol baseline"},
    "amem": {
        "paper": "A-MEM: Agentic Memory for LLM Agents",
        "repo": "https://github.com/agiresearch/A-mem",
    },
    "ddo": {
        "paper": "Jia et al., EMNLP 2025, DDO: Dual-Decision Optimization for LLM-Based Medical Consultation via Multi-Agent Collaboration",
        "repo": "https://github.com/zh-jia/DDO",
    },
    "colacare": {
        "paper": "Wang et al., WWW 2025, ColaCare: Enhancing Electronic Health Record Modeling through LLM-Driven Multi-Agent Collaboration",
        "repo": "https://github.com/PKU-AICare/ColaCare",
    },
}


def run_static_rag_adapter(
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    memory_dir: str | Path,
    fail_on_llm_error: bool = False,
    top_k: int = 8,
) -> dict[str, Any]:
    memory_path = Path(memory_dir) / f"{case['case_id']}.static-rag.jsonl"
    store = bootstrap_memory(case, memory_path, include_poison=False)
    retrieved = store.retrieve("final diagnosis treatment imaging pathology laboratory follow-up", k=top_k)
    context = "\n".join([f"case_id: {case['case_id']}", "[STATIC_RETRIEVED_CARDS]", *compact_memory_lines(retrieved)])
    pred = run_llm_prediction(
        case,
        method="baseline_static_rag",
        client=client,
        context=context,
        extra="Use only the statically retrieved cards. No critic, memory update, evidence-note injection, or audit is available.",
        fail_on_llm_error=fail_on_llm_error,
        temperature=0.05,
    )
    pred["retrieved_memory_count"] = len(retrieved)
    pred["baseline_source"] = BASELINE_SOURCES["static_rag"]
    return pred


def run_ddo_adapter(
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    fail_on_llm_error: bool = False,
    polluted: bool = False,
) -> dict[str, Any]:
    events = case.get("events", [])
    # DDO is a consultation framework. For this longitudinal EHR adapter, it sees an inquiry-style
    # partial symptom/evidence sequence and must infer the diagnosis under limited observations.
    max_events = max(2, int(len(events) * 0.65))
    extra = (
        "DDO-style adapter: decouple symptom inquiry from diagnosis. "
        "First infer key missing symptom clusters from partial evidence, then provide the likely diagnosis. "
        "Do not assume access to future follow-up after the truncated timeline."
    )
    method = "baseline_ddo_adapter"
    if polluted:
        method = "baseline_polluted_ddo_adapter"
        extra += (
            "\nThis polluted variant also receives previously stored memory cards. "
            "Some may be stale, mixed, or cross-patient; no proposed cleaning module is available.\n"
            f"{pollution_memory_context(case)}"
        )
    pred = run_llm_prediction(
        case,
        method=method,
        client=client,
        context=case_context(case, max_events=max_events, include_labs=False),
        extra=extra,
        fallback_max_events=max_events,
        temperature=0.15,
        fail_on_llm_error=fail_on_llm_error,
    )
    pred["pollution_exposed"] = polluted
    pred["baseline_source"] = BASELINE_SOURCES["ddo"]
    pred["adapter_note"] = "DDO official repo is tracked; this adapter maps longitudinal EHR JSONL to DDO's consultation/diagnosis interface."
    return pred


def run_colacare_adapter(
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    fail_on_llm_error: bool = False,
    polluted: bool = False,
) -> dict[str, Any]:
    # ColaCare is originally mortality/readmission EHR modeling. This adapter exposes structured
    # visits/labs and asks a meta-doctor to diagnose from multi-agent evidence summaries.
    encounter_lines = []
    for enc in case.get("encounters", []):
        summary = sanitize_runtime_text(enc.get("summary"))
        if not summary:
            continue
        labs = ", ".join(f"{lab.get('name')}={lab.get('value')}{lab.get('unit')}" for lab in enc.get("labs", [])[:4])
        encounter_lines.append(f"- t={enc.get('time')} summary={summary} labs={labs}")
    context = "\n".join(
        [
            f"case_id: {case['case_id']}",
            f"demographics: {case.get('demographics', {})}",
            "structured visits:",
            *encounter_lines,
        ]
    )
    extra = (
        "ColaCare-style adapter: simulate DoctorAgents reading structured EHR/labs and a MetaAgent "
        "making a final evidence-based diagnosis. Focus on structured EHR evidence."
    )
    method = "baseline_colacare_adapter"
    if polluted:
        method = "baseline_polluted_colacare_adapter"
        extra += (
            "\nThis polluted variant also receives previously stored memory cards. "
            "Some may be stale, mixed, or cross-patient; no proposed cleaning module is available.\n"
            f"{pollution_memory_context(case)}"
        )
    pred = run_llm_prediction(
        case,
        method=method,
        client=client,
        context=context,
        extra=extra,
        fallback_max_events=max(2, int(len(case.get("events", [])) * 0.75)),
        temperature=0.12,
        fail_on_llm_error=fail_on_llm_error,
    )
    pred["pollution_exposed"] = polluted
    pred["baseline_source"] = BASELINE_SOURCES["colacare"]
    pred["adapter_note"] = "ColaCare official repo is tracked; this adapter maps JSONL cases to structured-visit MetaAgent diagnosis."
    return pred


def run_baseline(
    name: str,
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    fail_on_llm_error: bool = False,
    polluted: bool = False,
) -> dict[str, Any]:
    if name == "amem":
        return run_amem_adapter(case, client, fail_on_llm_error=fail_on_llm_error, polluted=polluted)
    if name == "ddo":
        return run_ddo_adapter(case, client, fail_on_llm_error=fail_on_llm_error, polluted=polluted)
    if name == "colacare":
        return run_colacare_adapter(case, client, fail_on_llm_error=fail_on_llm_error, polluted=polluted)
    raise ValueError(f"Unknown baseline: {name}")
