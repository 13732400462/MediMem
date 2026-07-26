from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pathlib import Path

from .agents import (
    case_context,
    compact_case_context,
    compact_memory_lines,
    pollution_memory_context,
    run_llm_prediction,
    sanitize_runtime_text,
)
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
    "clincare": {
        "paper": "Li et al., AAAI 2026, CliCARE",
        "repo": "released-source CliCARE KG alignment adapter",
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


def _clincare_term_path() -> Path | None:
    configured = os.getenv("CLICARE_ROOT", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured) / "KG_Alignment" / "entities_and_relations.txt")
    package_path = Path(__file__).resolve()
    candidates.extend(
        [
            Path.cwd() / "external_baselines" / "CliCARE-main" / "KG_Alignment" / "entities_and_relations.txt",
            package_path.parents[1] / "external_baselines" / "CliCARE-main" / "KG_Alignment" / "entities_and_relations.txt",
            package_path.parents[2] / "external_baselines" / "CliCARE-main" / "KG_Alignment" / "entities_and_relations.txt",
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


@lru_cache(maxsize=1)
def load_clincare_terms() -> tuple[str, ...]:
    path = _clincare_term_path()
    if path is None:
        return ()
    terms = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        term = line.strip(" -\t")
        if term and len(term) <= 80:
            terms.append(term)
    return tuple(terms[:200])


def run_clincare_adapter(
    case: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    fail_on_llm_error: bool = False,
) -> dict[str, Any]:
    text = compact_case_context(
        case,
        max_events=16,
        include_labs=True,
        include_time=True,
        event_text_limit=140,
    )
    matched = [term for term in load_clincare_terms() if term.lower() in text.lower()][:20]
    context = "\n".join(
        [
            f"case_id: {case['case_id']}",
            "[CLICARE_OFFICIAL_KG_ALIGNMENT_ADAPTER]",
            (
                "The released CliCARE KG alignment code expects Neo4j guideline and temporal KG assets. "
                "This protocol adapter uses the downloaded KG schema terms and trajectory-alignment framing "
                "under the same model endpoint."
            ),
            "[MATCHED_KG_TERMS]",
            "\n".join(f"- {term}" for term in matched) if matched else "- no direct KG term match",
            "[CURRENT_CASE_CONTEXT]",
            text,
        ]
    )
    pred = run_llm_prediction(
        case,
        method="official_clincare_adapter",
        client=client,
        context=context,
        extra=(
            "CliCARE-style clinical KG alignment: align visible case events to guideline/KG concepts before "
            "answering. Do not assume unavailable MIMIC or Neo4j records."
        ),
        temperature=0.05,
        fail_on_llm_error=fail_on_llm_error,
    )
    pred["official_baseline"] = "CliCARE"
    pred["baseline_source"] = BASELINE_SOURCES["clincare"]
    pred["adapter_note"] = (
        "Released KG/Neo4j assets are unavailable; this frozen adapter uses downloaded KG schema terms and "
        "CliCARE trajectory-alignment framing, with the dependency difference recorded."
    )
    pred["matched_kg_terms"] = matched
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
    if name == "clincare":
        if polluted:
            raise ValueError("CliCARE adapter has no polluted-memory variant.")
        return run_clincare_adapter(case, client, fail_on_llm_error=fail_on_llm_error)
    raise ValueError(f"Unknown baseline: {name}")
