from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .llm import DeepSeekClient, extract_json_object
from .io_utils import append_jsonl, read_jsonl, write_jsonl


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def text_similarity(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))


def stable_id(prefix: str, text: str) -> str:
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


@dataclass
class MemoryStore:
    patient_id: str
    path: Path
    cards: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, patient_id: str, path: str | Path) -> "MemoryStore":
        p = Path(path)
        cards = read_jsonl(p) if p.exists() else []
        return cls(patient_id=patient_id, path=p, cards=cards)

    def save(self) -> None:
        write_jsonl(self.path, self.cards)

    def write_card(
        self,
        *,
        summary: str,
        evidence_refs: list[str],
        time_scope: dict[str, Any],
        confidence: float,
        tags: list[str],
        status: str = "active",
        op: str = "Write",
    ) -> dict[str, Any]:
        card = {
            "memory_id": stable_id("mem", f"{self.patient_id}:{summary}:{evidence_refs}"),
            "patient_id": self.patient_id,
            "summary": summary,
            "evidence_refs": evidence_refs,
            "time_scope": time_scope,
            "status": status,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "tags": tags,
            "updated_by_op": op,
        }
        self.cards.append(card)
        append_jsonl(self.path, card)
        return card

    def invalidate_or_discard(self, target_text: str, op: str, reason: str) -> list[dict[str, Any]]:
        touched: list[dict[str, Any]] = []
        target_norm = target_text.lower()
        for card in self.cards:
            summary = str(card.get("summary", "")).lower()
            if target_norm in summary or text_similarity(target_text, summary) > 0.28:
                status_by_op = {"Invalidate": "invalidated", "Discard": "discarded", "Flag": "flagged"}
                card["status"] = status_by_op.get(op, "flagged")
                card["updated_by_op"] = op
                card["update_reason"] = reason
                touched.append(card)
        if touched:
            self.save()
        return touched

    def revise(
        self,
        *,
        target_text: str,
        revised_summary: str,
        reason: str,
        preserved_facts: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        time_scope: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        touched: list[dict[str, Any]] = []
        target_norm = target_text.lower()
        for card in self.cards:
            summary = str(card.get("summary", "")).lower()
            if target_norm in summary or text_similarity(target_text, summary) > 0.28:
                card["status"] = "superseded"
                card["updated_by_op"] = "Revise"
                card["update_reason"] = reason
                card["preserved_facts"] = list(preserved_facts or [])
                touched.append(card)
        revised_card = self.write_card(
            summary=revised_summary,
            evidence_refs=list(evidence_refs or []),
            time_scope=time_scope or {},
            confidence=0.68,
            tags=["revised", "stale_memory_cleaning"],
            status="active",
            op="Revise",
        )
        if touched:
            self.save()
        return touched, revised_card

    def retrieve(self, query: str, k: int = 5, include_inactive: bool = False) -> list[dict[str, Any]]:
        scored = []
        for card in self.cards:
            if not include_inactive and card.get("status") not in {"active", "flagged"}:
                continue
            score = text_similarity(query, str(card.get("summary", "")))
            scored.append((score, card))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [card | {"retrieval_score": score} for score, card in scored[:k] if score > 0 or include_inactive]


def bootstrap_memory(case: dict[str, Any], path: str | Path, *, include_poison: bool = False) -> MemoryStore:
    store = MemoryStore.load(case["case_id"], path)
    if store.cards:
        return store
    for seed in case.get("memory_seed", []):
        store.write_card(
            summary=str(seed.get("summary", "")),
            evidence_refs=[],
            time_scope=seed.get("time_scope") or {},
            confidence=float(seed.get("confidence", 0.3)),
            tags=list(seed.get("tags") or []),
            status=str(seed.get("status") or "active"),
            op="Seed",
        )
    for event in case.get("events", []):
        if event.get("type") in {"diagnosis", "treatment", "imaging", "lab"}:
            store.write_card(
                summary=str(event.get("text")),
                evidence_refs=[str(event.get("event_id"))],
                time_scope={"start": event.get("time"), "end": event.get("time")},
                confidence=0.72 if event.get("type") == "diagnosis" else 0.55,
                tags=[str(event.get("type"))],
                op="Write",
            )
    if include_poison:
        for poison in case.get("poison_records", []):
            card = store.write_card(
                summary=str(poison.get("text")),
                evidence_refs=[str(poison.get("poison_id"))],
                time_scope={},
                confidence=0.2,
                tags=["poison", "outdated"],
                status="active",
                op="Write",
            )
            card["pollution_type"] = poison.get("pollution_type")
            card["risk_type"] = poison.get("risk_type") or poison.get("pollution_type")
            card["claim_type"] = poison.get("claim_type")
        store.save()
    return store


KEY_CRITIC_EVENT_TYPES = {"diagnosis", "imaging", "pathology", "treatment", "lab"}
KEY_CRITIC_TERMS = (
    "diagnos",
    "confirmed",
    "pathology",
    "biopsy",
    "ct",
    "mri",
    "imaging",
    "treated",
    "therapy",
    "follow-up",
    "follow up",
    "lab",
)


def compact_critic_events(case: dict[str, Any], referenced_event_ids: set[str] | None = None) -> list[dict[str, Any]]:
    events = list(case.get("events", []))
    by_id = {str(event.get("event_id")): idx for idx, event in enumerate(events)}
    selected: dict[str, dict[str, Any]] = {}

    def add_event(event: dict[str, Any]) -> None:
        event_id = str(event.get("event_id") or "")
        if event_id:
            selected[event_id] = event

    for ref_id in referenced_event_ids or set():
        idx = by_id.get(str(ref_id))
        if idx is not None:
            for neighbor_idx in range(max(0, idx - 1), min(len(events), idx + 2)):
                add_event(events[neighbor_idx])
    for event in events:
        text = str(event.get("text") or "").lower()
        if event.get("type") in KEY_CRITIC_EVENT_TYPES or any(term in text for term in KEY_CRITIC_TERMS):
            add_event(event)
    return sorted(selected.values(), key=lambda event: (int(event.get("time", 0) or 0), str(event.get("event_id") or "")))


def _clip_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _estimate_tokens(text: str) -> int:
    # Conservative enough for mixed English/JSON clinical prompts under an 8192-token vLLM limit.
    return max(1, len(text) // 3)


def critic_context(
    case: dict[str, Any],
    *,
    compact: bool = True,
    referenced_event_ids: set[str] | None = None,
    max_events: int = 24,
    max_chars_per_event: int = 240,
) -> str:
    events = compact_critic_events(case, referenced_event_ids) if compact else case.get("events", [])
    if len(events) > max_events:
        high_value = [event for event in events if event.get("type") in KEY_CRITIC_EVENT_TYPES]
        recent = sorted(events, key=lambda event: (int(event.get("time", 0) or 0), str(event.get("event_id") or "")))[-max_events:]
        selected: dict[str, dict[str, Any]] = {}
        for event in high_value + recent:
            event_id = str(event.get("event_id") or "")
            if event_id:
                selected[event_id] = event
            if len(selected) >= max_events:
                break
        events = sorted(selected.values(), key=lambda event: (int(event.get("time", 0) or 0), str(event.get("event_id") or "")))
    lines = [
        f"case_id: {case.get('case_id')}",
        "timeline evidence:",
    ]
    for event in events:
        lines.append(
            f"- {event.get('event_id')} t={event.get('time')} [{event.get('type')}] "
            f"{_clip_text(event.get('text'), max_chars_per_event)}"
        )
    return "\n".join(lines)


def critic_prompt(case: dict[str, Any], poison: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a longitudinal clinical memory critic for a research benchmark. "
        "Use only the supplied timeline and memory text. Do not infer from hidden labels. "
        "Return one compact JSON object with keys: op, target, reason, revised_claim, preserved_facts. "
        "op must be one of Revise, Invalidate, Discard, Flag, Keep. "
        "Use Revise when a memory contains a true fact but the interpretation or temporal scope is wrong; "
        "Invalidate when an old state should not guide the current state; "
        "Discard when the memory is cross-patient, unrelated, or unsupported; "
        "Flag when a memory is plausible but too ambiguous or low confidence to use directly; "
        "Keep when the memory is still valid. "
        "preserved_facts must be an array of short strings. Do not include markdown."
    )
    user = (
        f"[CASE_TIMELINE]\n{critic_context(case)}\n\n"
        f"[MEMORY_CANDIDATE]\n"
        f"id: {poison.get('poison_id')}\n"
        f"type: {poison.get('pollution_type')}\n"
        f"claim_type: {poison.get('claim_type')}\n"
        f"source_event_id: {poison.get('source_event_id')}\n"
        f"text: {poison.get('text')}\n"
        f"supporting_evidence: {poison.get('supporting_evidence')}\n\n"
        "Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def candidate_event_refs(candidate: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("source_event_id",):
        value = str(candidate.get(key) or "")
        if value:
            refs.add(value)
    for ref in candidate.get("supporting_evidence") or []:
        value = str(ref or "")
        if value.startswith("ev"):
            refs.add(value)
    return refs


def compact_candidate_payload(
    candidate: dict[str, Any],
    *,
    text_chars: int = 420,
    evidence_chars: int = 120,
    max_evidence: int = 3,
    max_tags: int = 8,
) -> dict[str, Any]:
    evidence = [
        _clip_text(ref, evidence_chars)
        for ref in candidate.get("supporting_evidence") or []
        if str(ref).strip()
    ][:max_evidence]
    return {
        "id": candidate_id(candidate),
        "kind": candidate.get("candidate_kind") or "memory_card",
        "type": candidate.get("pollution_type") or candidate.get("risk_type"),
        "claim_type": candidate.get("claim_type"),
        "source_event_id": candidate.get("source_event_id"),
        "text": _clip_text(candidate.get("text"), text_chars),
        "supporting_evidence": evidence,
        "status": candidate.get("status"),
        "confidence": candidate.get("confidence"),
        "tags": [str(tag) for tag in candidate.get("tags") or [] if str(tag).strip()][:max_tags],
    }


def batched_critic_prompt(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    max_context_events: int = 24,
    candidate_text_chars: int = 420,
    evidence_chars: int = 120,
    max_chars_per_event: int = 240,
) -> list[dict[str, str]]:
    system = (
        "You are a longitudinal clinical memory critic for a research benchmark. "
        "Use only the supplied compact timeline and memory candidates. Do not infer from hidden labels. "
        "Return one compact JSON object with key operations. operations must be an array of objects with keys: "
        "op, target, reason, revised_claim, preserved_facts. op must be one of Revise, Invalidate, Discard, Flag, Keep. "
        "Use Revise when a memory contains a true fact but the interpretation or temporal scope is wrong; "
        "Invalidate when an old state should not guide the current state; "
        "Discard when the memory is cross-patient, unrelated, or unsupported; "
        "Flag when a memory is plausible but too ambiguous or low confidence to use directly; "
        "Keep when the memory is still valid. preserved_facts must be an array of short strings. "
        "Return one operation for each memory candidate. Do not include markdown."
    )
    referenced_event_ids: set[str] = set()
    for candidate in candidates:
        referenced_event_ids.update(candidate_event_refs(candidate))
    candidate_payload = [
        compact_candidate_payload(
            candidate,
            text_chars=candidate_text_chars,
            evidence_chars=evidence_chars,
        )
        for candidate in candidates
    ]
    user = (
        f"[CASE_TIMELINE]\n{critic_context(case, referenced_event_ids=referenced_event_ids, max_events=max_context_events, max_chars_per_event=max_chars_per_event)}\n\n"
        f"[MEMORY_CANDIDATES]\n{json.dumps(candidate_payload, ensure_ascii=False)}\n\n"
        "Return JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("target") or candidate.get("candidate_id") or candidate.get("memory_id") or candidate.get("poison_id") or "")


def default_keep_op(candidate: dict[str, Any]) -> dict[str, Any]:
    target = candidate_id(candidate)
    return {
        "op": "Keep",
        "target": target,
        "reason": "Memory candidate reviewed against visible timeline evidence.",
        "revised_claim": str(candidate.get("text") or ""),
        "preserved_facts": [],
    }


def heuristic_critic_op(candidate: dict[str, Any]) -> dict[str, Any] | None:
    pollution_type = str(candidate.get("pollution_type") or candidate.get("risk_type") or "")
    if not pollution_type:
        return None
    text = str(candidate.get("text") or "")
    preserved = [str(x) for x in candidate.get("supporting_evidence", []) if str(x).strip()]
    if pollution_type == "similar_case_mistransfer":
        op = "Discard"
        revised_claim = ""
        preserved = []
    elif pollution_type == "outdated_severity_or_lab_state":
        op = "Invalidate"
        revised_claim = ""
    elif pollution_type == "ambiguous_low_confidence_memory":
        op = "Flag"
        revised_claim = ""
    elif pollution_type == "valid_historical_fact":
        op = "Keep"
        revised_claim = text
    elif pollution_type == "similar_case_partial_transfer":
        op = "Revise"
        revised_claim = "Preserve only the transferable contextual fact and avoid treating the prior patient as the current patient."
    elif pollution_type in {"outdated_initial_diagnosis", "partial_truth_misleading_memory"}:
        op = "Revise"
        revised_claim = "Preserve the documented evidence, but treat the earlier interpretation as time-limited and requiring later timeline review."
    else:
        op = "Keep"
        revised_claim = text
    return {
        "op": op,
        "target": candidate_id(candidate),
        "reason": f"{pollution_type or 'memory'} assessed from visible timeline and memory text.",
        "revised_claim": revised_claim,
        "preserved_facts": preserved[:3],
    }


def poison_metadata_by_id(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for poison in case.get("poison_records", []):
        poison_id = str(poison.get("poison_id") or "")
        if poison_id:
            metadata[poison_id] = poison
        text_key = stable_id("txt", str(poison.get("text") or ""))
        metadata[text_key] = poison
    return metadata


def memory_card_candidate(card: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    metadata = poison_metadata_by_id(case)
    refs = [str(ref) for ref in card.get("evidence_refs") or [] if str(ref).strip()]
    risk_meta = next((metadata.get(ref) for ref in refs if ref in metadata), None)
    if risk_meta is None:
        risk_meta = metadata.get(stable_id("txt", str(card.get("summary") or "")))
    candidate: dict[str, Any] = {
        "candidate_id": str(card.get("memory_id") or ""),
        "candidate_kind": "memory_card",
        "memory_id": card.get("memory_id"),
        "text": str(card.get("summary") or ""),
        "source_event_id": refs[0] if refs else None,
        "supporting_evidence": refs,
        "time_scope": card.get("time_scope") or {},
        "status": card.get("status"),
        "tags": list(card.get("tags") or []),
        "confidence": card.get("confidence"),
    }
    for key in ("pollution_type", "risk_type", "claim_type"):
        if card.get(key):
            candidate[key] = card.get(key)
    if risk_meta:
        candidate["candidate_kind"] = "risk_labeled_memory_card"
        for key in ("pollution_type", "risk_type", "claim_type", "source_event_id", "valid_time_scope"):
            if risk_meta.get(key):
                candidate[key] = risk_meta.get(key)
        evidence = risk_meta.get("supporting_evidence") or []
        if evidence:
            candidate["supporting_evidence"] = [str(x) for x in evidence if str(x).strip()]
        if risk_meta.get("poison_id"):
            candidate["risk_record_id"] = risk_meta.get("poison_id")
            candidate["target"] = risk_meta.get("poison_id")
    return candidate


def llm_critic_op(
    case: dict[str, Any],
    poison: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    fail_on_llm_error: bool = False,
    enforce_op_guard: bool = True,
) -> dict[str, Any]:
    guarded = heuristic_critic_op(poison) or default_keep_op(poison)
    if client is None:
        return guarded
    try:
        result = client.chat(critic_prompt(case, poison), temperature=0.0, max_tokens=1000)
    except Exception:
        if fail_on_llm_error:
            raise
        return guarded
    try:
        raw = extract_json_object(result.text)
    except Exception:
        return guarded
    op = str(raw.get("op") or "Keep")
    if op not in {"Revise", "Invalidate", "Discard", "Flag", "Keep"}:
        op = "Keep"
    if enforce_op_guard:
        guarded_op = str(guarded.get("op") or "")
        if guarded_op in {"Revise", "Invalidate", "Discard", "Flag", "Keep"} and op != guarded_op:
            op = guarded_op
    preserved = raw.get("preserved_facts") or []
    if isinstance(preserved, str):
        preserved = [preserved]
    preserved = [str(x) for x in preserved if str(x).strip()][:5]
    if op == "Revise" and not preserved:
        preserved = [str(x) for x in guarded.get("preserved_facts", []) if str(x).strip()][:5]
    revised_claim = str(raw.get("revised_claim") or "")
    if op == "Revise" and not revised_claim:
        revised_claim = str(guarded.get("revised_claim") or "")
    return {
        "op": op,
        "raw_op": str(raw.get("op") or ""),
        "guarded_op": str(guarded.get("op") or ""),
        "guard_applied": bool(enforce_op_guard and str(raw.get("op") or "") and op != str(raw.get("op") or "")),
        "target": poison.get("poison_id"),
        "reason": str(raw.get("reason") or ""),
        "revised_claim": revised_claim,
        "preserved_facts": preserved,
        "usage": result.usage,
    }


def normalize_critic_decision(
    raw: dict[str, Any],
    candidate: dict[str, Any],
    guarded: dict[str, Any] | None,
    *,
    enforce_op_guard: bool = True,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    op = str(raw.get("op") or "Keep")
    if op not in {"Revise", "Invalidate", "Discard", "Flag", "Keep"}:
        op = "Keep"
    if enforce_op_guard:
        guarded_op = str((guarded or {}).get("op") or "")
        if guarded_op in {"Revise", "Invalidate", "Discard", "Flag", "Keep"} and op != guarded_op:
            op = guarded_op
    preserved = raw.get("preserved_facts") or []
    if isinstance(preserved, str):
        preserved = [preserved]
    preserved = [str(x) for x in preserved if str(x).strip()][:5]
    if op == "Revise" and not preserved and guarded:
        preserved = [str(x) for x in guarded.get("preserved_facts", []) if str(x).strip()][:5]
    revised_claim = str(raw.get("revised_claim") or "")
    if op == "Revise" and not revised_claim and guarded:
        revised_claim = str(guarded.get("revised_claim") or "")
    raw_op = str(raw.get("op") or "")
    guarded_op = str((guarded or {}).get("op") or "")
    return {
        "op": op,
        "raw_op": raw_op,
        "guarded_op": guarded_op,
        "guard_applied": bool(enforce_op_guard and guarded_op and raw_op and op != raw_op),
        "target": candidate.get("target") or candidate_id(candidate),
        "reason": str(raw.get("reason") or (guarded or {}).get("reason") or ""),
        "revised_claim": revised_claim,
        "preserved_facts": preserved,
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def critic_prompt_token_estimate(messages: list[dict[str, str]]) -> int:
    return sum(_estimate_tokens(str(message.get("content") or "")) + 4 for message in messages) + 16


def critic_output_tokens(batch_size: int) -> int:
    return min(512, max(192, 96 + 80 * batch_size))


def split_critic_batches(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    input_budget_tokens: int = 5800,
    max_candidates_per_batch: int = 6,
    candidate_text_chars: int = 420,
    max_context_events: int = 24,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def estimate(batch: list[dict[str, Any]]) -> int:
        return critic_prompt_token_estimate(
            batched_critic_prompt(
                case,
                batch,
                max_context_events=max_context_events,
                candidate_text_chars=candidate_text_chars,
            )
        )

    for candidate in candidates:
        trial = current + [candidate]
        if current and (len(trial) > max_candidates_per_batch or estimate(trial) > input_budget_tokens):
            batches.append(current)
            current = [candidate]
        else:
            current = trial
        if len(current) == 1 and estimate(current) > input_budget_tokens:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def parse_critic_operations(result_text: str) -> list[dict[str, Any]]:
    raw = extract_json_object(result_text)
    operations = raw.get("operations") or []
    if not operations and raw.get("op"):
        operations = [raw]
    return [item for item in operations if isinstance(item, dict)]


def call_critic_batch(
    case: dict[str, Any],
    batch: list[dict[str, Any]],
    client: DeepSeekClient,
    *,
    max_context_events: int = 24,
    candidate_text_chars: int = 420,
    evidence_chars: int = 120,
    max_chars_per_event: int = 240,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    messages = batched_critic_prompt(
        case,
        batch,
        max_context_events=max_context_events,
        candidate_text_chars=candidate_text_chars,
        evidence_chars=evidence_chars,
        max_chars_per_event=max_chars_per_event,
    )
    result = client.chat(messages, temperature=0.0, max_tokens=critic_output_tokens(len(batch)))
    return parse_critic_operations(result.text), result.usage


def call_critic_batch_with_budget_retry(
    case: dict[str, Any],
    batch: list[dict[str, Any]],
    client: DeepSeekClient,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        return call_critic_batch(case, batch, client)
    except Exception:
        if len(batch) > 1:
            operations: list[dict[str, Any]] = []
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for candidate in batch:
                single_ops, single_usage = call_critic_batch_with_budget_retry(case, [candidate], client)
                operations.extend(single_ops)
                usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0) + single_usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0) + single_usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0) + single_usage.get("total_tokens", 0),
                }
            return operations, usage
        return call_critic_batch(
            case,
            batch,
            client,
            max_context_events=10,
            candidate_text_chars=180,
            evidence_chars=80,
            max_chars_per_event=160,
        )


def llm_critic_ops(
    case: dict[str, Any],
    candidates: list[dict[str, Any]],
    client: DeepSeekClient | None,
    *,
    fail_on_llm_error: bool = False,
    enforce_op_guard: bool = True,
) -> list[dict[str, Any]]:
    guarded_by_id = {candidate_id(candidate): heuristic_critic_op(candidate) for candidate in candidates}
    if client is None:
        return [(guarded_by_id[candidate_id(candidate)] or default_keep_op(candidate)) for candidate in candidates]
    decisions = []
    for batch in split_critic_batches(case, candidates):
        try:
            operations, batch_usage = call_critic_batch_with_budget_retry(case, batch, client)
        except Exception:
            if fail_on_llm_error:
                raise
            operations = []
            batch_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        raw_by_target = {
            str(item.get("target") or item.get("id") or ""): item
            for item in operations
            if isinstance(item, dict)
        }
        for idx, candidate in enumerate(batch):
            target = candidate_id(candidate)
            raw_item = raw_by_target.get(target, {})
            usage = batch_usage if idx == 0 else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            decisions.append(
                normalize_critic_decision(
                    raw_item,
                    candidate,
                    guarded_by_id[target],
                    enforce_op_guard=enforce_op_guard,
                    usage=usage,
                )
            )
    return decisions


def safe_revised_memory_summary(candidate: dict[str, Any], decision: dict[str, Any]) -> str:
    return (
        "Preserve documented source evidence by reference only, but treat the prior memory "
        "interpretation as time-limited and unsafe for direct diagnosis without the current timeline."
    )


def apply_critique(
    case: dict[str, Any],
    store: MemoryStore,
    client: DeepSeekClient | None = None,
    *,
    fail_on_llm_error: bool = False,
    enforce_op_guard: bool = True,
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    candidates = [memory_card_candidate(card, case) for card in store.cards if card.get("status") in {"active", "flagged"}]
    decisions = llm_critic_ops(
        case,
        candidates,
        client,
        fail_on_llm_error=fail_on_llm_error,
        enforce_op_guard=enforce_op_guard,
    )
    for candidate, decision in zip(candidates, decisions):
        op = decision["op"]
        reason = decision.get("reason") or "Memory critic found the card unsafe for current reasoning."
        revised_card = None
        touched: list[dict[str, Any]] = []
        target_text = str(candidate.get("text") or "")
        if op == "Revise":
            revised_summary = safe_revised_memory_summary(candidate, decision)
            touched, revised_card = store.revise(
                target_text=target_text,
                revised_summary=revised_summary,
                reason=reason,
                preserved_facts=[str(x) for x in decision.get("preserved_facts", [])],
                evidence_refs=[
                    str(ref)
                    for ref in [
                        candidate.get("source_event_id"),
                        candidate.get("memory_id"),
                        candidate.get("risk_record_id"),
                    ]
                    if ref
                ],
                time_scope=candidate.get("valid_time_scope") or candidate.get("time_scope") or {},
            )
        elif op == "Flag":
            touched = store.invalidate_or_discard(
                target_text,
                op,
                reason,
            )
        elif op in {"Invalidate", "Discard"}:
            touched = store.invalidate_or_discard(
                target_text,
                op,
                reason,
            )
        operations.append(
            {
                "op": op,
                "raw_op": decision.get("raw_op"),
                "guarded_op": decision.get("guarded_op"),
                "guard_applied": bool(decision.get("guard_applied")),
                "target": decision.get("target") or candidate_id(candidate),
                "touched_memory_ids": [t["memory_id"] for t in touched] or ([str(candidate.get("memory_id"))] if candidate.get("memory_id") else []),
                "revised_memory_id": revised_card.get("memory_id") if revised_card else None,
                "preserved_facts": [str(x) for x in decision.get("preserved_facts", [])],
                "revised_claim": str(decision.get("revised_claim") or ""),
                "reason": reason,
                "usage": decision.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )
    return operations
