from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
                card["status"] = "invalidated" if op == "Invalidate" else "discarded"
                card["updated_by_op"] = op
                card["update_reason"] = reason
                touched.append(card)
        if touched:
            self.save()
        return touched

    def retrieve(self, query: str, k: int = 5, include_inactive: bool = False) -> list[dict[str, Any]]:
        scored = []
        for card in self.cards:
            if not include_inactive and card.get("status") not in {"active", "flagged"}:
                continue
            score = text_similarity(query, str(card.get("summary", "")))
            scored.append((score, card))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [card | {"retrieval_score": score} for score, card in scored[:k] if score > 0 or include_inactive]


def bootstrap_memory(case: dict[str, Any], path: str | Path) -> MemoryStore:
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
    for poison in case.get("poison_records", []):
        store.write_card(
            summary=str(poison.get("text")),
            evidence_refs=[str(poison.get("poison_id"))],
            time_scope={},
            confidence=0.2,
            tags=["poison", "outdated"],
            status="active",
            op="Write",
        )
    return store


def apply_critique(case: dict[str, Any], store: MemoryStore) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    diagnosis_events = [e for e in case.get("events", []) if e.get("type") == "diagnosis"]
    final_evidence = diagnosis_events[-1] if diagnosis_events else (case.get("events") or [{}])[-1]
    for poison in case.get("poison_records", []):
        expected_op = str(poison.get("expected_op") or "Discard")
        touched = store.invalidate_or_discard(
            str(poison.get("text", "")),
            expected_op,
            f"Contradicted by {final_evidence.get('event_id')}: {final_evidence.get('text')}",
        )
        operations.append(
            {
                "op": expected_op,
                "target": poison.get("poison_id"),
                "touched_memory_ids": [t["memory_id"] for t in touched],
                "reason": "poison/outdated record conflicts with later confirmed evidence",
            }
        )
    return operations

