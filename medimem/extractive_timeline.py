from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


EXTRACTION_VERSION = "uniform_session_coverage_v1"
MIN_SESSION_CHARS = 80
TARGET_RECORD_CHARS = 160


def _clean(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _uniform_indices(size: int, count: int) -> list[int]:
    if size <= 0 or count <= 0:
        return []
    if count >= size:
        return list(range(size))
    if count == 1:
        return [0]
    indices = {round(index * (size - 1) / (count - 1)) for index in range(count)}
    return sorted(indices)


def _turn_record(turn: dict[str, Any], original_index: int) -> dict[str, Any] | None:
    text = _clean(turn.get("text"), 100_000)
    if not text:
        return None
    session = _clean(turn.get("session") or turn.get("session_date") or turn.get("time") or "unknown", 80)
    date = _clean(turn.get("session_date") or turn.get("time"), 40)
    speaker = _clean(turn.get("speaker"), 40)
    refs_value = turn.get("evidence_refs") or []
    if isinstance(refs_value, str):
        refs_value = [refs_value]
    refs = _clean(",".join(str(ref) for ref in refs_value if str(ref or "").strip()), 100)
    prefix = f"[date={date} session={session} speaker={speaker} refs={refs}] "
    return {
        "original_index": original_index,
        "session": session,
        "prefix": prefix,
        "text": text,
        "full_length": len(prefix) + len(text),
    }


def build_extractive_timeline(
    turns: list[dict[str, Any]],
    *,
    max_chars: int,
) -> tuple[str, dict[str, Any]]:
    if max_chars < 256:
        raise ValueError("Extractive timeline max_chars must be at least 256.")
    records = [
        record
        for index, turn in enumerate(turns)
        if (record := _turn_record(turn, index)) is not None
    ]
    header = f"[EXTRACTIVE_TIMELINE version={EXTRACTION_VERSION}]\n"
    if not records:
        memory = header + "No timeline facts stored."
        return memory[:max_chars], {
            "algorithm_version": EXTRACTION_VERSION,
            "input_turn_count": len(turns),
            "input_record_count": 0,
            "input_session_count": 0,
            "retained_record_count": 0,
            "retained_session_count": 0,
            "output_chars": len(memory[:max_chars]),
            "records_truncated": 0,
            "sessions_sampled": False,
        }

    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    session_order: list[str] = []
    for record in records:
        if record["session"] not in by_session:
            session_order.append(record["session"])
        by_session[record["session"]].append(record)

    available = max_chars - len(header)
    max_session_count = max(1, available // MIN_SESSION_CHARS)
    selected_session_indices = _uniform_indices(len(session_order), min(len(session_order), max_session_count))
    selected_sessions = [session_order[index] for index in selected_session_indices]

    floor = min(MIN_SESSION_CHARS, max(1, available // len(selected_sessions)))
    budgets = {session: floor for session in selected_sessions}
    remaining = max(0, available - floor * len(selected_sessions))
    weights = {
        session: math.sqrt(sum(record["full_length"] for record in by_session[session]))
        for session in selected_sessions
    }
    weight_total = sum(weights.values()) or 1.0
    fractions: list[tuple[float, str]] = []
    allocated = 0
    for session in selected_sessions:
        exact = remaining * weights[session] / weight_total
        extra = int(exact)
        budgets[session] += extra
        allocated += extra
        fractions.append((exact - extra, session))
    for _, session in sorted(fractions, reverse=True)[: remaining - allocated]:
        budgets[session] += 1

    selected_records: list[tuple[int, str, bool]] = []
    for session in selected_sessions:
        session_records = by_session[session]
        budget = budgets[session]
        record_count = min(len(session_records), max(1, budget // TARGET_RECORD_CHARS))
        chosen = [session_records[index] for index in _uniform_indices(len(session_records), record_count)]
        per_record = max(1, (budget - max(0, len(chosen) - 1)) // len(chosen))
        for record in chosen:
            prefix = record["prefix"]
            if len(prefix) >= per_record:
                rendered = _clean(prefix, per_record)
            else:
                rendered = prefix + _clean(record["text"], per_record - len(prefix))
            selected_records.append(
                (record["original_index"], rendered, len(rendered) < record["full_length"])
            )

    selected_records.sort(key=lambda item: item[0])
    body = "\n".join(record for _, record, _ in selected_records)
    memory = (header + body)[:max_chars]
    diagnostics = {
        "algorithm_version": EXTRACTION_VERSION,
        "input_turn_count": len(turns),
        "input_record_count": len(records),
        "input_session_count": len(session_order),
        "retained_record_count": len(selected_records),
        "retained_session_count": len(selected_sessions),
        "output_chars": len(memory),
        "records_truncated": sum(int(truncated) for _, _, truncated in selected_records),
        "sessions_sampled": len(selected_sessions) < len(session_order),
    }
    return memory, diagnostics
