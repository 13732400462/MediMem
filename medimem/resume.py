from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def prediction_jsonl_paths(sources: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            paths.extend(sorted(path.rglob("predictions/*.jsonl")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Resume prediction source does not exist: {path}")
    return paths


def load_valid_resume_predictions(
    samples: list[dict[str, Any]],
    sources: Iterable[str | Path],
    *,
    dataset: str,
    method: str,
    endpoint: str,
    allowed_endpoints: Iterable[str] = (),
    expected_ingestion_mode: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sample_ids = {str(sample["sample_id"]) for sample in samples}
    accepted: dict[str, dict[str, Any]] = {}
    files = prediction_jsonl_paths(sources)
    accepted_endpoints = {endpoint, *allowed_endpoints}
    rows_read = 0
    duplicates = 0
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rows_read += 1
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            location = f"{path}:{line_number}"
            if sample_id not in sample_ids:
                raise ValueError(f"Resume row has an unknown sample_id at {location}: {sample_id}")
            if row.get("dataset") != dataset:
                raise ValueError(f"Resume row has the wrong dataset at {location}: {row.get('dataset')}")
            if row.get("method") != method:
                raise ValueError(f"Resume row has the wrong method at {location}: {row.get('method')}")
            if row.get("endpoint") not in accepted_endpoints:
                raise ValueError(f"Resume row has the wrong endpoint at {location}: {row.get('endpoint')}")
            if row.get("guard_passed") is not True:
                raise ValueError(f"Resume row did not pass its guard at {location}")
            if expected_ingestion_mode is not None and row.get("ingestion_mode") != expected_ingestion_mode:
                raise ValueError(
                    f"Resume row has the wrong ingestion mode at {location}: {row.get('ingestion_mode')}"
                )
            if not str(row.get("answer") or "").strip():
                raise ValueError(f"Resume row has an empty answer at {location}")
            previous = accepted.get(sample_id)
            if previous is not None:
                duplicates += 1
                if str(previous["answer"]).strip() != str(row["answer"]).strip():
                    raise ValueError(f"Conflicting resume answers for sample_id {sample_id}")
                continue
            accepted[sample_id] = row
    diagnostics = {
        "sources": [str(source) for source in sources],
        "files": [str(path) for path in files],
        "rows_read": rows_read,
        "accepted_count": len(accepted),
        "duplicate_count": duplicates,
        "allowed_endpoints": sorted(accepted_endpoints),
        "expected_ingestion_mode": expected_ingestion_mode,
    }
    return accepted, diagnostics
