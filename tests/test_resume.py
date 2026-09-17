import json

import pytest

from medimem.resume import load_valid_resume_predictions


def prediction(sample_id: str, answer: str = "Boston") -> dict:
    return {
        "sample_id": sample_id,
        "dataset": "rhelm",
        "method": "official_letta_memgpt_timeline_adapter",
        "endpoint": "http://127.0.0.1:8001/v1",
        "answer": answer,
        "guard_passed": True,
    }


def test_resume_accepts_valid_rows_and_reports_diagnostics(tmp_path):
    source = tmp_path / "predictions.jsonl"
    source.write_text(json.dumps(prediction("s1")) + "\n", encoding="utf-8")
    rows, diagnostics = load_valid_resume_predictions(
        [{"sample_id": "s1"}, {"sample_id": "s2"}],
        [source],
        dataset="rhelm",
        method="official_letta_memgpt_timeline_adapter",
        endpoint="http://127.0.0.1:8001/v1",
    )
    assert list(rows) == ["s1"]
    assert diagnostics["accepted_count"] == 1
    assert diagnostics["rows_read"] == 1


def test_resume_rejects_conflicting_duplicate_answers(tmp_path):
    source = tmp_path / "predictions.jsonl"
    source.write_text(
        "\n".join((json.dumps(prediction("s1")), json.dumps(prediction("s1", "Chicago")))) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Conflicting resume answers"):
        load_valid_resume_predictions(
            [{"sample_id": "s1"}],
            [source],
            dataset="rhelm",
            method="official_letta_memgpt_timeline_adapter",
            endpoint="http://127.0.0.1:8001/v1",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_id", "unknown", "unknown sample_id"),
        ("dataset", "locomo", "wrong dataset"),
        ("method", "other", "wrong method"),
        ("endpoint", "http://127.0.0.1:9999/v1", "wrong endpoint"),
        ("answer", "", "empty answer"),
        ("guard_passed", False, "did not pass"),
    ],
)
def test_resume_rejects_invalid_rows(tmp_path, field, value, message):
    row = prediction("s1")
    row[field] = value
    source = tmp_path / "predictions.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_valid_resume_predictions(
            [{"sample_id": "s1"}],
            [source],
            dataset="rhelm",
            method="official_letta_memgpt_timeline_adapter",
            endpoint="http://127.0.0.1:8001/v1",
        )


def test_resume_accepts_explicit_equivalent_endpoint(tmp_path):
    source = tmp_path / "predictions.jsonl"
    row = prediction("s1")
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows, diagnostics = load_valid_resume_predictions(
        [{"sample_id": "s1"}],
        [source],
        dataset="rhelm",
        method="official_letta_memgpt_timeline_adapter",
        endpoint="http://127.0.0.1:8002/v1",
        allowed_endpoints=["http://127.0.0.1:8001/v1"],
    )
    assert list(rows) == ["s1"]
    assert diagnostics["allowed_endpoints"] == [
        "http://127.0.0.1:8001/v1",
        "http://127.0.0.1:8002/v1",
    ]


def test_resume_rejects_a_different_ingestion_mode(tmp_path):
    source = tmp_path / "predictions.jsonl"
    row = {**prediction("s1"), "ingestion_mode": "rolling_llm"}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong ingestion mode"):
        load_valid_resume_predictions(
            [{"sample_id": "s1"}],
            [source],
            dataset="rhelm",
            method="official_letta_memgpt_timeline_adapter",
            endpoint="http://127.0.0.1:8001/v1",
            expected_ingestion_mode="extractive",
        )

