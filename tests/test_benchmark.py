import json

from mem_ehr_agent.benchmark import (
    evaluate_benchmark_predictions,
    load_locomo_samples,
    parse_methods,
    validate_horizontal_run,
)


def test_parse_methods_normalizes_comma_list():
    assert parse_methods("ours, amem") == ["ours", "amem"]


def test_load_locomo_samples_from_official_json(tmp_path):
    path = tmp_path / "locomo.json"
    payload = [
        {
            "qa": [{"question": "When did A visit?", "answer": "Monday", "category": 2, "evidence": ["D1:0"]}],
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "Monday",
                "session_1": [{"speaker": "A", "dia_id": "D1:0", "text": "I visited on Monday."}],
            },
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    samples = load_locomo_samples(path)
    assert len(samples) == 1
    assert samples[0]["dataset"] == "locomo"
    assert "I visited on Monday" in samples[0]["context"]


def test_horizontal_validation_requires_same_sample_ids_per_method():
    samples = [
        {"sample_id": "s1", "split": "official"},
        {"sample_id": "s2", "split": "official"},
    ]
    predictions = [
        {"sample_id": "s1", "method": "ours"},
        {"sample_id": "s2", "method": "ours"},
        {"sample_id": "s1", "method": "amem"},
    ]
    validation = validate_horizontal_run(samples, predictions)
    assert not validation["passed"]
    assert validation["failures"][0]["method"] == "amem"


def test_benchmark_metrics_are_grouped_by_method_and_split():
    samples = [{"sample_id": "s1", "split": "official", "answer": "Monday", "dataset": "locomo"}]
    predictions = [
        {
            "sample_id": "s1",
            "method": "ours",
            "split": "official",
            "answer": "Monday",
            "usage": {"total_tokens": 10},
            "retrieved_memory_count": 3,
        }
    ]
    rows = evaluate_benchmark_predictions(samples, predictions)
    assert rows[0]["method"] == "ours"
    assert rows[0]["exact_match"] == 1.0
    assert rows[0]["qa_f1"] == 1.0

