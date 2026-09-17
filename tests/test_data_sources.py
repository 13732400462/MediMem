import pytest

from medimem import data_sources


def test_fetch_hf_rows_discovers_preferred_split(monkeypatch):
    calls = []

    def fake_splits(dataset):
        assert dataset == "demo/dataset"
        return [("default", "train"), ("default", "train_DSR1")]

    def fake_rows(dataset, *, split, config, n, offset=0):
        calls.append((dataset, config, split, n, offset))
        return [{"id": f"{split}-{offset}"}]

    monkeypatch.setattr(data_sources, "hf_dataset_splits", fake_splits)
    monkeypatch.setattr(data_sources, "hf_first_rows", fake_rows)
    notes = []
    rows = data_sources.fetch_hf_rows("demo/dataset", preferred_splits=["train_DSR1", "train"], n=1, notes=notes)
    assert rows == [{"id": "train_DSR1-0"}]
    assert calls[0][2] == "train_DSR1"
    assert "fetched 1 rows" in notes[0]


def test_require_real_data_blocks_fallback(monkeypatch):
    monkeypatch.setattr(data_sources, "fetch_pmoa_rows", lambda n, notes=None, cache_dir=None: [])
    monkeypatch.setattr(data_sources, "fetch_pmc_patient_rows", lambda n, notes=None, cache_dir=None: [])
    with pytest.raises(RuntimeError, match="Real-data build required"):
        data_sources.paired_source_rows(2, require_real_data=True)


def test_load_cached_rows_accepts_hf_rows_payload(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "pmoa_tts.json").write_text('{"rows": [{"row": {"case_report_id": "real-1"}}]}', encoding="utf-8")
    rows = data_sources.load_cached_rows(cache, data_sources.PMOA_CACHE_NAMES, 5)
    assert rows == [{"case_report_id": "real-1"}]


def test_hf_json_uses_env_dataset_server(monkeypatch):
    seen = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {}

    def fake_get(url, params, timeout):
        seen["url"] = url
        seen["params"] = params
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("HF_DATASET_SERVER", "https://hf-mirror.example/api")
    monkeypatch.setattr(data_sources.requests, "get", fake_get)
    data_sources.hf_json("rows", {"dataset": "demo"})
    assert seen["url"] == "https://hf-mirror.example/api/rows"

