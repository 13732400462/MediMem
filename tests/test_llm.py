from medimem.config import get_deepseek_config
from medimem.llm import extract_json_object


def test_extract_json_object_accepts_markdown_and_explanation():
    text = """Here is the result:

```json
{"primary_diagnosis": "X", "evidence": "single string"}
```
"""
    assert extract_json_object(text)["primary_diagnosis"] == "X"


def test_extract_json_object_ignores_braces_inside_strings():
    text = 'prefix {"reasoning_summary": "mentions {not object}", "confidence": 0.7} suffix'
    assert extract_json_object(text)["confidence"] == 0.7


def test_config_reads_api_key_from_file_without_repr_leak(tmp_path, monkeypatch):
    key_path = tmp_path / "api.key"
    key_path.write_text("secret-test-key\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY_FILE", str(key_path))
    monkeypatch.setenv("DEEPSEEK_MODEL", "test-model")
    cfg = get_deepseek_config(env_path=tmp_path / "missing.env")
    assert cfg.api_key == "secret-test-key"
    assert "secret-test-key" not in repr(cfg)


def test_config_explicit_api_key_takes_precedence_over_key_file(tmp_path, monkeypatch):
    key_path = tmp_path / "api.key"
    key_path.write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY_FILE", str(key_path))
    cfg = get_deepseek_config(env_path=tmp_path / "missing.env")
    assert cfg.api_key == "environment-key"

