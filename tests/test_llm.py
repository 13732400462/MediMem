from mem_ehr_agent.llm import extract_json_object


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
