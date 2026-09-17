from __future__ import annotations

from types import SimpleNamespace

import pytest

from medimem.llm import CompletionBudgetClient, LLMError, LLMResult


class FakeClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_tokens=128)

    def chat(self, messages, *, temperature=0.1, max_tokens=None, json_mode=True):
        used = min(int(max_tokens), 40)
        return LLMResult(
            text='{"ok": true}',
            usage={"prompt_tokens": 10, "completion_tokens": used, "total_tokens": 10 + used, "calls": 1, "latency_ms": 5},
            latency_s=0.005,
        )


def test_completion_budget_is_cumulative() -> None:
    client = CompletionBudgetClient(FakeClient(), 60)
    first = client.chat([], max_tokens=50)
    second = client.chat([], max_tokens=50)
    assert first.usage["completion_tokens"] == 40
    assert second.usage["completion_tokens"] == 20
    assert client.remaining_completion_tokens == 0
    assert client.cumulative_usage["completion_tokens"] == 60
    assert client.cumulative_usage["calls"] == 2
    with pytest.raises(LLMError):
        client.chat([], max_tokens=1)


def test_completion_budget_preserves_reserved_tokens() -> None:
    client = CompletionBudgetClient(FakeClient(), 100)
    client.reserved_completion_tokens = 60
    result = client.chat([], max_tokens=80)
    assert result.usage["completion_tokens"] == 40
    assert client.remaining_completion_tokens == 60
    with pytest.raises(LLMError):
        client.chat([], max_tokens=1)
    client.reserved_completion_tokens = 0
    client.chat([], max_tokens=20)
    assert client.cumulative_usage["completion_tokens"] == 60

