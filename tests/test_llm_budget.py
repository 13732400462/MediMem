from __future__ import annotations

from types import SimpleNamespace

import pytest

from mem_ehr_agent.llm import CompletionBudgetClient, LLMError, LLMResult


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
    with pytest.raises(LLMError):
        client.chat([], max_tokens=1)
