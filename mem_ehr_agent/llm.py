from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import DeepSeekConfig


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    usage: dict[str, int]
    latency_s: float


class DeepSeekClient:
    def __init__(self, config: DeepSeekConfig):
        self.config = config

    def healthcheck(self) -> tuple[bool, str]:
        if not self.config.enabled:
            return False, "DeepSeek config missing DEEPSEEK_API_KEY or model."
        try:
            result = self.chat(
                [
                    {"role": "system", "content": "Return exactly OK."},
                    {"role": "user", "content": "ping"},
                ],
                temperature=0,
                max_tokens=8,
                json_mode=False,
            )
            text = result.text.strip()
            # Some OpenAI-compatible gateways answer health pings with "PONG".
            # A successful authenticated completion is enough for runtime readiness.
            return bool(text), text
        except Exception as exc:  # noqa: BLE001 - surfaced as blocker text
            return False, str(exc)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> LLMResult:
        if not self.config.enabled:
            raise LLMError("DeepSeek API config is not enabled.")
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        request_seed = os.environ.get("MEDICAL_LLM_SEED", "").strip()
        if request_seed:
            payload["seed"] = int(request_seed)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        started = time.time()
        response = None
        retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        last_error: Exception | None = None
        for token_attempt in range(4):
            for attempt in range(4):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
                except requests.RequestException as exc:
                    last_error = exc
                    if attempt == 3:
                        raise LLMError(f"DeepSeek request failed after retries: {exc}") from exc
                    time.sleep(2**attempt)
                    continue
                if response.status_code not in retryable_statuses:
                    break
                if attempt == 3:
                    break
                time.sleep(2**attempt)
            if response is None:
                break
            body = response.text[:500] if response.status_code >= 400 else ""
            if (
                response.status_code == 400
                and "maximum context length" in body
                and int(payload["max_tokens"]) > 128
                and token_attempt < 3
            ):
                payload["max_tokens"] = max(128, int(payload["max_tokens"]) - 256)
                continue
            break
        latency = time.time() - started
        if response is None:
            raise LLMError(f"DeepSeek request failed: {last_error}")
        if response.status_code >= 400:
            body = response.text[:500]
            raise LLMError(f"DeepSeek API HTTP {response.status_code}: {body}")
        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected DeepSeek response shape: {data}") from exc
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "calls": 1,
                "latency_ms": int(round(latency * 1000)),
            },
            latency_s=latency,
        )


class CompletionBudgetClient:
    """Per-case wrapper that enforces a cumulative generated-token budget."""

    def __init__(self, client: DeepSeekClient, completion_token_budget: int):
        if completion_token_budget <= 0:
            raise ValueError("completion_token_budget must be positive")
        self.client = client
        self.config = client.config
        self.completion_token_budget = int(completion_token_budget)
        self.remaining_completion_tokens = int(completion_token_budget)
        self.reserved_completion_tokens = 0
        self.cumulative_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "latency_ms": 0,
        }

    def healthcheck(self) -> tuple[bool, str]:
        return self.client.healthcheck()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> LLMResult:
        available = self.remaining_completion_tokens - self.reserved_completion_tokens
        if available <= 0:
            raise LLMError("Per-case completion-token budget exhausted.")
        requested = int(max_tokens if max_tokens is not None else self.config.max_tokens)
        allowed = min(requested, available)
        result = self.client.chat(
            messages,
            temperature=temperature,
            max_tokens=allowed,
            json_mode=json_mode,
        )
        used = int(result.usage.get("completion_tokens", 0) or 0)
        self.remaining_completion_tokens = max(0, self.remaining_completion_tokens - used)
        for key in self.cumulative_usage:
            self.cumulative_usage[key] += int(result.usage.get(key, 0) or 0)
        result.usage["completion_budget"] = self.completion_token_budget
        result.usage["completion_budget_remaining"] = self.remaining_completion_tokens
        result.usage["completion_budget_reserved"] = self.reserved_completion_tokens
        return result


def with_completion_budget(
    client: DeepSeekClient | CompletionBudgetClient | None, completion_token_budget: int | None
) -> DeepSeekClient | CompletionBudgetClient | None:
    if client is None or completion_token_budget is None:
        return client
    base = client.client if isinstance(client, CompletionBudgetClient) else client
    return CompletionBudgetClient(base, completion_token_budget)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    candidate = first_balanced_json_object(cleaned)
    if candidate is None:
        raise ValueError(f"No JSON object found in LLM output: {text[:300]}")
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON output is not an object.")
    return obj


def first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find("{", start + 1)
    return None
