from __future__ import annotations

import json
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
        max_tokens: int = 1200,
    ) -> LLMResult:
        if not self.config.enabled:
            raise LLMError("DeepSeek API config is not enabled.")
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        started = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.config.timeout)
        except requests.RequestException as exc:
            raise LLMError(f"DeepSeek request failed: {exc}") from exc
        latency = time.time() - started
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
            },
            latency_s=latency,
        )


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
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {text[:300]}")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON output is not an object.")
    return obj
