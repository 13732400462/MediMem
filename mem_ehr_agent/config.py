from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class DeepSeekConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout: int = 180
    max_tokens: int = 700

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


def get_deepseek_config(env_path: str | Path = ".env") -> DeepSeekConfig:
    load_dotenv(env_path)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        key_file = os.getenv("DEEPSEEK_API_KEY_FILE", "").strip()
        if key_file:
            path = Path(key_file)
            if path.is_file():
                api_key = path.read_text(encoding="utf-8").strip()
    return DeepSeekConfig(
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.shunyu.tech/v1").rstrip("/"),
        api_key=api_key,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v3.2"),
        timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "180")),
        max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "700")),
    )
