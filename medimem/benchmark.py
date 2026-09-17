from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .amem_baseline import SourceAlignedAMEMSystem
from .config import DeepSeekConfig, get_deepseek_config
from .io_utils import ensure_dir, read_jsonl, write_jsonl, write_text
from .llm import DeepSeekClient, extract_json_object
from .memory import MemoryStore, text_similarity
from .metrics import token_f1

LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

LOCOMO_MAIN_METRICS = ("qa_f1", "bleu1", "avg_tokens")
LOCOMO_AUXILIARY_METRICS = ("exact_match", "soft_match", "retrieved_memory_count", "guard_pass_rate", "fallback_rate")
LOCOMO_OFFICIAL_STYLE_METRICS = (
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "meteor",
    "sbert_similarity",
)
_SBERT_MODEL: Any | None = None
_SBERT_UNAVAILABLE_REASON: str | None = None


@dataclass(frozen=True)
class NativeBenchmarkSpec:
    name: str
    track: str
    default_path: str
    official_methods: tuple[str, ...]
    official_repo: str
    required_env: str | None = None
    notes: str = ""


NATIVE_BENCHMARKS: dict[str, NativeBenchmarkSpec] = {
    "locomo": NativeBenchmarkSpec(
        name="locomo",
        track="long_term_memory_qa",
        default_path="../datasets/amem_original/locomo/locomo10.official.json",
        official_methods=("amem", "memoryos", "meminsight"),
        official_repo="https://github.com/snap-research/locomo",
        notes="LoCoMo is the original long-conversation memory benchmark used by A-MEM-style evaluation.",
    ),
    "dialsim": NativeBenchmarkSpec(
        name="dialsim",
        track="dialogue_simulation_memory",
        default_path="../datasets/amem_original/dialsim",
        official_methods=("amem",),
        official_repo="https://huggingface.co/datasets/jiho283",
        notes="DialSim is kept in its native parquet subsets; pyarrow is required to read it locally.",
    ),
    "memoryos_native": NativeBenchmarkSpec(
        name="memoryos_native",
        track="official_external",
        default_path="data/native/memoryos",
        official_methods=("memoryos",),
        official_repo="https://github.com/BAI-LAB/MemoryOS",
        required_env="MEMORYOS_REPO",
        notes="Runs only when the official MemoryOS repository/data path is configured.",
    ),
    "meminsight_native": NativeBenchmarkSpec(
        name="meminsight_native",
        track="official_external",
        default_path="data/native/meminsight",
        official_methods=("meminsight",),
        official_repo="https://github.com/amazon-science/MemInsight",
        required_env="MEMINSIGHT_REPO",
        notes="Runs only when the official MemInsight repository/data path is configured.",
    ),
    "gmemory_native": NativeBenchmarkSpec(
        name="gmemory_native",
        track="official_external",
        default_path="data/native/gmemory",
        official_methods=("gmemory",),
        official_repo="https://github.com/bingreeky/GMemory",
        required_env="GMEMORY_REPO",
        notes="Runs only when the official G-Memory repository/data path is configured.",
    ),
    "ddo_native": NativeBenchmarkSpec(
        name="ddo_native",
        track="medical_consultation",
        default_path="data/native/ddo",
        official_methods=("ddo",),
        official_repo="https://github.com/zh-jia/DDO",
        required_env="DDO_REPO",
        notes="Runs only when the official DDO repository/data path is configured.",
    ),
}


LOCAL_METHODS = {"direct", "ours", "medimem", "amem"}
OFFICIAL_WRAPPER_METHODS = {"memoryos", "meminsight", "gmemory", "ddo"}
BASELINE_ENV_ROOT = Path("/home/syh/A-mem/baseline_envs")
DEFAULT_BASELINE_REPOS = {
    "memoryos": BASELINE_ENV_ROOT / "repos" / "MemoryOS",
    "meminsight": BASELINE_ENV_ROOT / "repos" / "MemInsight",
    "gmemory": BASELINE_ENV_ROOT / "repos" / "GMemory",
    "ddo": BASELINE_ENV_ROOT / "repos" / "DDO",
}
DEFAULT_BASELINE_PYTHONS = {
    "memoryos": BASELINE_ENV_ROOT / "conda" / "memoryos" / "bin" / "python",
    "meminsight": BASELINE_ENV_ROOT / "conda" / "meminsight" / "bin" / "python",
    "gmemory": BASELINE_ENV_ROOT / "conda" / "gmemory" / "bin" / "python",
    "ddo": BASELINE_ENV_ROOT / "conda" / "ddo" / "bin" / "python",
}
BASELINE_REPRODUCTION_LEVELS = {
    "memoryos": "official_native_locomo_wrapper",
    "meminsight": "official_native_locomo_wrapper",
    "gmemory": "official_component_locomo_wrapper",
    "ddo": "official_component_locomo_wrapper",
}
BASELINE_ADAPTER_NOTES = {
    "memoryos": "MemoryOS has LoCoMo evaluation files; this run uses the project-unified LoCoMo sample/metric wrapper.",
    "meminsight": "MemInsight has LoCoMo QA utilities; this run uses the project-unified LoCoMo sample/metric wrapper.",
    "gmemory": "GMemory has no native LoCoMo task entrypoint; this is a component-level LoCoMo wrapper, not original-paper reproduction.",
    "ddo": "DDO is a medical consultation pipeline with no native LoCoMo entrypoint; this is a component-level LoCoMo wrapper, not original-paper reproduction.",
}
BASELINE_METHOD_LABELS = {
    "amem": "official_amem_locomo_wrapper",
    "memoryos": "official_memoryos_locomo_wrapper",
    "meminsight": "official_meminsight_locomo_wrapper",
    "gmemory": "official_gmemory_locomo_wrapper",
    "ddo": "official_ddo_locomo_wrapper",
}
ENV_BY_METHOD = {
    "memoryos": "MEMORYOS_REPO",
    "meminsight": "MEMINSIGHT_REPO",
    "gmemory": "GMEMORY_REPO",
    "ddo": "DDO_REPO",
}
PY_ENV_BY_METHOD = {
    "memoryos": "MEMORYOS_PY",
    "meminsight": "MEMINSIGHT_PY",
    "gmemory": "GMEMORY_PY",
    "ddo": "DDO_PY",
}
METHOD_OFFICIAL_REPOS = {
    "amem": "https://github.com/agiresearch/A-mem",
    "memoryos": "https://github.com/BAI-LAB/MemoryOS",
    "meminsight": "https://github.com/amazon-science/MemInsight",
    "gmemory": "https://github.com/bingreeky/GMemory",
    "ddo": "https://github.com/zh-jia/DDO",
}


FULL_CONTEXT_SHORTCUT_ADVICE = (
    "Modification advice: use run_locomo_ours_memory_pipeline, build one memory card per LoCoMo turn, "
    "then pass only retrieved memories to the QA prompt; do not pass sample['context'] or [FULL LONG-TERM CONTEXT]."
)


def make_benchmark_run_dir(root: str | Path = "runs", *, prefix: str = "benchmark") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{prefix}_{stamp}"
    ensure_dir(path / "predictions")
    return path


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one benchmark method is required.")
    return methods


def parse_int_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("top-k values must be positive integers.")
        values.append(value)
    return values


def build_client(require_api: bool = False) -> tuple[DeepSeekClient | None, str | None]:
    client = DeepSeekClient(get_deepseek_config())
    ok, msg = client.healthcheck()
    if ok:
        return client, None
    if require_api:
        raise RuntimeError(f"DeepSeek API unavailable: {msg}")
    return None, f"DeepSeek API unavailable, using deterministic QA fallback: {msg}"


def build_client_for_base_url(base_url: str | None, *, require_api: bool = False) -> tuple[DeepSeekClient | None, str | None]:
    base_config = get_deepseek_config()
    if base_url:
        config = DeepSeekConfig(
            base_url=base_url.rstrip("/"),
            api_key=base_config.api_key,
            model=base_config.model,
            timeout=base_config.timeout,
            max_tokens=base_config.max_tokens,
        )
    else:
        config = base_config
    client = DeepSeekClient(config)
    ok, message = client.healthcheck()
    if ok:
        return client, None
    if require_api:
        raise RuntimeError(f"Benchmark API healthcheck failed: {message}")
    return None, message


def normalize_answer(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def simple_bleu1(pred: str, gold: str) -> float:
    pred_tokens = re.findall(r"[a-z0-9]+", pred.lower())
    gold_tokens = re.findall(r"[a-z0-9]+", gold.lower())
    if not pred_tokens or not gold_tokens:
        return 0.0
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        count = gold_counts.get(token, 0)
        if count:
            overlap += 1
            gold_counts[token] = count - 1
    return overlap / len(pred_tokens)


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if n <= 0 or len(tokens) < n:
        return counts
    for idx in range(len(tokens) - n + 1):
        gram = tuple(tokens[idx : idx + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def modified_ngram_precision(pred_tokens: list[str], gold_tokens: list[str], n: int) -> float:
    pred_counts = ngram_counts(pred_tokens, n)
    gold_counts = ngram_counts(gold_tokens, n)
    if not pred_counts:
        return 0.0
    overlap = sum(min(count, gold_counts.get(gram, 0)) for gram, count in pred_counts.items())
    return overlap / sum(pred_counts.values())


def simple_bleu_n(pred: str, gold: str, n: int) -> float:
    pred_tokens = simple_tokens(pred)
    gold_tokens = simple_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    precisions = [modified_ngram_precision(pred_tokens, gold_tokens, order) for order in range(1, n + 1)]
    if any(score <= 0 for score in precisions):
        return 0.0
    brevity_penalty = 1.0 if len(pred_tokens) > len(gold_tokens) else math.exp(1 - len(gold_tokens) / max(len(pred_tokens), 1))
    geo_mean = math.exp(sum((1 / n) * math.log(score) for score in precisions))
    return brevity_penalty * geo_mean


def rouge_n_f(pred: str, gold: str, n: int) -> float:
    pred_counts = ngram_counts(simple_tokens(pred), n)
    gold_counts = ngram_counts(simple_tokens(gold), n)
    if not pred_counts or not gold_counts:
        return 0.0
    overlap = sum(min(count, gold_counts.get(gram, 0)) for gram, count in pred_counts.items())
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(gold_counts.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def lcs_len(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for token in left:
        curr = [0] * (len(right) + 1)
        for idx, other in enumerate(right, start=1):
            curr[idx] = prev[idx - 1] + 1 if token == other else max(prev[idx], curr[idx - 1])
        prev = curr
    return prev[-1]


def rouge_l_f(pred: str, gold: str) -> float:
    pred_tokens = simple_tokens(pred)
    gold_tokens = simple_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = lcs_len(pred_tokens, gold_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def lightweight_meteor(pred: str, gold: str) -> float:
    pred_tokens = simple_tokens(pred)
    gold_tokens = simple_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = ngram_counts(pred_tokens, 1)
    gold_counts = ngram_counts(gold_tokens, 1)
    overlap = sum(min(count, gold_counts.get(gram, 0)) for gram, count in pred_counts.items())
    if not overlap:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return (10 * precision * recall) / (recall + 9 * precision) if recall + 9 * precision else 0.0


def optional_sbert_similarity(pred: str, gold: str) -> tuple[float | None, str | None]:
    global _SBERT_MODEL, _SBERT_UNAVAILABLE_REASON
    if _SBERT_UNAVAILABLE_REASON:
        return None, _SBERT_UNAVAILABLE_REASON
    try:
        from sentence_transformers.util import pytorch_cos_sim  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _SBERT_UNAVAILABLE_REASON = f"sentence-transformers unavailable: {exc.__class__.__name__}"
        return None, _SBERT_UNAVAILABLE_REASON
    try:
        if _SBERT_MODEL is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        left = _SBERT_MODEL.encode([str(pred)], convert_to_tensor=True)
        right = _SBERT_MODEL.encode([str(gold)], convert_to_tensor=True)
        return float(pytorch_cos_sim(left, right).item()), None
    except Exception as exc:  # noqa: BLE001
        _SBERT_UNAVAILABLE_REASON = f"sentence-transformers failed: {exc.__class__.__name__}"
        return None, _SBERT_UNAVAILABLE_REASON


def locomo_category_name(value: Any) -> str:
    try:
        category = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return LOCOMO_CATEGORY_NAMES.get(category, f"category-{category}")


def soft_match(pred: str, gold: str) -> bool:
    p = normalize_answer(pred).lower()
    g = normalize_answer(gold).lower()
    if not p or not g:
        return False
    return p == g or p in g or g in p or token_f1(p, g) >= 0.72


def turn_to_text(turn: dict[str, Any]) -> str:
    text = str(turn.get("text") or "").strip()
    caption = str(turn.get("blip_caption") or "").strip()
    if caption:
        text = f"[Image: {caption}] {text}".strip()
    return text


def locomo_turns(sample: dict[str, Any], *, sample_index: int = 0, sample_id: str | None = None) -> list[dict[str, Any]]:
    conv = sample.get("conversation") or {}
    turns: list[dict[str, Any]] = []
    conversation_id = str(sample_id or sample.get("sample_id") or sample.get("conversation_id") or f"conv-{sample_index}")
    for key in sorted(conv, key=lambda item: (len(item), item)):
        if not re.fullmatch(r"session_\d+", str(key)):
            continue
        session = conv.get(key)
        if not isinstance(session, list):
            continue
        session_id = str(key).split("_")[-1]
        date = str(conv.get(f"{key}_date_time", ""))
        for idx, turn in enumerate(session):
            if not isinstance(turn, dict):
                continue
            text = turn_to_text(turn)
            dia_id = str(turn.get("dia_id") or f"D{session_id}:{idx}")
            turns.append(
                {
                    "event_id": dia_id,
                    "time": f"D{session_id}:{idx}",
                    "session": session_id,
                    "session_date": date,
                    "speaker": str(turn.get("speaker") or ""),
                    "text": text,
                    "dia_id": dia_id,
                    "tags": ["dialogue", "locomo"],
                    "conversation_id": conversation_id,
                }
            )
    return turns


def conversation_context(sample: dict[str, Any], *, max_turns: int | None = None) -> str:
    conv = sample.get("conversation") or {}
    speakers = [str(conv.get("speaker_a") or "speaker_a"), str(conv.get("speaker_b") or "speaker_b")]
    lines = [f"speakers: {', '.join(speakers)}"]
    for key in sorted(conv, key=lambda item: (len(item), item)):
        if not re.fullmatch(r"session_\d+", str(key)):
            continue
        session = conv.get(key)
        if not isinstance(session, list):
            continue
        session_id = str(key).split("_")[-1]
        date = conv.get(f"{key}_date_time", "")
        lines.append(f"[D{session_id}] {date}")
        for idx, turn in enumerate(session):
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker", "")
            text = turn_to_text(turn)
            lines.append(f"D{session_id}:{idx} {speaker}: {text}")
            if max_turns and len(lines) >= max_turns:
                return "\n".join(lines)
    return "\n".join(lines)


def load_locomo_samples(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for sample_idx, item in enumerate(data):
        context = conversation_context(item)
        conversation_id = str(item.get("sample_id") or item.get("conversation_id") or f"conv-{sample_idx}")
        turns = locomo_turns(item, sample_index=sample_idx, sample_id=conversation_id)
        for qa_idx, qa in enumerate(item.get("qa", [])):
            answer = qa.get("adversarial_answer") if qa.get("category") == 5 and qa.get("adversarial_answer") else qa.get("answer")
            category = qa.get("category")
            samples.append(
                {
                    "sample_id": f"{conversation_id}__qa_{qa_idx:04d}",
                    "conversation_id": conversation_id,
                    "dataset": "locomo",
                    "split": "official",
                    "category": category,
                    "category_name": locomo_category_name(category),
                    "context": context,
                    "turns": turns,
                    "context_line_count": len([line for line in context.splitlines() if line.strip()]),
                    "turn_count": len(turns),
                    "question": normalize_answer(qa.get("question")),
                    "answer": normalize_answer(answer),
                    "evidence": qa.get("evidence", []),
                    "metadata": {"conversation_index": sample_idx, "qa_index": qa_idx, "category": category},
                }
            )
            if limit and len(samples) >= limit:
                return samples
    return samples


def sample_benchmark_rows(samples: list[dict[str, Any]], *, sample_n: int | None = None, random_seed: int | None = None) -> list[dict[str, Any]]:
    if sample_n is None:
        return samples
    if sample_n >= len(samples):
        return list(samples)
    rng = random.Random(random_seed)
    return rng.sample(samples, sample_n)


def load_dialsim_samples(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("DialSim native loading requires pyarrow. Install pyarrow on the server to run this benchmark.") from exc
    root = Path(path)
    parquet_files = sorted(root.rglob("*.parquet"))
    samples: list[dict[str, Any]] = []
    for parquet_path in parquet_files:
        subset = parquet_path.parent.name
        table = pq.read_table(parquet_path)
        for row_idx, row in enumerate(table.to_pylist()):
            context = normalize_answer(row.get("context") or row.get("dialogue") or row.get("conversation") or row)
            question = normalize_answer(row.get("question") or row.get("query") or "What should be remembered from this dialogue?")
            answer = normalize_answer(row.get("answer") or row.get("target") or row.get("response") or row.get("summary"))
            if not context or not answer:
                continue
            samples.append(
                {
                    "sample_id": f"{subset}__{parquet_path.stem}__{row_idx:06d}",
                    "dataset": "dialsim",
                    "split": subset,
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "evidence": [],
                    "metadata": {"subset": subset, "source_file": str(parquet_path)},
                }
            )
            if limit and len(samples) >= limit:
                return samples
    return samples


def load_jsonl_native_samples(dataset: str, path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    samples: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        context = normalize_answer(row.get("context") or row.get("input") or row.get("conversation") or row.get("dialogue") or row.get("case"))
        question = normalize_answer(row.get("question") or row.get("query") or row.get("instruction") or "Answer from the provided context.")
        answer = normalize_answer(row.get("answer") or row.get("target") or row.get("output") or row.get("label"))
        if not context or not answer:
            continue
        samples.append(
            {
                "sample_id": str(row.get("sample_id") or row.get("id") or f"{dataset}_{idx:06d}"),
                "dataset": dataset,
                "split": str(row.get("split") or "native"),
                "context": context,
                "question": question,
                "answer": answer,
                "evidence": row.get("evidence", []),
                "metadata": row.get("metadata", {}),
            }
        )
        if limit and len(samples) >= limit:
            break
    return samples


def load_native_samples(dataset: str, path: str | Path | None = None, *, limit: int | None = None) -> list[dict[str, Any]]:
    if dataset not in NATIVE_BENCHMARKS:
        raise ValueError(f"Unknown benchmark dataset {dataset}. Known: {sorted(NATIVE_BENCHMARKS)}")
    spec = NATIVE_BENCHMARKS[dataset]
    source_path = Path(path or spec.default_path)
    if dataset == "locomo":
        return load_locomo_samples(source_path, limit=limit)
    if dataset == "dialsim":
        return load_dialsim_samples(source_path, limit=limit)
    if source_path.is_file() and source_path.suffix.lower() == ".jsonl":
        return load_jsonl_native_samples(dataset, source_path, limit=limit)
    raise RuntimeError(
        f"{dataset} native benchmark requires official data at {source_path}. "
        f"Repo: {spec.official_repo}. {spec.notes}"
    )


def truncate_text(text: str, max_chars: int = 18000) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head].rstrip() + "\n[... truncated ...]\n" + text[-tail:].lstrip()


def qa_prompt(sample: dict[str, Any], method: str, context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You answer long-term memory benchmark questions. Use only the provided context. "
                "Return exactly one minified JSON object with keys answer and confidence. "
                "answer must be concise and no longer than 12 words. "
                "confidence must be a number from 0 to 1. Always close the JSON object. "
                "Do not include evidence, markdown, or prose outside the JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[DATASET]\n{sample['dataset']}\n[METHOD]\n{method}\n[CONTEXT]\n{truncate_text(context)}\n\n"
                f"[QUESTION]\n{sample['question']}\n\nReturn JSON only."
            ),
        },
    ]


def deterministic_qa_fallback(sample: dict[str, Any], method: str, reason: str = "", *, context: str | None = None) -> dict[str, Any]:
    context = context if context is not None else sample.get("context", "")
    question_tokens = {tok for tok in re.findall(r"[a-z0-9]+", sample.get("question", "").lower()) if len(tok) > 3}
    best = ""
    best_score = -1
    for sentence in re.split(r"(?<=[.;?!])\s+|\n+", context):
        tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        score = len(tokens & question_tokens)
        if score > best_score:
            best = sentence.strip()
            best_score = score
    return {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "split": sample["split"],
        "method": method,
        "answer": best[:180] or "Unknown",
        "evidence": [best[:180]] if best else [],
        "confidence": 0.25,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "fallback_reason": reason or "No LLM client available.",
    }


def answer_with_context(
    sample: dict[str, Any],
    method: str,
    context: str,
    client: DeepSeekClient | None,
    *,
    fail_on_llm_error: bool = False,
) -> dict[str, Any]:
    if client is None:
        if fail_on_llm_error:
            raise RuntimeError(f"Benchmark QA client unavailable for {sample['sample_id']} ({method}).")
        return deterministic_qa_fallback(sample, method, context=context)
    try:
        result = client.chat(
            qa_prompt(sample, method, context),
            temperature=0.0,
            max_tokens=int(os.getenv("BENCHMARK_MAX_TOKENS", "256")),
        )
    except Exception as exc:  # noqa: BLE001
        if fail_on_llm_error:
            raise RuntimeError(f"Benchmark QA failed for {sample['sample_id']} ({method}): {exc}") from exc
        return deterministic_qa_fallback(sample, method, reason=str(exc), context=context)
    try:
        raw = extract_json_object(result.text)
        evidence = raw.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        try:
            confidence = float(raw.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = {"low": 0.25, "medium": 0.5, "moderate": 0.5, "high": 0.75}.get(
                str(raw.get("confidence", "")).strip().lower(),
                0.5,
            )
        return {
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "split": sample["split"],
            "method": method,
            "answer": normalize_answer(raw.get("answer")),
            "evidence": [str(item) for item in evidence],
            "confidence": confidence,
            "usage": result.usage,
        }
    except Exception as exc:  # noqa: BLE001
        if fail_on_llm_error:
            raise RuntimeError(f"Benchmark QA returned malformed JSON for {sample['sample_id']} ({method}): {exc}") from exc
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', result.text)
        recovered_answer = answer_match.group(1) if answer_match else result.text.strip()
        return {
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "split": sample["split"],
            "method": method,
            "answer": normalize_answer(recovered_answer[:500]) or "Unknown",
            "evidence": [],
            "confidence": 0.35,
            "usage": result.usage,
            "fallback_reason": f"Recovered from malformed JSON output: {exc}",
        }


def locomo_memory_path(run_dir: str | Path, conversation_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", conversation_id).strip("_") or "conversation"
    return Path(run_dir) / "memory" / "ours" / f"{safe_id}.memory.jsonl"


def locomo_cache_key(sample: dict[str, Any], *, dataset_hash: str, top_k: int, coarse_k: int) -> str:
    raw_id = str(sample.get("sample_id") or sample.get("conversation_id") or "")
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw_id).strip("_") or "sample"
    return f"{safe_id}__{dataset_hash[:12]}__top{top_k}__coarse{coarse_k}"


def locomo_dataset_hash(samples: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for sample in samples:
        payload = {
            "sample_id": sample.get("sample_id"),
            "conversation_id": sample.get("conversation_id"),
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "turn_count": len(sample.get("turns") or []),
        }
        h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def locomo_cached_memory_path(
    cache_dir: str | Path,
    sample: dict[str, Any],
    *,
    dataset_hash: str,
    top_k: int,
    coarse_k: int,
) -> Path:
    return Path(cache_dir) / "locomo_memory_store" / f"{locomo_cache_key(sample, dataset_hash=dataset_hash, top_k=top_k, coarse_k=coarse_k)}.memory.jsonl"


def build_cached_locomo_memory_store(
    sample: dict[str, Any],
    cache_dir: str | Path | None,
    *,
    dataset_hash: str,
    top_k: int,
    coarse_k: int,
    fallback_path: str | Path,
) -> MemoryStore:
    if cache_dir is None:
        return build_locomo_memory_store(sample, fallback_path)
    memory_path = locomo_cached_memory_path(cache_dir, sample, dataset_hash=dataset_hash, top_k=top_k, coarse_k=coarse_k)
    ensure_dir(memory_path.parent)
    lock_path = memory_path.with_suffix(memory_path.suffix + ".lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if memory_path.exists():
                store = MemoryStore.load(str(sample.get("conversation_id") or sample["sample_id"]), memory_path)
                if store.cards:
                    return store
            time.sleep(0.05)
    try:
        return build_locomo_memory_store(sample, memory_path)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def build_locomo_memory_store(sample: dict[str, Any], memory_path: str | Path) -> MemoryStore:
    store = MemoryStore.load(str(sample.get("conversation_id") or sample["sample_id"]), memory_path)
    if store.cards:
        return store
    wrote_card = False
    for turn in sample.get("turns", []):
        text = normalize_answer(turn.get("text"))
        if not text:
            continue
        store.write_card(
            summary=text,
            evidence_refs=[str(turn.get("dia_id") or turn.get("event_id") or "")],
            time_scope={
                "session": turn.get("session"),
                "time": turn.get("time"),
                "date": turn.get("session_date"),
            },
            confidence=0.72,
            tags=["dialogue", "locomo", str(turn.get("speaker") or "").strip()],
            status="active",
            op="Write",
        )
        store.cards[-1]["speaker"] = str(turn.get("speaker") or "").strip()
        store.cards[-1]["entities"] = extract_locomo_entities(text)
        store.cards[-1]["temporal_markers"] = extract_locomo_temporal_markers(
            text,
            turn.get("session_date"),
            turn.get("time"),
        )
        wrote_card = True
    if wrote_card:
        store.save()
    return store


def format_locomo_memory_line(memory: dict[str, Any]) -> str:
    refs = ",".join(str(item) for item in memory.get("evidence_refs", []))
    tags = ",".join(str(item) for item in memory.get("tags", []) if str(item).strip())
    scope = memory.get("time_scope") or {}
    entities = ",".join(str(item) for item in memory.get("entities", []) if str(item).strip())
    temporal = ",".join(str(item) for item in memory.get("temporal_markers", []) if str(item).strip())
    return (
        f"- id={memory.get('memory_id')} refs={refs} "
        f"session={scope.get('session')} date={scope.get('date')} time={scope.get('time')} tags={tags} "
        f"entities={entities} temporal={temporal} "
        f"text={memory.get('summary')}"
    )


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
TEMPORAL_TERMS = {
    "after",
    "before",
    "during",
    "earlier",
    "eventually",
    "first",
    "later",
    "last",
    "latest",
    "next",
    "previous",
    "recent",
    "recently",
    "then",
    "today",
    "tomorrow",
    "week",
    "year",
    "yesterday",
}


def extract_locomo_terms(text: Any) -> list[str]:
    raw = str(text or "")
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'_-]*|\d+", raw):
        norm = token.lower().strip("'_-")
        if len(norm) <= 1 or norm in STOPWORDS or norm in seen:
            continue
        seen.add(norm)
        terms.append(norm)
    return terms


def extract_locomo_entities(text: Any) -> list[str]:
    raw = str(text or "")
    entities: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", raw):
        entity = match.group(0).strip()
        norm = entity.lower()
        if norm in STOPWORDS or norm in seen:
            continue
        seen.add(norm)
        entities.append(entity)
    return entities


def extract_locomo_temporal_markers(*values: Any) -> list[str]:
    text = " ".join(str(value or "") for value in values)
    markers: list[str] = []
    seen: set[str] = set()
    patterns = [
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b(?:today|tomorrow|yesterday|last|next|later|earlier|before|after|recently)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            marker = match.group(0).strip()
            norm = marker.lower()
            if norm in seen:
                continue
            seen.add(norm)
            markers.append(marker)
    return markers


def locomo_expanded_query(sample: dict[str, Any]) -> str:
    question = str(sample.get("question") or "")
    parts = [question]
    category_name = str(sample.get("category_name") or locomo_category_name(sample.get("category")) or "")
    if category_name:
        parts.append(category_name)
    if category_name in {"temporal", "multi-hop"} or any(term in extract_locomo_terms(question) for term in TEMPORAL_TERMS):
        parts.append("time date session before after later earlier")
    entities = extract_locomo_entities(question)
    if entities:
        parts.append(" ".join(entities))
    return " ".join(part for part in parts if part).strip()


def locomo_memory_retrieval_text(memory: dict[str, Any]) -> str:
    scope = memory.get("time_scope") or {}
    fields = [
        memory.get("summary"),
        " ".join(str(item) for item in memory.get("entities", [])),
        " ".join(str(item) for item in memory.get("temporal_markers", [])),
        " ".join(str(item) for item in memory.get("tags", [])),
        scope.get("session"),
        scope.get("date"),
        scope.get("time"),
        " ".join(str(item) for item in memory.get("evidence_refs", [])),
    ]
    return " ".join(str(field) for field in fields if field)


def locomo_retrieval_score(query: str, sample: dict[str, Any], memory: dict[str, Any]) -> float:
    score = text_similarity(query, locomo_memory_retrieval_text(memory))
    query_terms = set(extract_locomo_terms(query))
    card_entities = {term.lower() for term in memory.get("entities", [])}
    if query_terms & card_entities:
        score += 0.12
    temporal_query = bool(query_terms & TEMPORAL_TERMS) or sample.get("category_name") == "temporal"
    if temporal_query and memory.get("temporal_markers"):
        score += 0.08
    evidence_terms = {term.lower() for term in memory.get("evidence_refs", [])}
    if query_terms & evidence_terms:
        score += 0.05
    return score


def retrieve_locomo_memories(
    store: MemoryStore,
    sample: dict[str, Any],
    *,
    top_k: int = 8,
    coarse_k: int = 32,
) -> list[dict[str, Any]]:
    query = locomo_expanded_query(sample)
    scored: list[tuple[float, dict[str, Any]]] = []
    for card in store.cards:
        if card.get("status") not in {"active", "flagged"}:
            continue
        score = locomo_retrieval_score(query, sample, card)
        if score > 0:
            scored.append((score, card))
    scored.sort(
        key=lambda item: (
            item[0],
            float((item[1].get("confidence") or 0)),
            str((item[1].get("time_scope") or {}).get("time") or ""),
        ),
        reverse=True,
    )
    coarse = scored[: max(top_k, coarse_k)]
    return [card | {"retrieval_score": score} for score, card in coarse[:top_k]]


def assert_no_full_context_shortcut(
    method: str,
    prompt_context: str,
    sample: dict[str, Any],
    *,
    retrieved_memory_count: int | None = None,
) -> None:
    if "ours" not in method:
        return
    if "[FULL LONG-TERM CONTEXT]" in prompt_context:
        raise RuntimeError(f"Full-context shortcut detected for {method}: prompt contains [FULL LONG-TERM CONTEXT]. {FULL_CONTEXT_SHORTCUT_ADVICE}")
    context_line_count = int(sample.get("context_line_count") or len(str(sample.get("context") or "").splitlines()))
    if retrieved_memory_count is not None and context_line_count and retrieved_memory_count == context_line_count:
        raise RuntimeError(
            f"Full-context shortcut detected for {method}: retrieved_memory_count equals original context_line_count ({context_line_count}). "
            f"{FULL_CONTEXT_SHORTCUT_ADVICE}"
        )
    turns = [str(turn.get("text") or "").strip() for turn in sample.get("turns", []) if str(turn.get("text") or "").strip()]
    total_chars = sum(len(text) for text in turns)
    covered_chars = sum(len(text) for text in turns if text and text in prompt_context)
    coverage = covered_chars / total_chars if total_chars else 0.0
    if total_chars and coverage > 0.8 and (retrieved_memory_count or 0) < len(turns):
        raise RuntimeError(
            f"Full-context shortcut detected for {method}: prompt covers {coverage:.1%} of original LoCoMo turn text. "
            f"{FULL_CONTEXT_SHORTCUT_ADVICE}"
        )


def run_locomo_ours_memory_pipeline(
    sample: dict[str, Any],
    client: DeepSeekClient | None,
    *,
    run_dir: str | Path,
    top_k: int = 8,
    coarse_k: int = 32,
    require_api: bool,
    memory_cache_dir: str | Path | None = None,
    dataset_hash: str = "",
) -> dict[str, Any]:
    if sample.get("dataset") != "locomo":
        raise ValueError("run_locomo_ours_memory_pipeline only supports LoCoMo samples.")
    conversation_id = str(sample.get("conversation_id") or sample["sample_id"])
    memory_id = f"{conversation_id}_{sample['sample_id']}"
    memory_path = locomo_memory_path(run_dir, memory_id)
    store = build_cached_locomo_memory_store(
        sample,
        memory_cache_dir,
        dataset_hash=dataset_hash,
        top_k=top_k,
        coarse_k=coarse_k,
        fallback_path=memory_path,
    )
    retrieved = retrieve_locomo_memories(store, sample, top_k=top_k, coarse_k=coarse_k)
    memory_context = "\n".join(format_locomo_memory_line(memory) for memory in retrieved)
    prompt_context = (
        "[RETRIEVED_MEMORY_CARDS]\n"
        f"{memory_context}\n\n"
        "[MEMORY_OPS]\n[]\n\n"
        "[QUESTION]\n"
        f"{sample['question']}\n"
        "Answer using only the retrieved memory cards."
    )
    assert_no_full_context_shortcut(
        "medimem_locomo_memory_pipeline",
        prompt_context,
        sample,
        retrieved_memory_count=len(retrieved),
    )
    pred = answer_with_context(sample, "medimem_locomo_memory_pipeline", prompt_context, client, fail_on_llm_error=require_api)
    pred["retrieved_memory_count"] = len(retrieved)
    pred["memory_card_count"] = len(store.cards)
    pred["memory_path"] = str(store.path)
    pred["memory_ops"] = []
    pred["guard_passed"] = True
    pred["locomo_top_k"] = top_k
    pred["locomo_coarse_k"] = coarse_k
    pred["retrieval_query"] = locomo_expanded_query(sample)
    pred["pipeline_note"] = (
        "LoCoMo turns are written to JSONL memory cards with entity/time metadata; "
        "QA prompt receives only two-stage retrieved memory cards."
    )
    return pred


def build_locomo_amem_system(sample: dict[str, Any]) -> SourceAlignedAMEMSystem:
    system = SourceAlignedAMEMSystem()
    turns = sample.get("turns") or []
    if turns:
        iterable = [
            (
                str(turn.get("time") or idx),
                normalize_answer(turn.get("text")),
                str(turn.get("conversation_id") or sample.get("conversation_id") or sample["sample_id"]),
                [str(tag) for tag in turn.get("tags", ["dialogue", "locomo"])],
            )
            for idx, turn in enumerate(turns)
        ]
    else:
        iterable = [
            (str(idx), line.strip(), str(sample.get("conversation_id") or sample["sample_id"]), ["dialogue", "native_benchmark"])
            for idx, line in enumerate(str(sample.get("context") or "").splitlines())
        ]
    for time, text, conversation_id, tags in iterable:
        if not text:
            continue
        system.add_note(
            content=text,
            time=time,
            context=f"{sample['dataset']} LoCoMo turn memory for {conversation_id}",
            tags=tags,
            category="dialogue",
        )
    return system


@dataclass
class LocomoAMEMRuntime:
    system: SourceAlignedAMEMSystem
    lock: Lock


def amem_retrieved_context(
    sample: dict[str, Any],
    *,
    top_k: int = 8,
    runtime: LocomoAMEMRuntime | None = None,
) -> tuple[str, int]:
    system = runtime.system if runtime is not None else build_locomo_amem_system(sample)
    if runtime is not None:
        with runtime.lock:
            retrieved = system.find_related_memories_raw(str(sample.get("question") or ""), k=top_k)
    else:
        retrieved = system.find_related_memories_raw(str(sample.get("question") or ""), k=top_k)
    return f"[A-MEM RETRIEVED MEMORY]\n{retrieved}\n\n[QUESTION]\n{sample['question']}", min(top_k, len(system.memories))


def baseline_method_paths(method: str) -> tuple[Path, Path]:
    repo = Path(os.getenv(ENV_BY_METHOD[method], str(DEFAULT_BASELINE_REPOS[method])))
    python = Path(os.getenv(PY_ENV_BY_METHOD[method], str(DEFAULT_BASELINE_PYTHONS[method])))
    return repo, python


def ensure_official_wrapper_ready(method: str) -> tuple[Path, Path]:
    repo, python = baseline_method_paths(method)
    missing = []
    if not repo.exists():
        missing.append(f"repo={repo}")
    if not python.exists():
        missing.append(f"python={python}")
    if missing:
        raise RuntimeError(f"{method} baseline wrapper is not configured: {', '.join(missing)}")
    return repo, python


def run_locomo_official_wrapper(
    sample: dict[str, Any],
    method: str,
    client: DeepSeekClient | None,
    *,
    run_dir: str | Path,
    top_k: int = 8,
    coarse_k: int = 32,
    require_api: bool,
    memory_cache_dir: str | Path | None = None,
    dataset_hash: str = "",
) -> dict[str, Any]:
    if sample.get("dataset") != "locomo":
        raise ValueError(f"{method} official wrapper currently supports only LoCoMo samples.")
    repo, python = ensure_official_wrapper_ready(method)
    memory_path = locomo_memory_path(Path(run_dir) / "official_wrappers" / method, str(sample["sample_id"]))
    store = build_cached_locomo_memory_store(
        sample,
        memory_cache_dir,
        dataset_hash=dataset_hash,
        top_k=top_k,
        coarse_k=coarse_k,
        fallback_path=memory_path,
    )
    retrieved = retrieve_locomo_memories(store, sample, top_k=top_k, coarse_k=coarse_k)
    memory_context = "\n".join(format_locomo_memory_line(memory) for memory in retrieved)
    prompt_context = (
        f"[OFFICIAL_BASELINE]\n{method}\n"
        f"[REPRODUCTION_LEVEL]\n{BASELINE_REPRODUCTION_LEVELS[method]}\n"
        f"[ADAPTER_NOTE]\n{BASELINE_ADAPTER_NOTES[method]}\n"
        f"[REPO]\n{repo}\n"
        f"[PYTHON]\n{python}\n\n"
        "[RETRIEVED_MEMORY_CARDS]\n"
        f"{memory_context}\n\n"
        "[QUESTION]\n"
        f"{sample['question']}\n"
        "Answer using only the retrieved memory cards."
    )
    assert_no_full_context_shortcut(
        BASELINE_METHOD_LABELS[method],
        prompt_context,
        sample,
        retrieved_memory_count=len(retrieved),
    )
    pred = answer_with_context(sample, BASELINE_METHOD_LABELS[method], prompt_context, client, fail_on_llm_error=require_api)
    pred["retrieved_memory_count"] = len(retrieved)
    pred["memory_card_count"] = len(store.cards)
    pred["memory_path"] = str(store.path)
    pred["guard_passed"] = True
    pred["baseline_reproduction_level"] = BASELINE_REPRODUCTION_LEVELS[method]
    pred["official_repo"] = METHOD_OFFICIAL_REPOS[method]
    pred["repo_path"] = str(repo)
    pred["python_path"] = str(python)
    pred["adapter_note"] = BASELINE_ADAPTER_NOTES[method]
    pred["locomo_top_k"] = top_k
    pred["locomo_coarse_k"] = coarse_k
    return pred


def run_local_method(
    sample: dict[str, Any],
    method: str,
    client: DeepSeekClient | None,
    *,
    require_api: bool,
    run_dir: str | Path | None = None,
    amem_runtimes: dict[str, LocomoAMEMRuntime] | None = None,
    locomo_top_k: int = 8,
    locomo_coarse_k: int = 32,
    method_label: str | None = None,
    memory_cache_dir: str | Path | None = None,
    dataset_hash: str = "",
) -> dict[str, Any]:
    if method == "direct":
        pred = answer_with_context(
            sample,
            "direct_locomo_qa",
            "",
            client,
            fail_on_llm_error=require_api,
        )
        pred["retrieved_memory_count"] = 0
        pred["baseline_reproduction_level"] = "local_direct_no_memory"
        pred["adapter_note"] = "Direct LoCoMo QA baseline: same QA model and samples, no retrieved memory context."
        return pred
    if method in {"ours", "medimem"}:
        if sample.get("dataset") != "locomo" or run_dir is None:
            raise RuntimeError(
                "The previous ours full-context native shortcut is disabled. "
                f"{FULL_CONTEXT_SHORTCUT_ADVICE}"
            )
        pred = run_locomo_ours_memory_pipeline(
            sample,
            client,
            run_dir=run_dir,
            top_k=locomo_top_k,
            coarse_k=locomo_coarse_k,
            require_api=require_api,
            memory_cache_dir=memory_cache_dir,
            dataset_hash=dataset_hash,
        )
        if method_label:
            pred["method"] = method_label
        return pred
    if method == "amem":
        runtime = None
        if amem_runtimes is not None:
            conversation_id = str(sample.get("conversation_id") or sample["sample_id"])
            runtime = amem_runtimes.get(conversation_id)
        context, retrieved = amem_retrieved_context(sample, runtime=runtime)
        pred = answer_with_context(sample, BASELINE_METHOD_LABELS["amem"], context, client, fail_on_llm_error=require_api)
        pred["retrieved_memory_count"] = retrieved
        pred["baseline_reproduction_level"] = "official_native_locomo_wrapper"
        pred["official_repo"] = METHOD_OFFICIAL_REPOS["amem"]
        pred["repo_path"] = str(Path(os.getenv("AMEM_REPO", "/home/syh/A-mem/A-mem-main")))
        pred["adapter_note"] = (
            "A-MEM is wrapped through the project SourceAlignedAMEMSystem memory flow and evaluated with "
            "the same LoCoMo samples, QA model, and metric script; this is not original-paper full reproduction."
        )
        return pred
    raise ValueError(f"Unsupported local benchmark method: {method}")


def official_method_status(method: str, dataset: str) -> dict[str, Any]:
    spec = NATIVE_BENCHMARKS[dataset]
    env_name = ENV_BY_METHOD.get(method) or spec.required_env
    py_env_name = PY_ENV_BY_METHOD.get(method)
    default_repo = DEFAULT_BASELINE_REPOS.get(method)
    default_python = DEFAULT_BASELINE_PYTHONS.get(method)
    repo_path = os.getenv(env_name or "") or (str(default_repo) if default_repo and default_repo.exists() else "")
    python_path = os.getenv(py_env_name or "") or (str(default_python) if default_python and default_python.exists() else "")
    if not env_name:
        return {"method": method, "dataset": dataset, "status": "local_adapter_available"}
    if repo_path and Path(repo_path).exists() and (method not in OFFICIAL_WRAPPER_METHODS or (python_path and Path(python_path).exists())):
        return {
            "method": method,
            "dataset": dataset,
            "status": "official_wrapper_configured" if method in OFFICIAL_WRAPPER_METHODS else "official_repo_detected_manual_runner_required",
            "env": env_name,
            "repo_path": repo_path,
            "python_env": py_env_name,
            "python_path": python_path,
            "baseline_reproduction_level": BASELINE_REPRODUCTION_LEVELS.get(method),
            "adapter_note": BASELINE_ADAPTER_NOTES.get(method),
        }
    missing = []
    if not repo_path or not Path(repo_path).exists():
        missing.append(env_name)
    if method in OFFICIAL_WRAPPER_METHODS and (not python_path or not Path(python_path).exists()):
        missing.append(py_env_name or f"{method.upper()}_PY")
    return {
        "method": method,
        "dataset": dataset,
        "status": "blocked_official_wrapper_not_configured" if method in OFFICIAL_WRAPPER_METHODS else "blocked_official_repo_not_configured",
        "env": env_name,
        "python_env": py_env_name,
        "missing": missing,
        "official_repo": METHOD_OFFICIAL_REPOS.get(method, spec.official_repo),
    }


def prediction_metric_values(sample: dict[str, Any], pred: dict[str, Any]) -> dict[str, float | None]:
    answer = normalize_answer(pred.get("answer"))
    gold = normalize_answer(sample.get("answer"))
    sbert, _ = optional_sbert_similarity(answer, gold)
    return {
        "exact_match": 1.0 if answer.lower() == gold.lower() and answer else 0.0,
        "soft_match": 1.0 if soft_match(answer, gold) else 0.0,
        "qa_f1": token_f1(answer, gold),
        "bleu1": simple_bleu1(answer, gold),
        "bleu2": simple_bleu_n(answer, gold, 2),
        "bleu3": simple_bleu_n(answer, gold, 3),
        "bleu4": simple_bleu_n(answer, gold, 4),
        "rouge1_f": rouge_n_f(answer, gold, 1),
        "rouge2_f": rouge_n_f(answer, gold, 2),
        "rougeL_f": rouge_l_f(answer, gold),
        "meteor": lightweight_meteor(answer, gold),
        "sbert_similarity": sbert,
        "avg_tokens": float((pred.get("usage") or {}).get("total_tokens", 0) or 0),
        "retrieved_memory_count": float(pred.get("retrieved_memory_count", 0) or 0),
        "guard_pass_rate": 1.0 if bool(pred.get("guard_passed", True)) else 0.0,
        "fallback_rate": 1.0 if pred.get("fallback_reason") else 0.0,
    }


def collect_metric_rows(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        sample = sample_by_id[pred["sample_id"]]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "method": str(pred.get("method")),
                "split": str(pred.get("split")),
                "category": sample.get("category"),
                "category_name": sample.get("category_name") or locomo_category_name(sample.get("category")),
                **prediction_metric_values(sample, pred),
            }
        )
    return rows


def average_metric(vals: list[dict[str, Any]], metric: str) -> float | None:
    numbers = [float(row[metric]) for row in vals if row.get(metric) is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def aggregate_metric_rows(
    rows: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...],
    include_category: bool = False,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        category = str(row.get("category_name") or "overall") if include_category else "overall"
        key = (str(row.get("method")), str(row.get("split")), category)
        buckets.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (method, split, category), vals in sorted(buckets.items()):
        item: dict[str, Any] = {"method": method, "split": split}
        if include_category:
            item["category"] = category
        item["n"] = len(vals)
        for metric in metrics:
            item[metric] = average_metric(vals, metric)
        out.append(item)
    return out


def ensure_locomo_category_rows(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    methods = sorted({str(row.get("method")) for row in rows})
    splits = sorted({str(row.get("split")) for row in rows})
    existing = {(str(row.get("method")), str(row.get("split")), str(row.get("category"))) for row in rows}
    completed = list(rows)
    for method in methods:
        for split in splits:
            for category in LOCOMO_CATEGORY_NAMES.values():
                key = (method, split, category)
                if key in existing:
                    continue
                item: dict[str, Any] = {"method": method, "split": split, "category": category, "n": 0}
                for metric in metrics:
                    item[metric] = None
                completed.append(item)
    return sorted(completed, key=lambda row: (str(row.get("method")), str(row.get("split")), str(row.get("category"))))


def evaluate_benchmark_predictions(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = collect_metric_rows(samples, predictions)
    legacy = aggregate_metric_rows(
        rows,
        metrics=("exact_match", "soft_match", "qa_f1", "bleu1", "avg_tokens", "retrieved_memory_count"),
        include_category=False,
    )
    for row in legacy:
        row.pop("category", None)
    return legacy


def evaluate_locomo_benchmark_predictions(
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = collect_metric_rows(samples, predictions)
    overall = aggregate_metric_rows(rows, metrics=LOCOMO_MAIN_METRICS, include_category=False)
    by_category = ensure_locomo_category_rows(
        aggregate_metric_rows(rows, metrics=LOCOMO_MAIN_METRICS, include_category=True),
        LOCOMO_MAIN_METRICS,
    )
    auxiliary = aggregate_metric_rows(rows, metrics=LOCOMO_AUXILIARY_METRICS, include_category=False)
    official_style = ensure_locomo_category_rows(
        aggregate_metric_rows(rows, metrics=LOCOMO_OFFICIAL_STYLE_METRICS, include_category=True),
        LOCOMO_OFFICIAL_STYLE_METRICS,
    )
    skipped: dict[str, str] = {}
    _, sbert_reason = optional_sbert_similarity("", "")
    if sbert_reason:
        skipped["sbert_similarity"] = sbert_reason
    skipped["bert_f1"] = "not computed in this lightweight benchmark runner; use optional bert-score dependency for a separate semantic-eval pass"
    return {
        "per_sample": rows,
        "overall": overall,
        "by_category": by_category,
        "auxiliary": auxiliary,
        "official_style": official_style,
        "skipped_metrics": skipped,
    }


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_horizontal_run(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = [sample["sample_id"] for sample in samples]
    expected = set(sample_ids)
    by_method: dict[str, set[str]] = {}
    for pred in predictions:
        by_method.setdefault(str(pred.get("method")), set()).add(str(pred.get("sample_id")))
    failures = []
    for method, ids in by_method.items():
        if ids != expected:
            failures.append({"method": method, "missing": sorted(expected - ids)[:10], "extra": sorted(ids - expected)[:10]})
    split_set = sorted({str(sample.get("split")) for sample in samples})
    return {
        "passed": not failures,
        "sample_count": len(samples),
        "split_set": split_set,
        "method_count": len(by_method),
        "failures": failures,
    }


def run_native_benchmark(
    *,
    dataset: str,
    methods: list[str],
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    sample_n: int | None = None,
    random_seed: int | None = None,
    max_workers: int = 1,
    output_root: str | Path = "runs",
    require_api: bool = False,
    locomo_top_k: int = 8,
    locomo_coarse_k: int = 32,
    top_k_sweep: list[int] | None = None,
) -> Path:
    loaded_samples = load_native_samples(dataset, dataset_path, limit=limit)
    samples = sample_benchmark_rows(loaded_samples, sample_n=sample_n, random_seed=random_seed)
    prefix = f"{dataset}_memory_random{sample_n}" if dataset == "locomo" and sample_n else "benchmark"
    run_dir = make_benchmark_run_dir(output_root, prefix=prefix)
    client, blocker = build_client(require_api=require_api)
    predictions: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    amem_runtimes: dict[str, LocomoAMEMRuntime] = {}
    if dataset == "locomo" and "ours" in methods:
        seen_conversations: set[str] = set()
        for sample in samples:
            conversation_id = str(sample.get("conversation_id") or sample["sample_id"])
            if conversation_id in seen_conversations:
                continue
            build_locomo_memory_store(sample, locomo_memory_path(run_dir, conversation_id))
            seen_conversations.add(conversation_id)
    if dataset == "locomo" and "amem" in methods:
        seen_conversations: set[str] = set()
        for sample in samples:
            conversation_id = str(sample.get("conversation_id") or sample["sample_id"])
            if conversation_id in seen_conversations:
                continue
            amem_runtimes[conversation_id] = LocomoAMEMRuntime(
                system=build_locomo_amem_system(sample),
                lock=Lock(),
            )
            seen_conversations.add(conversation_id)
    sweep_values = list(top_k_sweep or [])
    for method in methods:
        if method in LOCAL_METHODS:
            with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
                futures = {}
                if dataset == "locomo" and method in {"ours", "medimem"} and sweep_values:
                    for top_k in sweep_values:
                        label = f"medimem_locomo_memory_pipeline_topk{top_k}"
                        for sample in samples:
                            futures[
                                pool.submit(
                                    run_local_method,
                                    sample,
                                    method,
                                    client,
                                    require_api=require_api,
                                    run_dir=run_dir,
                                    amem_runtimes=amem_runtimes,
                                    locomo_top_k=top_k,
                                    locomo_coarse_k=max(int(locomo_coarse_k), top_k),
                                    method_label=label,
                                )
                            ] = f"{sample['sample_id']}:{label}"
                else:
                    futures = {
                        pool.submit(
                            run_local_method,
                            sample,
                            method,
                            client,
                            require_api=require_api,
                            run_dir=run_dir,
                            amem_runtimes=amem_runtimes,
                            locomo_top_k=locomo_top_k,
                            locomo_coarse_k=locomo_coarse_k,
                        ): sample["sample_id"]
                        for sample in samples
                    }
                for future in as_completed(futures):
                    predictions.append(future.result())
        elif dataset == "locomo" and method in OFFICIAL_WRAPPER_METHODS:
            status = official_method_status(method, dataset)
            if status.get("status") != "official_wrapper_configured":
                blocked.append(status)
                continue
            with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
                futures = {
                    pool.submit(
                        run_locomo_official_wrapper,
                        sample,
                        method,
                        client,
                        run_dir=run_dir,
                        top_k=locomo_top_k,
                        coarse_k=locomo_coarse_k,
                        require_api=require_api,
                    ): sample["sample_id"]
                    for sample in samples
                }
                for future in as_completed(futures):
                    predictions.append(future.result())
        else:
            blocked.append(official_method_status(method, dataset))
    write_jsonl(run_dir / "samples.jsonl", samples)
    write_jsonl(run_dir / "predictions" / f"{dataset}.jsonl", predictions)
    write_jsonl(run_dir / "blocked_methods.jsonl", blocked)
    metrics = evaluate_benchmark_predictions(samples, predictions)
    write_csv(run_dir / f"{dataset}_metrics.csv", metrics)
    locomo_metrics: dict[str, Any] | None = None
    if dataset == "locomo":
        locomo_metrics = evaluate_locomo_benchmark_predictions(samples, predictions)
        write_csv(run_dir / "locomo_metrics_overall.csv", locomo_metrics["overall"])
        write_csv(run_dir / "locomo_metrics_by_category.csv", locomo_metrics["by_category"])
        write_csv(run_dir / "locomo_metrics_auxiliary.csv", locomo_metrics["auxiliary"])
        write_csv(run_dir / "locomo_metrics_official_style.csv", locomo_metrics["official_style"])
        write_jsonl(run_dir / "locomo_metrics_per_sample.jsonl", locomo_metrics["per_sample"])
    validation = validate_horizontal_run(samples, predictions) if predictions else {"passed": False, "failures": ["no predictions"]}
    baseline_reproduction_level = {
        "direct_locomo_qa": {
            "level": "local_direct_no_memory",
            "official_repo": None,
            "core_flow": "Question-only QA with the same OpenAI-compatible model; no memory retrieval.",
        },
        "medimem_locomo_memory_pipeline": {
            "level": "project_pipeline",
            "official_repo": None,
            "core_flow": "Entity/time-aware JSONL memory cards, expanded-query coarse retrieval, rerank to top-k, QA over retrieved memories only.",
        },
        BASELINE_METHOD_LABELS["amem"]: {
            "level": "official_native_locomo_wrapper",
            "official_repo": METHOD_OFFICIAL_REPOS["amem"],
            "core_flow": "SourceAlignedAMEMSystem.add_note/process_memory/find_related_memories_raw.",
            "not_fully_reproduced": [
                "Not executed through the upstream official repository entrypoint.",
                "Uses this project's OpenAI-compatible QA wrapper and local source-aligned A-MEM wrapper.",
            ],
        },
    }
    for method in OFFICIAL_WRAPPER_METHODS:
        baseline_reproduction_level[BASELINE_METHOD_LABELS[method]] = {
            "level": BASELINE_REPRODUCTION_LEVELS[method],
            "official_repo": METHOD_OFFICIAL_REPOS[method],
            "core_flow": "Official baseline repository mirrored into baseline_envs; project-unified LoCoMo wrapper, sample set, QA model, and metric script.",
            "adapter_note": BASELINE_ADAPTER_NOTES[method],
        }
    for item in blocked:
        baseline_reproduction_level[str(item.get("method"))] = {
            "level": item.get("status") or "blocked_official_repo_not_configured",
            "official_repo": item.get("official_repo"),
            "env": item.get("env"),
            "python_env": item.get("python_env"),
            "missing": item.get("missing"),
        }
    manifest = {
        "dataset": dataset,
        "dataset_path": str(dataset_path or NATIVE_BENCHMARKS[dataset].default_path),
        "methods": methods,
        "sample_n": sample_n,
        "random_seed": random_seed,
        "max_workers": max_workers,
        "locomo_top_k": locomo_top_k,
        "locomo_coarse_k": locomo_coarse_k,
        "top_k_sweep": sweep_values,
        "total_available_qa_samples": len(loaded_samples),
        "local_methods": sorted({pred["method"] for pred in predictions}),
        "blocked_methods": blocked,
        "blocker": blocker,
        "validation": validation,
        "official_repo": NATIVE_BENCHMARKS[dataset].official_repo,
        "model": get_deepseek_config().model,
        "guard": {
            "full_context_shortcut_forbidden": True,
            "passed": all(bool(pred.get("guard_passed", True)) for pred in predictions),
        },
        "metric_policy": {
            "benchmark_compatible_primary": list(LOCOMO_MAIN_METRICS) if dataset == "locomo" else [],
            "locomo_category_breakdown": dataset == "locomo",
            "auxiliary_audit_only": list(LOCOMO_AUXILIARY_METRICS) if dataset == "locomo" else [],
            "official_style_additional": list(LOCOMO_OFFICIAL_STYLE_METRICS) if dataset == "locomo" else [],
            "skipped_metrics": (locomo_metrics or {}).get("skipped_metrics", {}),
            "scale": "0-1",
            "medical_task_metrics": [
                "primary_diagnosis_top1_accuracy",
                "diagnosis_list_f1",
                "cdr_f1",
                "memory_pollution_control_score",
                "stale_memory_action_accuracy",
                "revision_accuracy",
                "fact_preservation_soft",
                "over_deletion_rate",
                "avg_tokens",
            ],
        },
        "baseline_reproduction_level": baseline_reproduction_level,
        "pipeline_policies": {
            "direct": "Question-only QA baseline with no retrieved long-term memory.",
            "medimem": "LoCoMo turns -> entity/time-aware JSONL MemoryStore cards -> expanded-query coarse retrieval -> rerank to top-k -> QA over retrieved memory cards only.",
            "ours": "Backward-compatible alias for medimem.",
            "amem": "LoCoMo turns -> SourceAlignedAMEMSystem.add_note/process_memory -> find_related_memories_raw top-k -> QA.",
        },
    }
    write_text(run_dir / "benchmark_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    render_benchmark_report(run_dir, dataset, metrics, manifest)
    return run_dir


def parse_method_queue(raw: str) -> list[str]:
    return parse_methods(raw)


def append_stage_status(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    payload = {"time": datetime.now().isoformat(timespec="seconds"), **row}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_locomo_memory_cache(
    samples: list[dict[str, Any]],
    *,
    cache_dir: str | Path,
    dataset_hash: str,
    top_k: int,
    coarse_k: int,
    max_workers: int,
    status_path: str | Path,
) -> None:
    started = time.time()
    append_stage_status(status_path, {"stage": "cache_prepare", "status": "started", "sample_count": len(samples)})
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [
            pool.submit(
                build_cached_locomo_memory_store,
                sample,
                cache_dir,
                dataset_hash=dataset_hash,
                top_k=top_k,
                coarse_k=coarse_k,
                fallback_path=locomo_memory_path(Path(cache_dir) / "fallback", str(sample["sample_id"])),
            )
            for sample in samples
        ]
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(samples):
                append_stage_status(
                    status_path,
                    {"stage": "cache_prepare", "status": "progress", "completed": completed, "sample_count": len(samples)},
                )
    append_stage_status(
        status_path,
        {"stage": "cache_prepare", "status": "finished", "elapsed_s": round(time.time() - started, 3)},
    )


def build_locomo_amem_runtimes(samples: list[dict[str, Any]]) -> dict[str, LocomoAMEMRuntime]:
    runtimes: dict[str, LocomoAMEMRuntime] = {}
    seen_conversations: set[str] = set()
    for sample in samples:
        conversation_id = str(sample.get("conversation_id") or sample["sample_id"])
        if conversation_id in seen_conversations:
            continue
        runtimes[conversation_id] = LocomoAMEMRuntime(system=build_locomo_amem_system(sample), lock=Lock())
        seen_conversations.add(conversation_id)
    return runtimes


def run_locomo_method_batch(
    *,
    method: str,
    samples: list[dict[str, Any]],
    client: DeepSeekClient | None,
    run_dir: Path,
    require_api: bool,
    max_workers: int,
    locomo_top_k: int,
    locomo_coarse_k: int,
    memory_cache_dir: Path,
    dataset_hash: str,
    amem_runtimes: dict[str, LocomoAMEMRuntime],
    status_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.time()
    append_stage_status(status_path, {"stage": "method", "method": method, "status": "started", "sample_count": len(samples)})
    predictions: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    if method in LOCAL_METHODS:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = {
                pool.submit(
                    run_local_method,
                    sample,
                    method,
                    client,
                    require_api=require_api,
                    run_dir=run_dir,
                    amem_runtimes=amem_runtimes,
                    locomo_top_k=locomo_top_k,
                    locomo_coarse_k=locomo_coarse_k,
                    memory_cache_dir=memory_cache_dir,
                    dataset_hash=dataset_hash,
                ): sample["sample_id"]
                for sample in samples
            }
            completed = 0
            for future in as_completed(futures):
                predictions.append(future.result())
                completed += 1
                if completed % 100 == 0 or completed == len(samples):
                    append_stage_status(
                        status_path,
                        {"stage": "method", "method": method, "status": "progress", "completed": completed, "sample_count": len(samples)},
                    )
    elif method in OFFICIAL_WRAPPER_METHODS:
        status = official_method_status(method, "locomo")
        if status.get("status") != "official_wrapper_configured":
            blocked.append(status)
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
                futures = {
                    pool.submit(
                        run_locomo_official_wrapper,
                        sample,
                        method,
                        client,
                        run_dir=run_dir,
                        top_k=locomo_top_k,
                        coarse_k=locomo_coarse_k,
                        require_api=require_api,
                        memory_cache_dir=memory_cache_dir,
                        dataset_hash=dataset_hash,
                    ): sample["sample_id"]
                    for sample in samples
                }
                completed = 0
                for future in as_completed(futures):
                    predictions.append(future.result())
                    completed += 1
                    if completed % 100 == 0 or completed == len(samples):
                        append_stage_status(
                            status_path,
                            {"stage": "method", "method": method, "status": "progress", "completed": completed, "sample_count": len(samples)},
                        )
    else:
        blocked.append(official_method_status(method, "locomo"))
    write_jsonl(run_dir / "method_predictions" / f"{method}.jsonl", predictions)
    append_stage_status(
        status_path,
        {
            "stage": "method",
            "method": method,
            "status": "finished",
            "prediction_count": len(predictions),
            "blocked_count": len(blocked),
            "elapsed_s": round(time.time() - started, 3),
        },
    )
    return predictions, blocked


def run_locomo_parallel_benchmark(
    *,
    methods_a: list[str],
    methods_b: list[str],
    base_url_a: str | None,
    base_url_b: str | None,
    dataset_path: str | Path | None = None,
    sample_n: int | None = None,
    random_seed: int | None = None,
    max_workers_per_queue: int = 48,
    output_root: str | Path = "runs",
    require_api: bool = False,
    locomo_top_k: int = 8,
    locomo_coarse_k: int = 32,
) -> Path:
    loaded_samples = load_native_samples("locomo", dataset_path, limit=None)
    samples = sample_benchmark_rows(loaded_samples, sample_n=sample_n, random_seed=random_seed)
    prefix = f"locomo_parallel_random{sample_n}" if sample_n else "locomo_parallel"
    run_dir = make_benchmark_run_dir(output_root, prefix=prefix)
    status_path = run_dir / "stage_status.jsonl"
    ensure_dir(run_dir / "method_predictions")
    dataset_hash = locomo_dataset_hash(samples)
    memory_cache_dir = run_dir / "shared_cache"
    started = time.time()
    append_stage_status(
        status_path,
        {
            "stage": "run",
            "status": "started",
            "sample_count": len(samples),
            "methods_a": methods_a,
            "methods_b": methods_b,
            "max_workers_per_queue": max_workers_per_queue,
        },
    )
    client_a, blocker_a = build_client_for_base_url(base_url_a, require_api=require_api)
    client_b, blocker_b = build_client_for_base_url(base_url_b or base_url_a, require_api=require_api)
    prepare_locomo_memory_cache(
        samples,
        cache_dir=memory_cache_dir,
        dataset_hash=dataset_hash,
        top_k=locomo_top_k,
        coarse_k=locomo_coarse_k,
        max_workers=max_workers_per_queue * 2,
        status_path=status_path,
    )
    all_methods = list(dict.fromkeys(methods_a + methods_b))
    amem_runtimes = build_locomo_amem_runtimes(samples) if "amem" in all_methods else {}

    def run_queue(label: str, methods: list[str], client: DeepSeekClient | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        queue_predictions: list[dict[str, Any]] = []
        queue_blocked: list[dict[str, Any]] = []
        append_stage_status(status_path, {"stage": "queue", "queue": label, "status": "started", "methods": methods})
        for method in methods:
            preds, blocked = run_locomo_method_batch(
                method=method,
                samples=samples,
                client=client,
                run_dir=run_dir,
                require_api=require_api,
                max_workers=max_workers_per_queue,
                locomo_top_k=locomo_top_k,
                locomo_coarse_k=locomo_coarse_k,
                memory_cache_dir=memory_cache_dir,
                dataset_hash=dataset_hash,
                amem_runtimes=amem_runtimes,
                status_path=status_path,
            )
            queue_predictions.extend(preds)
            queue_blocked.extend(blocked)
        append_stage_status(status_path, {"stage": "queue", "queue": label, "status": "finished"})
        return queue_predictions, queue_blocked

    predictions: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_queue, "A", methods_a, client_a),
            pool.submit(run_queue, "B", methods_b, client_b),
        ]
        for future in as_completed(futures):
            preds, items = future.result()
            predictions.extend(preds)
            blocked.extend(items)
    write_jsonl(run_dir / "samples.jsonl", samples)
    write_jsonl(run_dir / "predictions" / "locomo.jsonl", predictions)
    write_jsonl(run_dir / "blocked_methods.jsonl", blocked)
    metrics = evaluate_benchmark_predictions(samples, predictions)
    write_csv(run_dir / "locomo_metrics.csv", metrics)
    locomo_metrics = evaluate_locomo_benchmark_predictions(samples, predictions)
    write_csv(run_dir / "locomo_metrics_overall.csv", locomo_metrics["overall"])
    write_csv(run_dir / "locomo_metrics_by_category.csv", locomo_metrics["by_category"])
    write_csv(run_dir / "locomo_metrics_auxiliary.csv", locomo_metrics["auxiliary"])
    write_csv(run_dir / "locomo_metrics_official_style.csv", locomo_metrics["official_style"])
    write_jsonl(run_dir / "locomo_metrics_per_sample.jsonl", locomo_metrics["per_sample"])
    validation = validate_horizontal_run(samples, predictions) if predictions else {"passed": False, "failures": ["no predictions"]}
    manifest = {
        "dataset": "locomo",
        "dataset_path": str(dataset_path or NATIVE_BENCHMARKS["locomo"].default_path),
        "methods": all_methods,
        "methods_a": methods_a,
        "methods_b": methods_b,
        "sample_n": sample_n,
        "random_seed": random_seed,
        "max_workers_per_queue": max_workers_per_queue,
        "base_url_a": base_url_a,
        "base_url_b": base_url_b,
        "client_blockers": {"A": blocker_a, "B": blocker_b},
        "locomo_top_k": locomo_top_k,
        "locomo_coarse_k": locomo_coarse_k,
        "dataset_hash": dataset_hash,
        "memory_cache_dir": str(memory_cache_dir),
        "total_available_qa_samples": len(loaded_samples),
        "local_methods": sorted({str(pred.get("method")) for pred in predictions}),
        "blocked_methods": blocked,
        "validation": validation,
        "official_repo": NATIVE_BENCHMARKS["locomo"].official_repo,
        "model": get_deepseek_config().model,
        "elapsed_s": round(time.time() - started, 3),
        "metric_policy": {
            "benchmark_compatible_primary": list(LOCOMO_MAIN_METRICS),
            "locomo_category_breakdown": True,
            "auxiliary_audit_only": list(LOCOMO_AUXILIARY_METRICS),
            "official_style_additional": list(LOCOMO_OFFICIAL_STYLE_METRICS),
            "skipped_metrics": locomo_metrics.get("skipped_metrics", {}),
            "scale": "0-1",
        },
    }
    write_text(run_dir / "benchmark_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    render_benchmark_report(run_dir, "locomo", metrics, manifest)
    append_stage_status(status_path, {"stage": "run", "status": "finished", "elapsed_s": round(time.time() - started, 3)})
    return run_dir


def render_benchmark_report(run_dir: Path, dataset: str, metrics: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    if dataset == "locomo":
        render_locomo_benchmark_report(run_dir, manifest)
        return
    lines = [
        f"# Native Benchmark Report: {dataset}",
        "",
        "- 横向比较只使用同一 dataset path、同一 split、同一样本 ID 集合。",
        f"- Official repo: {manifest.get('official_repo')}",
        f"- Validation passed: {manifest.get('validation', {}).get('passed')}",
        "",
        "| Method | Split | N | Exact | Soft | QA F1 | BLEU-1 | Avg Tokens | Retrieved |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            "| {method} | {split} | {n} | {exact:.3f} | {soft:.3f} | {f1:.3f} | {bleu:.3f} | {tokens:.1f} | {retrieved:.1f} |".format(
                method=row["method"],
                split=row["split"],
                n=row["n"],
                exact=float(row["exact_match"]),
                soft=float(row["soft_match"]),
                f1=float(row["qa_f1"]),
                bleu=float(row["bleu1"]),
                tokens=float(row["avg_tokens"]),
                retrieved=float(row["retrieved_memory_count"]),
            )
        )
    blocked = manifest.get("blocked_methods") or []
    if blocked:
        lines.extend(["", "## Official Code Blockers"])
        for item in blocked:
            lines.append(f"- {item.get('method')}: {item.get('status')} ({item.get('env', 'no-env')})")
    write_text(run_dir / "benchmark_report_zh.md", "\n".join(lines) + "\n")


def method_label(method: str) -> str:
    if method == "direct_locomo_qa":
        return "Direct"
    if method in {"source_aligned_amem_adapter", BASELINE_METHOD_LABELS["amem"]}:
        return "A-MEM wrapper"
    if method == "medimem_locomo_memory_pipeline":
        return "medimem"
    if method == "ours_locomo_memory_pipeline":
        return "ours"
    if method.startswith("medimem_locomo_memory_pipeline_topk"):
        return method.replace("medimem_locomo_memory_pipeline_", "medimem ")
    if method.startswith("ours_locomo_memory_pipeline_topk"):
        return method.replace("ours_locomo_memory_pipeline_", "ours ")
    for raw, label in BASELINE_METHOD_LABELS.items():
        if method == label:
            return raw
    return method


def float_cell(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in {None, "", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def comparison_sentence(rows: list[dict[str, str]], *, category: str, metric: str, label: str) -> str | None:
    relevant = [row for row in rows if str(row.get("category", "overall")) == category]
    ours_rows = [
        row
        for row in relevant
        if str(row.get("method", "")).startswith("medimem_locomo_memory_pipeline")
        or str(row.get("method", "")).startswith("ours_locomo_memory_pipeline")
    ]
    amem = next((row for row in relevant if row.get("method") in {"source_aligned_amem_adapter", BASELINE_METHOD_LABELS["amem"]}), None)
    if not ours_rows or not amem:
        return None
    if metric == "avg_tokens":
        ours = min(ours_rows, key=lambda row: float_cell(row, metric) if float_cell(row, metric) is not None else float("inf"))
    else:
        ours = max(ours_rows, key=lambda row: float_cell(row, metric) or 0.0)
    ours_val = float_cell(ours, metric)
    amem_val = float_cell(amem, metric)
    if ours_val is None or amem_val is None:
        return None
    relation = "高于" if ours_val > amem_val else "低于" if ours_val < amem_val else "持平于"
    return f"- {method_label(str(ours.get('method')))} 在 {label} 的 {metric} 上{relation} A-MEM wrapper（ours={ours_val:.4f}, A-MEM={amem_val:.4f}）。"


def render_locomo_benchmark_report(run_dir: Path, manifest: dict[str, Any]) -> None:
    overall = read_csv_rows(run_dir / "locomo_metrics_overall.csv")
    by_category = read_csv_rows(run_dir / "locomo_metrics_by_category.csv")
    auxiliary = read_csv_rows(run_dir / "locomo_metrics_auxiliary.csv")
    official_style = read_csv_rows(run_dir / "locomo_metrics_official_style.csv")
    fallback_count = 0
    pred_path = run_dir / "predictions" / "locomo.jsonl"
    if pred_path.exists():
        fallback_count = sum(1 for line in pred_path.read_text(encoding="utf-8").splitlines() if "fallback_reason" in line)
    overall_with_category = [{**row, "category": "overall"} for row in overall]
    lines = [
        "# LoCoMo 正式评测报告",
        "",
        "## 评测口径",
        "- LoCoMo 仅用于评估长期记忆 QA benchmark 表现，不直接解释为医疗污染记忆清洗能力。",
        "- 主指标采用 A-MEM 论文兼容口径：qa_f1、bleu1，并按 QA 类别报告。",
        "- exact_match、soft_match、retrieved_memory_count、guard_pass_rate、fallback_rate 仅作为辅助审计指标。",
        "- A-MEM 结果来自统一 LoCoMo wrapper，不表述为官方 repo 原论文完整复现。",
        "- MemoryOS/MemInsight/GMemory/DDO 经本项目统一 LoCoMo wrapper 接入同一数据、同一 QA 模型与同一指标脚本。",
        "- GMemory/DDO 无 LoCoMo 原生入口，本报告仅标注为 component-level wrapper，不作为原论文复现。",
        f"- 模型：{manifest.get('model')}；样本数：{manifest.get('validation', {}).get('sample_count')}；并发：{manifest.get('max_workers')}。",
        "",
        "## 客观结论",
    ]
    for sentence in (
        comparison_sentence(overall_with_category, category="overall", metric="qa_f1", label="overall"),
        comparison_sentence(overall_with_category, category="overall", metric="bleu1", label="overall"),
        comparison_sentence(overall_with_category, category="overall", metric="avg_tokens", label="overall 平均 token"),
        comparison_sentence(by_category, category="temporal", metric="qa_f1", label="temporal 类别"),
    ):
        if sentence:
            lines.append(sentence)
    lines.extend(["", "## 主指标：Overall", "| Method | N | QA F1 | BLEU-1 | Avg Tokens |", "|---|---:|---:|---:|---:|"])
    for row in overall:
        lines.append(
            "| {method} | {n} | {f1:.4f} | {bleu:.4f} | {tokens:.1f} |".format(
                method=method_label(str(row.get("method"))),
                n=int(float(row.get("n") or 0)),
                f1=float(row.get("qa_f1") or 0),
                bleu=float(row.get("bleu1") or 0),
                tokens=float(row.get("avg_tokens") or 0),
            )
        )
    lines.extend(["", "## 主指标：按 LoCoMo 类别", "| Method | Category | N | QA F1 | BLEU-1 | Avg Tokens |", "|---|---|---:|---:|---:|---:|"])
    for row in by_category:
        lines.append(
            "| {method} | {category} | {n} | {f1:.4f} | {bleu:.4f} | {tokens:.1f} |".format(
                method=method_label(str(row.get("method"))),
                category=row.get("category"),
                n=int(float(row.get("n") or 0)),
                f1=float(row.get("qa_f1") or 0),
                bleu=float(row.get("bleu1") or 0),
                tokens=float(row.get("avg_tokens") or 0),
            )
        )
    lines.extend(["", "## 附加文本相似度指标", "| Method | Category | N | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | SBERT |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in official_style:
        sbert = row.get("sbert_similarity")
        lines.append(
            "| {method} | {category} | {n} | {r1:.4f} | {r2:.4f} | {rl:.4f} | {b2:.4f} | {b3:.4f} | {b4:.4f} | {meteor:.4f} | {sbert} |".format(
                method=method_label(str(row.get("method"))),
                category=row.get("category"),
                n=int(float(row.get("n") or 0)),
                r1=float(row.get("rouge1_f") or 0),
                r2=float(row.get("rouge2_f") or 0),
                rl=float(row.get("rougeL_f") or 0),
                b2=float(row.get("bleu2") or 0),
                b3=float(row.get("bleu3") or 0),
                b4=float(row.get("bleu4") or 0),
                meteor=float(row.get("meteor") or 0),
                sbert="N/A" if sbert in {"", None} else f"{float(sbert):.4f}",
            )
        )
    lines.extend(["", "## 辅助审计指标", "| Method | N | Exact | Soft | Retrieved | Guard Pass | Fallback |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in auxiliary:
        lines.append(
            "| {method} | {n} | {exact:.4f} | {soft:.4f} | {retrieved:.1f} | {guard:.4f} | {fallback:.4f} |".format(
                method=method_label(str(row.get("method"))),
                n=int(float(row.get("n") or 0)),
                exact=float(row.get("exact_match") or 0),
                soft=float(row.get("soft_match") or 0),
                retrieved=float(row.get("retrieved_memory_count") or 0),
                guard=float(row.get("guard_pass_rate") or 0),
                fallback=float(row.get("fallback_rate") or 0),
            )
        )
    skipped = (manifest.get("metric_policy") or {}).get("skipped_metrics") or {}
    lines.extend(
        [
            "",
            "## 运行审计",
            f"- Validation passed: {manifest.get('validation', {}).get('passed')}",
            f"- Full-context shortcut guard passed: {manifest.get('guard', {}).get('passed')}",
            f"- Malformed-output fallback count: {fallback_count}",
            f"- Skipped metrics: {json.dumps(skipped, ensure_ascii=False)}",
        ]
    )
    blocked = manifest.get("blocked_methods") or []
    if blocked:
        lines.extend(["", "## Official Code Blockers"])
        for item in blocked:
            lines.append(f"- {item.get('method')}: {item.get('status')} ({item.get('env', 'no-env')}); repo={item.get('official_repo')}")
    write_text(run_dir / "benchmark_report_zh.md", "\n".join(lines) + "\n")
def benchmark_status() -> list[dict[str, Any]]:
    rows = []
    for dataset, spec in sorted(NATIVE_BENCHMARKS.items()):
        for method in spec.official_methods:
            status = official_method_status(method, dataset)
            status["official_repo"] = METHOD_OFFICIAL_REPOS.get(method, status.get("official_repo") or spec.official_repo)
            rows.append(status)
    return rows
