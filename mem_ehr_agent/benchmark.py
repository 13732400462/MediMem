from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .amem_baseline import SourceAlignedAMEMSystem
from .config import get_deepseek_config
from .io_utils import ensure_dir, read_jsonl, write_jsonl, write_text
from .llm import DeepSeekClient, extract_json_object
from .metrics import token_f1


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


LOCAL_METHODS = {"ours", "amem"}
METHOD_OFFICIAL_REPOS = {
    "amem": "https://github.com/agiresearch/A-mem",
    "memoryos": "https://github.com/BAI-LAB/MemoryOS",
    "meminsight": "https://github.com/amazon-science/MemInsight",
    "gmemory": "https://github.com/bingreeky/GMemory",
    "ddo": "https://github.com/zh-jia/DDO",
}


def make_benchmark_run_dir(root: str | Path = "runs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"benchmark_{stamp}"
    ensure_dir(path / "predictions")
    return path


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one benchmark method is required.")
    return methods


def build_client(require_api: bool = False) -> tuple[DeepSeekClient | None, str | None]:
    client = DeepSeekClient(get_deepseek_config())
    ok, msg = client.healthcheck()
    if ok:
        return client, None
    if require_api:
        raise RuntimeError(f"DeepSeek API unavailable: {msg}")
    return None, f"DeepSeek API unavailable, using deterministic QA fallback: {msg}"


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
        sample_id = str(item.get("sample_id") or item.get("conversation_id") or f"conv-{sample_idx}")
        for qa_idx, qa in enumerate(item.get("qa", [])):
            answer = qa.get("adversarial_answer") if qa.get("category") == 5 and qa.get("adversarial_answer") else qa.get("answer")
            samples.append(
                {
                    "sample_id": f"{sample_id}__qa_{qa_idx:04d}",
                    "dataset": "locomo",
                    "split": "official",
                    "context": context,
                    "question": normalize_answer(qa.get("question")),
                    "answer": normalize_answer(answer),
                    "evidence": qa.get("evidence", []),
                    "metadata": {"conversation_index": sample_idx, "qa_index": qa_idx, "category": qa.get("category")},
                }
            )
            if limit and len(samples) >= limit:
                return samples
    return samples


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
                "Return one compact JSON object with keys answer, evidence, confidence. "
                "answer must be concise. evidence must be an array of at most 3 strings, "
                "and each evidence string must be no longer than 80 characters. "
                "Do not quote long dialogue spans."
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


def deterministic_qa_fallback(sample: dict[str, Any], method: str, reason: str = "") -> dict[str, Any]:
    context = sample.get("context", "")
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
        return deterministic_qa_fallback(sample, method)
    try:
        result = client.chat(qa_prompt(sample, method, context), temperature=0.0, max_tokens=900)
        raw = extract_json_object(result.text)
        evidence = raw.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        return {
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "split": sample["split"],
            "method": method,
            "answer": normalize_answer(raw.get("answer")),
            "evidence": [str(item) for item in evidence],
            "confidence": float(raw.get("confidence", 0.5) or 0.5),
            "usage": result.usage,
        }
    except Exception as exc:  # noqa: BLE001
        if fail_on_llm_error:
            raise RuntimeError(f"Benchmark QA failed for {sample['sample_id']} ({method}): {exc}") from exc
        return deterministic_qa_fallback(sample, method, reason=str(exc))


def amem_retrieved_context(sample: dict[str, Any], *, top_k: int = 8) -> tuple[str, int]:
    system = SourceAlignedAMEMSystem()
    for idx, line in enumerate(str(sample.get("context") or "").splitlines()):
        text = line.strip()
        if not text:
            continue
        system.add_note(
            content=text,
            time=str(idx),
            context=f"{sample['dataset']} native memory line for {sample['sample_id']}",
            tags=["dialogue", "native_benchmark"],
            category="dialogue",
        )
    retrieved = system.find_related_memories_raw(str(sample.get("question") or ""), k=top_k)
    return f"[A-MEM RETRIEVED MEMORY]\n{retrieved}\n\n[QUESTION]\n{sample['question']}", min(top_k, len(system.memories))


def run_local_method(sample: dict[str, Any], method: str, client: DeepSeekClient | None, *, require_api: bool) -> dict[str, Any]:
    if method == "ours":
        context = (
            "[FULL LONG-TERM CONTEXT]\n"
            f"{sample['context']}\n\n"
            "[QUESTION]\n"
            f"{sample['question']}\n"
            "Use the full memory timeline and answer with the most specific supported fact."
        )
        pred = answer_with_context(sample, "ours_native_adapter", context, client, fail_on_llm_error=require_api)
        pred["retrieved_memory_count"] = len(str(sample.get("context", "")).splitlines())
        return pred
    if method == "amem":
        context, retrieved = amem_retrieved_context(sample)
        pred = answer_with_context(sample, "official_amem_native_adapter", context, client, fail_on_llm_error=require_api)
        pred["retrieved_memory_count"] = retrieved
        pred["official_code_priority_note"] = "Uses the local A-MEM source-aligned memory flow available in this workspace."
        return pred
    raise ValueError(f"Unsupported local benchmark method: {method}")


def official_method_status(method: str, dataset: str) -> dict[str, Any]:
    spec = NATIVE_BENCHMARKS[dataset]
    env_by_method = {
        "memoryos": "MEMORYOS_REPO",
        "meminsight": "MEMINSIGHT_REPO",
        "gmemory": "GMEMORY_REPO",
        "ddo": "DDO_REPO",
    }
    env_name = env_by_method.get(method) or spec.required_env
    repo_path = os.getenv(env_name or "")
    if not env_name:
        return {"method": method, "dataset": dataset, "status": "local_adapter_available"}
    if repo_path and Path(repo_path).exists():
        return {
            "method": method,
            "dataset": dataset,
            "status": "official_repo_detected_manual_runner_required",
            "env": env_name,
            "repo_path": repo_path,
        }
    return {
        "method": method,
        "dataset": dataset,
        "status": "blocked_official_repo_not_configured",
        "env": env_name,
        "official_repo": METHOD_OFFICIAL_REPOS.get(method, spec.official_repo),
    }


def evaluate_benchmark_predictions(samples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    buckets: dict[tuple[str, str], list[dict[str, float]]] = {}
    for pred in predictions:
        sample = sample_by_id[pred["sample_id"]]
        answer = normalize_answer(pred.get("answer"))
        gold = normalize_answer(sample.get("answer"))
        key = (str(pred.get("method")), str(pred.get("split")))
        buckets.setdefault(key, []).append(
            {
                "exact_match": 1.0 if answer.lower() == gold.lower() and answer else 0.0,
                "soft_match": 1.0 if soft_match(answer, gold) else 0.0,
                "qa_f1": token_f1(answer, gold),
                "bleu1": simple_bleu1(answer, gold),
                "avg_tokens": float((pred.get("usage") or {}).get("total_tokens", 0) or 0),
                "retrieved_memory_count": float(pred.get("retrieved_memory_count", 0) or 0),
            }
        )
    rows: list[dict[str, Any]] = []
    for (method, split), vals in sorted(buckets.items()):
        n = len(vals)
        row: dict[str, Any] = {"method": method, "split": split, "n": n}
        for metric in ("exact_match", "soft_match", "qa_f1", "bleu1", "avg_tokens", "retrieved_memory_count"):
            row[metric] = sum(v[metric] for v in vals) / n if n else 0.0
        rows.append(row)
    return rows


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
    output_root: str | Path = "runs",
    require_api: bool = False,
) -> Path:
    samples = load_native_samples(dataset, dataset_path, limit=limit)
    run_dir = make_benchmark_run_dir(output_root)
    client, blocker = build_client(require_api=require_api)
    predictions: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for method in methods:
        if method in LOCAL_METHODS:
            for sample in samples:
                predictions.append(run_local_method(sample, method, client, require_api=require_api))
        else:
            blocked.append(official_method_status(method, dataset))
    write_jsonl(run_dir / "samples.jsonl", samples)
    write_jsonl(run_dir / "predictions" / f"{dataset}.jsonl", predictions)
    write_jsonl(run_dir / "blocked_methods.jsonl", blocked)
    metrics = evaluate_benchmark_predictions(samples, predictions)
    write_csv(run_dir / f"{dataset}_metrics.csv", metrics)
    validation = validate_horizontal_run(samples, predictions) if predictions else {"passed": False, "failures": ["no predictions"]}
    manifest = {
        "dataset": dataset,
        "dataset_path": str(dataset_path or NATIVE_BENCHMARKS[dataset].default_path),
        "methods": methods,
        "local_methods": sorted({pred["method"] for pred in predictions}),
        "blocked_methods": blocked,
        "blocker": blocker,
        "validation": validation,
        "official_repo": NATIVE_BENCHMARKS[dataset].official_repo,
    }
    write_text(run_dir / "benchmark_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    render_benchmark_report(run_dir, dataset, metrics, manifest)
    return run_dir


def render_benchmark_report(run_dir: Path, dataset: str, metrics: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
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


def benchmark_status() -> list[dict[str, Any]]:
    rows = []
    for dataset, spec in sorted(NATIVE_BENCHMARKS.items()):
        for method in spec.official_methods:
            status = official_method_status(method, dataset)
            status["official_repo"] = METHOD_OFFICIAL_REPOS.get(method, status.get("official_repo") or spec.official_repo)
            rows.append(status)
    return rows
