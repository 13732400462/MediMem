from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from medimem.benchmark import (
    NATIVE_BENCHMARKS,
    answer_with_context,
    build_client_for_base_url,
    evaluate_benchmark_predictions,
    judge_predictions,
    load_dialsim_samples,
    load_frozen_sample_ids,
    load_native_samples,
    sample_benchmark_rows,
    select_frozen_samples,
    source_file_manifest,
    validate_horizontal_run,
    write_csv,
)
from medimem.io_utils import append_jsonl, ensure_dir, write_jsonl, write_text


METHOD = "official_mem0_timeline_adapter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("locomo", "longmemeval", "dialsim", "rhelm"))
    parser.add_argument("--dataset-path")
    parser.add_argument("--sample-manifest")
    parser.add_argument("--sample-n", type=int)
    parser.add_argument("--random-seed", type=int, default=20260716)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--output-root", default="runs/nonmedical_timeline_mem0")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--reset-store", action="store_true")
    parser.add_argument("--judge-answers", action="store_true")
    return parser.parse_args()


def memory_config(store_dir: Path, endpoint: str, collection: str) -> dict[str, Any]:
    return {
        "llm": {
            "provider": "vllm",
            "config": {
                "model": "qwen3-vl-8b",
                "vllm_base_url": endpoint,
                "api_key": "EMPTY",
                "max_tokens": 512,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "fastembed",
            "config": {"model": "BAAI/bge-small-en-v1.5", "embedding_dims": 384},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(store_dir / "qdrant"),
                "collection_name": collection,
                "embedding_model_dims": 384,
            },
        },
        "history_db_path": str(store_dir / "history.db"),
    }


def chunk_timeline(turns: list[dict[str, Any]], max_chars: int = 5000) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        text = str(turn.get("text") or "").strip()
        if text:
            by_session[str(turn.get("session") or turn.get("time") or "unknown")].append(turn)
    chunks: list[dict[str, Any]] = []
    for session, values in by_session.items():
        lines: list[str] = []
        refs: set[str] = set()
        current_chars = 0
        for turn in values:
            turn_refs = [str(ref) for ref in turn.get("evidence_refs", []) if str(ref).strip()]
            line = (
                f"date={turn.get('session_date') or turn.get('time') or ''} "
                f"speaker={turn.get('speaker') or ''} refs={','.join(turn_refs)} "
                f"text={str(turn.get('text') or '').strip()}"
            )
            if lines and current_chars + len(line) + 1 > max_chars:
                chunks.append({"text": f"Session {session}\n" + "\n".join(lines), "evidence_refs": sorted(refs)})
                lines, refs, current_chars = [], set(), 0
            lines.append(line)
            refs.update(turn_refs)
            current_chars += len(line) + 1
        if lines:
            chunks.append({"text": f"Session {session}\n" + "\n".join(lines), "evidence_refs": sorted(refs)})
    return chunks


def extract_memories(result: Any, top_k: int) -> list[dict[str, Any]]:
    values = (result.get("results") or result.get("memories") or []) if isinstance(result, dict) else (result or [])
    memories: list[dict[str, Any]] = []
    for item in values:
        if isinstance(item, dict):
            text = item.get("memory") or item.get("text") or item.get("content")
            metadata = item.get("metadata") or {}
            refs = metadata.get("evidence_refs") or []
        else:
            text, refs = item, []
        if text:
            memories.append({"text": str(text).strip(), "evidence_refs": [str(ref) for ref in refs]})
    return memories[:top_k]


def shard_groups(groups: list[tuple[str, list[dict[str, Any]]]], workers: int) -> list[list[tuple[str, list[dict[str, Any]]]]]:
    shards = [[] for _ in range(workers)]
    for index, group in enumerate(groups):
        shards[index % workers].append(group)
    return [shard for shard in shards if shard]


def run_worker(
    worker_id: int,
    dataset: str,
    groups: list[tuple[str, list[dict[str, Any]]]],
    run_dir_text: str,
    endpoint: str,
    top_k: int,
    require_api: bool,
    reset_store: bool,
) -> dict[str, Any]:
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    run_dir = Path(run_dir_text)
    worker_dir = run_dir / "workers" / f"worker_{worker_id:03d}"
    ensure_dir(worker_dir / "predictions")
    store_dir = worker_dir / "mem0_store"
    if reset_store and store_dir.exists():
        shutil.rmtree(store_dir)
    os.environ["MEM0_DIR"] = str(worker_dir / "mem0_home")
    from mem0 import Memory

    memory = Memory.from_config(memory_config(store_dir, endpoint, f"mem0_{dataset}_{worker_id:03d}"))
    client, blocker = build_client_for_base_url(endpoint, require_api=require_api)
    if blocker and require_api:
        raise RuntimeError(blocker)
    output = worker_dir / "predictions" / f"mem0_{dataset}.jsonl"
    predictions: list[dict[str, Any]] = []
    for conversation_id, samples in groups:
        chunks = chunk_timeline(list(samples[0].get("turns") or []))
        for chunk_index, chunk in enumerate(chunks):
            memory.add(
                chunk["text"],
                user_id=conversation_id,
                infer=False,
                metadata={"chunk_index": chunk_index, "evidence_refs": chunk["evidence_refs"]},
            )
        append_jsonl(worker_dir / "progress.jsonl", {"stage": "memory_ready", "conversation_id": conversation_id, "chunks": len(chunks)})
        for sample in samples:
            result = memory.search(sample["question"], filters={"user_id": conversation_id}, top_k=top_k)
            retrieved = extract_memories(result, top_k)
            context = "[RETRIEVED_MEMORIES]\n" + "\n".join(f"- {item['text']}" for item in retrieved)
            pred = answer_with_context(sample, METHOD, context, client, fail_on_llm_error=require_api)
            pred.update(
                {
                    "retrieved_memory_count": len(retrieved),
                    "retrieved_evidence_refs": sorted({ref for item in retrieved for ref in item["evidence_refs"]}),
                    "retrieved_evidence_refs_at5": sorted({ref for item in retrieved[:5] for ref in item["evidence_refs"]}),
                    "guard_passed": True,
                    "official_baseline": "mem0",
                    "official_repo": "https://github.com/mem0ai/mem0",
                    "baseline_reproduction_level": "official_native_memory_api_timeline_adapter",
                    "conversation_id": conversation_id,
                    "worker_id": worker_id,
                    "endpoint": endpoint,
                }
            )
            predictions.append(pred)
            append_jsonl(output, pred)
    return {"worker_id": worker_id, "predictions": str(output), "count": len(predictions)}


def load_samples(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], int]:
    source = Path(args.dataset_path or NATIVE_BENCHMARKS[args.dataset].default_path)
    if args.dataset == "dialsim":
        samples = load_dialsim_samples(source, sample_n=args.sample_n, random_seed=args.random_seed)
        available = len(samples)
    else:
        loaded = load_native_samples(args.dataset, source)
        available = len(loaded)
        samples = sample_benchmark_rows(loaded, sample_n=args.sample_n, random_seed=args.random_seed)
        if args.sample_manifest:
            samples = select_frozen_samples(loaded, load_frozen_sample_ids(args.sample_manifest))
    if not samples:
        raise RuntimeError("No benchmark samples selected.")
    return source, samples, available


def main() -> None:
    args = parse_args()
    source, samples, available = load_samples(args)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("conversation_id") or sample["sample_id"])].append(sample)
    groups = sorted(grouped.items())
    workers = max(1, min(args.workers, len(groups)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(args.output_root) / f"mem0_{args.dataset}_{stamp}_pid{os.getpid()}"
    ensure_dir(run_dir / "predictions")
    ensure_dir(run_dir / "workers")
    write_jsonl(run_dir / "samples.jsonl", samples)
    selected_ids = [str(sample["sample_id"]) for sample in samples]
    sample_hash = hashlib.sha256(("\n".join(selected_ids) + "\n").encode()).hexdigest()
    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "dataset_path": str(source),
        "available_samples": available,
        "sample_count": len(samples),
        "sample_ids_sha256": sample_hash,
        "source_files": source_file_manifest(source),
        "sample_manifest_input": args.sample_manifest,
        "random_seed": args.random_seed,
        "method": METHOD,
        "top_k": args.top_k,
        "workers": workers,
        "endpoint": args.endpoint,
        "adapter_note": "Official Memory.add(infer=False)/Memory.search; raw frozen timeline chunks are stored losslessly so ingestion policy cannot discard benchmark evidence.",
    }
    write_text(run_dir / "experiment_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    results: list[dict[str, Any]] = []
    # DialSim is loaded through pyarrow in the parent process.  Spawn clean
    # workers so pyarrow's system C++ runtime is not inherited before Mem0/ICU
    # imports the Conda runtime it was built against.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [
            pool.submit(run_worker, index, args.dataset, shard, str(run_dir), args.endpoint, args.top_k, args.require_api, args.reset_store)
            for index, shard in enumerate(shard_groups(groups, workers))
        ]
        for future in as_completed(futures):
            results.append(future.result())
    by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        for line in Path(result["predictions"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                pred = json.loads(line)
                by_id[str(pred["sample_id"])] = pred
    predictions = [by_id[str(sample["sample_id"])] for sample in samples if str(sample["sample_id"]) in by_id]
    client, blocker = build_client_for_base_url(args.endpoint, require_api=args.require_api)
    judge_results = judge_predictions(samples, predictions, client, require_api=args.require_api, max_workers=workers) if args.judge_answers else []
    write_jsonl(run_dir / "predictions" / f"mem0_{args.dataset}.jsonl", predictions)
    write_jsonl(run_dir / "judge_results.jsonl", judge_results)
    write_csv(run_dir / f"{args.dataset}_metrics.csv", evaluate_benchmark_predictions(samples, predictions))
    validation = validate_horizontal_run(samples, predictions)
    manifest.update(
        {
            "blocker": blocker,
            "validation": validation,
            "fallback_count": sum("fallback_reason" in pred for pred in predictions),
            "judge_missing_count": sum(pred.get("judge_correct") is None for pred in predictions) if args.judge_answers else None,
        }
    )
    write_text(run_dir / "experiment_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if not validation["passed"] or manifest["fallback_count"] or (args.judge_answers and manifest["judge_missing_count"]):
        raise RuntimeError(f"Mem0 formal gate failed: {manifest}")
    print(run_dir)


if __name__ == "__main__":
    main()

