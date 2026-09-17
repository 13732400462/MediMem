from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from medimem.benchmark import (
    NATIVE_BENCHMARKS,
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
from medimem.llm import extract_json_object


METHOD = "official_letta_memgpt_timeline_adapter"


@contextmanager
def letta_metadata_lock():
    path = Path(os.environ.get("LETTA_METADATA_LOCK", "/tmp/medimem_letta_timeline.lock"))
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("locomo", "longmemeval", "dialsim", "rhelm"))
    parser.add_argument("--dataset-path")
    parser.add_argument("--sample-manifest")
    parser.add_argument("--sample-n", type=int)
    parser.add_argument("--random-seed", type=int, default=20260716)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--output-root", default="runs/nonmedical_timeline_letta")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--judge-answers", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--qa-max-steps", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def parse_json(text: str) -> dict[str, Any]:
    try:
        return extract_json_object(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidate = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text[start : end + 1])
            return json.loads(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", candidate))
        raise


def assistant_texts(dump: dict[str, Any]) -> list[str]:
    return [
        str(message.get("content"))
        for message in dump.get("messages") or []
        if message.get("message_type") == "assistant_message" and message.get("content")
    ]


def successful_memory_tools(dump: dict[str, Any]) -> list[str]:
    return [
        str(message.get("name"))
        for message in dump.get("messages") or []
        if message.get("message_type") == "tool_return_message"
        and message.get("name") in {"memory_insert", "memory_replace"}
        and message.get("status") == "success"
    ]


def chunk_timeline(turns: list[dict[str, Any]], max_chars: int = 5000) -> list[str]:
    by_session: dict[str, list[str]] = defaultdict(list)
    for turn in turns:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        refs = ",".join(str(ref) for ref in turn.get("evidence_refs", []) if str(ref).strip())
        session = str(turn.get("session") or turn.get("time") or "unknown")
        by_session[session].append(
            f"date={turn.get('session_date') or turn.get('time') or ''} speaker={turn.get('speaker') or ''} refs={refs} text={text}"
        )
    chunks: list[str] = []
    for session, lines in by_session.items():
        current = f"Session {session}\n"
        for line in lines:
            if len(current) + len(line) + 1 > max_chars and current.strip() != f"Session {session}":
                chunks.append(current.strip())
                current = f"Session {session} continued\n"
            current += line + "\n"
        if current.strip():
            chunks.append(current.strip())
    return chunks


def shard_groups(groups: list[tuple[str, list[dict[str, Any]]]], workers: int) -> list[list[tuple[str, list[dict[str, Any]]]]]:
    shards = [[] for _ in range(workers)]
    for index, group in enumerate(groups):
        shards[index % workers].append(group)
    return [shard for shard in shards if shard]


async def create_run(server: Any, actor: Any, agent_id: Any, run_type: str, item_id: str) -> Any:
    from letta.schemas.run import Run

    return await server.run_manager.create_run(
        Run(agent_id=agent_id, background=False, metadata_={"run_type": run_type, "item_id": item_id}), actor=actor
    )


async def run_worker_async(
    worker_id: int,
    dataset: str,
    groups: list[tuple[str, list[dict[str, Any]]]],
    run_dir_text: str,
    endpoint: str,
    max_steps: int,
    qa_max_steps: int,
    max_tokens: int,
    require_api: bool,
) -> dict[str, Any]:
    os.environ["LETTA_PG_URI"] = os.environ.get("LETTA_PG_URI", "postgresql://ymu@127.0.0.1:55432/letta")
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_BASE_URL"] = endpoint
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    worker_dir = Path(run_dir_text) / "workers" / f"worker_{worker_id:03d}"
    ensure_dir(worker_dir / "predictions")
    ensure_dir(worker_dir / "letta_home")
    os.environ["HOME"] = str(worker_dir / "letta_home")
    os.environ["LETTA_DIR"] = str(worker_dir / "letta_home" / ".letta")

    from letta.agents.agent_loop import AgentLoop
    from letta.config import LettaConfig
    from letta.schemas.agent import CreateAgent
    from letta.schemas.block import CreateBlock
    from letta.schemas.embedding_config import EmbeddingConfig
    from letta.schemas.llm_config import LLMConfig
    from letta.schemas.message import MessageCreate, MessageRole
    from letta.server.server import SyncServer

    LettaConfig.load().save()
    server = SyncServer()
    with letta_metadata_lock():
        await server.init_async()
    actor = server.default_user
    llm = LLMConfig(
        model="qwen3-vl-8b",
        model_endpoint_type="openai",
        model_endpoint=endpoint,
        context_window=16384,
        max_tokens=max_tokens,
        temperature=0.0,
        handle="vllm/qwen3-vl-8b",
        enable_reasoner=False,
    )
    embedding = EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_endpoint=endpoint,
        embedding_model="qwen3-vl-8b",
        embedding_dim=4096,
    )
    output = worker_dir / "predictions" / f"letta_{dataset}.jsonl"
    predictions: list[dict[str, Any]] = []
    for conversation_id, samples in groups:
        agent = None
        try:
            request = CreateAgent(
                name=f"letta_{dataset}_{worker_id}_{hashlib.sha1(conversation_id.encode()).hexdigest()[:12]}",
                memory_blocks=[
                    CreateBlock(label="persona", value="Answer long-term memory questions using only durable stored timeline facts."),
                    CreateBlock(label="human", value="The user provides one frozen benchmark timeline."),
                    CreateBlock(
                        label="timeline_memory",
                        value="No timeline facts stored yet.",
                        limit=14000,
                        description="Compact dated facts and source refs from the current frozen timeline.",
                    ),
                ],
                llm_config=llm,
                embedding_config=embedding,
                include_base_tools=True,
                include_base_tool_rules=True,
                initial_message_sequence=[],
                message_buffer_autoclear=True,
                enable_sleeptime=False,
                parallel_tool_calls=False,
            )
            with letta_metadata_lock():
                agent = await server.create_agent_async(request, actor=actor)
            loop = AgentLoop.load(agent_state=agent, actor=actor)
            chunks = chunk_timeline(list(samples[0].get("turns") or []))
            for chunk_index, chunk in enumerate(chunks):
                prompt = (
                    "Update timeline_memory with the durable dated facts in this segment. Preserve compact refs tokens. "
                    "Use memory_insert for new facts; if the block is crowded, use memory_replace to consolidate before inserting. "
                    "Do not answer a question and execute at least one memory tool.\n\n" + chunk
                )
                run = await create_run(server, actor, agent.id, "timeline_ingest", f"{conversation_id}:{chunk_index}")
                response = await loop.step([MessageCreate(role=MessageRole.user, content=prompt)], max_steps=max_steps, run_id=run.id)
                tools = successful_memory_tools(response.model_dump(mode="json"))
                append_jsonl(worker_dir / "progress.jsonl", {"stage": "ingest", "conversation_id": conversation_id, "chunk": chunk_index, "tools": tools})
                if require_api and not tools:
                    raise RuntimeError(f"Letta executed no successful memory tool for {conversation_id} chunk {chunk_index}")
            for sample in samples:
                started = time.time()
                prompt = (
                    "Answer from timeline_memory only. Return exactly one minified JSON object with keys answer and confidence. "
                    "answer must be concise and no longer than 12 words; confidence must be 0 to 1. Do not modify memory.\nQuestion: "
                    + str(sample["question"])
                )
                raw = None
                dump: dict[str, Any] = {}
                for attempt in range(3):
                    run = await create_run(server, actor, agent.id, "timeline_qa", f"{sample['sample_id']}:{attempt}")
                    response = await loop.step([MessageCreate(role=MessageRole.user, content=prompt)], max_steps=qa_max_steps, run_id=run.id)
                    dump = response.model_dump(mode="json")
                    texts = assistant_texts(dump)
                    if texts:
                        try:
                            raw = parse_json(texts[-1])
                            break
                        except Exception:
                            pass
                if raw is None:
                    raise RuntimeError(f"Letta returned no parseable answer for {sample['sample_id']}")
                usage = dump.get("usage") or {}
                pred = {
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "split": sample["split"],
                    "method": METHOD,
                    "answer": str(raw.get("answer") or "").strip(),
                    "confidence": float(raw.get("confidence") or 0.5),
                    "evidence": [],
                    "usage": {
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "total_tokens": int(usage.get("total_tokens") or 0),
                        "latency_ms": int((time.time() - started) * 1000),
                    },
                    "retrieved_memory_count": None,
                    "guard_passed": True,
                    "official_baseline": "letta_memgpt",
                    "official_repo": "https://github.com/letta-ai/letta",
                    "baseline_reproduction_level": "official_core_memory_timeline_adapter",
                    "conversation_id": conversation_id,
                    "agent_id": str(agent.id),
                    "worker_id": worker_id,
                    "endpoint": endpoint,
                }
                predictions.append(pred)
                append_jsonl(output, pred)
        finally:
            if agent is not None:
                await server.agent_manager.delete_agent_async(agent.id, actor=actor)
    return {"worker_id": worker_id, "predictions": str(output), "count": len(predictions)}


def run_worker(*args: Any) -> dict[str, Any]:
    return asyncio.run(run_worker_async(*args))


def load_samples(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], int]:
    source = Path(args.dataset_path or NATIVE_BENCHMARKS[args.dataset].default_path)
    if args.dataset == "dialsim":
        samples = load_dialsim_samples(source, sample_n=args.sample_n, random_seed=args.random_seed)
        return source, samples, len(samples)
    loaded = load_native_samples(args.dataset, source)
    samples = sample_benchmark_rows(loaded, sample_n=args.sample_n, random_seed=args.random_seed)
    if args.sample_manifest:
        samples = select_frozen_samples(loaded, load_frozen_sample_ids(args.sample_manifest))
    return source, samples, len(loaded)


def main() -> None:
    args = parse_args()
    source, samples, available = load_samples(args)
    if not samples:
        raise RuntimeError("No benchmark samples selected.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("conversation_id") or sample["sample_id"])].append(sample)
    groups = sorted(grouped.items())
    workers = max(1, min(args.workers, len(groups)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(args.output_root) / f"letta_{args.dataset}_{stamp}_pid{os.getpid()}"
    ensure_dir(run_dir / "predictions")
    ensure_dir(run_dir / "workers")
    write_jsonl(run_dir / "samples.jsonl", samples)
    ids = [str(sample["sample_id"]) for sample in samples]
    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "dataset_path": str(source),
        "available_samples": available,
        "sample_count": len(samples),
        "sample_ids_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "source_files": source_file_manifest(source),
        "sample_manifest_input": args.sample_manifest,
        "random_seed": args.random_seed,
        "method": METHOD,
        "workers": workers,
        "endpoint": args.endpoint,
        "adapter_note": "Official SyncServer/CreateAgent/AgentLoop and core-memory tools; Evidence R@5 is not reported because core memory is not ranked retrieval.",
    }
    write_text(run_dir / "experiment_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_worker, index, args.dataset, shard, str(run_dir), args.endpoint, args.max_steps, args.qa_max_steps, args.max_tokens, args.require_api)
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
    judges = judge_predictions(samples, predictions, client, require_api=args.require_api, max_workers=workers) if args.judge_answers else []
    write_jsonl(run_dir / "predictions" / f"letta_{args.dataset}.jsonl", predictions)
    write_jsonl(run_dir / "judge_results.jsonl", judges)
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
        raise RuntimeError(f"Letta formal gate failed: {manifest}")
    print(run_dir)


if __name__ == "__main__":
    main()

