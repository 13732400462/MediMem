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

from mem_ehr_agent.benchmark import (
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
from mem_ehr_agent.io_utils import append_jsonl, ensure_dir, write_jsonl, write_text
from mem_ehr_agent.llm import extract_json_object
from mem_ehr_agent.extractive_timeline import EXTRACTION_VERSION, build_extractive_timeline
from mem_ehr_agent.resume import load_valid_resume_predictions


METHOD = "official_letta_memgpt_timeline_adapter"
TIMELINE_CHUNK_MAX_CHARS = 36000
CORE_MEMORY_LIMIT_CHARS = 7500
CORE_MEMORY_TARGET_CHARS = 6500
COMPACTION_MAX_TOKENS = 1800


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
    parser.add_argument("--resume-predictions", action="append", default=[])
    parser.add_argument("--resume-allowed-endpoint", action="append", default=[])
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
    parser.add_argument("--ingestion-mode", choices=("rolling_llm", "extractive"), default="rolling_llm")
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
    texts: list[str] = []
    for message in dump.get("messages") or []:
        message_type = str(message.get("message_type") or "")
        role = str(message.get("role") or "")
        content = message.get("content")
        if content and ("assistant" in message_type or role == "assistant"):
            texts.append(str(content))
    return texts


def recover_plain_answer(text: str) -> tuple[str | None, str]:
    """Recover a safe answer from a non-JSON assistant reply.

    This is intentionally conservative: tool calls and long explanations remain
    invalid, while a short answer (including an ``answer:`` field) is retained
    as a valid model prediction for downstream judging.
    """
    cleaned = text.strip().strip("`").strip()
    if not cleaned or "<tool_call>" in cleaned or len(cleaned) > 2000:
        return None, "json"
    match = re.search(r'["\']answer["\']\s*:\s*["\']([^"\']+)', cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), "recovered_answer_field"
    label_match = re.fullmatch(r"(?:answer\s*:\s*)?(.+)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if label_match:
        answer = label_match.group(1).strip()
        if answer:
            return answer, "plain_assistant_answer"
    return None, "json"


def qa_parse_diagnostics(dump: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep compact model-return evidence when a QA turn cannot be parsed."""
    diagnostics: list[dict[str, Any]] = []
    for message in dump.get("messages") or []:
        content = str(message.get("content") or "")
        if not content:
            continue
        diagnostics.append(
            {
                "message_type": str(message.get("message_type") or ""),
                "role": message.get("role"),
                "name": message.get("name"),
                "status": message.get("status"),
                "content": content[:2000],
            }
        )
    return diagnostics


def successful_memory_tools(dump: dict[str, Any]) -> list[str]:
    return [
        str(message.get("name"))
        for message in dump.get("messages") or []
        if message.get("message_type") == "tool_return_message"
        and message.get("name") in {"memory_insert", "memory_replace", "memory_rethink"}
        and message.get("status") == "success"
    ]


def memory_tool_diagnostics(dump: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for message in dump.get("messages") or []:
        message_type = str(message.get("message_type") or "")
        name = str(message.get("name") or "")
        if message_type not in {"assistant_message", "tool_call_message", "tool_return_message"}:
            continue
        if message_type != "assistant_message" and name not in {"memory_insert", "memory_replace", "memory_rethink"}:
            continue
        diagnostics.append(
            {
                "message_type": message_type,
                "name": name or None,
                "status": message.get("status"),
                "content": str(message.get("content") or "")[:2000],
                "tool_call": message.get("tool_call"),
            }
        )
    return diagnostics


def chunk_timeline(turns: list[dict[str, Any]], max_chars: int = TIMELINE_CHUNK_MAX_CHARS) -> list[str]:
    by_session: dict[str, list[str]] = defaultdict(list)
    for turn in turns:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        refs = ",".join(str(ref) for ref in turn.get("evidence_refs", []) if str(ref).strip())
        session = str(turn.get("session") or turn.get("time") or "unknown")
        prefix = f"date={turn.get('session_date') or turn.get('time') or ''} speaker={turn.get('speaker') or ''} refs={refs}"
        payload_chars = max(1000, max_chars - len(prefix) - 100)
        parts = [text[index : index + payload_chars] for index in range(0, len(text), payload_chars)] or [""]
        for part_index, part in enumerate(parts):
            marker = f" part={part_index + 1}/{len(parts)}" if len(parts) > 1 else ""
            by_session[session].append(f"{prefix}{marker} text={part}")
    chunks: list[str] = []
    current = ""
    for session, lines in by_session.items():
        header = f"Session {session}\n"
        session_started = False
        for line in lines:
            addition = (header if not session_started else "") + line + "\n"
            if current and len(current) + len(addition) > max_chars:
                chunks.append(current.strip())
                current = ""
                addition = f"Session {session} continued\n{line}\n"
            current += addition
            session_started = True
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
    ingestion_mode: str,
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
    from letta.schemas.block import BlockUpdate, CreateBlock
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
                        limit=CORE_MEMORY_LIMIT_CHARS,
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
            timeline_block = next(block for block in agent.blocks if block.label == "timeline_memory")
            timeline_turns = list(samples[0].get("turns") or [])
            if ingestion_mode == "extractive":
                timeline_memory, extraction_diagnostics = build_extractive_timeline(
                    timeline_turns,
                    max_chars=CORE_MEMORY_TARGET_CHARS,
                )
                timeline_block = await server.block_manager.update_block_async(
                    block_id=timeline_block.id,
                    block_update=BlockUpdate(value=timeline_memory),
                    actor=actor,
                )
                append_jsonl(
                    worker_dir / "progress.jsonl",
                    {
                        "stage": "ingest",
                        "conversation_id": conversation_id,
                        "chunk": 0,
                        "mode": "official_block_update_extractive",
                        "block_chars": len(timeline_memory),
                        **extraction_diagnostics,
                    },
                )
            else:
                os.environ["DEEPSEEK_API_KEY"] = "EMPTY"
                os.environ["DEEPSEEK_BASE_URL"] = endpoint
                os.environ["DEEPSEEK_MODEL"] = "qwen3-vl-8b"
                client, blocker = build_client_for_base_url(endpoint, require_api=require_api)
                if client is None:
                    raise RuntimeError(blocker or "Letta timeline compaction client unavailable")
                timeline_memory = ""
                chunks = chunk_timeline(timeline_turns)
                for chunk_index, chunk in enumerate(chunks):
                    combined = (timeline_memory + "\n" + chunk).strip()
                    compacted = False
                    if len(combined) > CORE_MEMORY_TARGET_CHARS:
                        result = client.chat(
                            [
                                {
                                    "role": "system",
                                    "content": (
                                        "Maintain a bounded long-term timeline. Rewrite the supplied existing memory plus new segment "
                                        f"into a factual chronological memory of at most {CORE_MEMORY_TARGET_CHARS} characters. Preserve important names, "
                                        "dates, events, preferences, relationships, and compact source refs for retained facts. "
                                        "Return only the rewritten memory, with no commentary."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"[EXISTING_MEMORY]\n{timeline_memory}\n\n[NEW_SEGMENT]\n{chunk}",
                                },
                            ],
                            temperature=0.0,
                            max_tokens=COMPACTION_MAX_TOKENS,
                        )
                        timeline_memory = str(result.text or "").strip()
                        compacted = True
                        if not timeline_memory:
                            raise RuntimeError(f"Letta block compaction returned empty output for {conversation_id} chunk {chunk_index}")
                        if len(timeline_memory) > CORE_MEMORY_TARGET_CHARS:
                            timeline_memory = timeline_memory[:CORE_MEMORY_TARGET_CHARS]
                    else:
                        timeline_memory = combined
                    timeline_block = await server.block_manager.update_block_async(
                        block_id=timeline_block.id,
                        block_update=BlockUpdate(value=timeline_memory),
                        actor=actor,
                    )
                    append_jsonl(
                        worker_dir / "progress.jsonl",
                        {
                            "stage": "ingest",
                            "conversation_id": conversation_id,
                            "chunk": chunk_index,
                            "mode": "official_block_update_rolling_llm",
                            "compacted": compacted,
                            "block_chars": len(timeline_memory),
                        },
                    )
            agent = await server.agent_manager.get_agent_by_id_async(agent_id=agent.id, actor=actor)
            loop = AgentLoop.load(agent_state=agent, actor=actor)
            for sample in samples:
                started = time.time()
                prompt = (
                    "Answer from timeline_memory only. Return exactly one minified JSON object with keys answer and confidence. "
                    "answer must be concise and no longer than 12 words; confidence must be 0 to 1. Do not modify memory.\nQuestion: "
                    + str(sample["question"])
                )
                raw = None
                parse_mode = "json"
                plain_candidate = ""
                dump: dict[str, Any] = {}
                failed_attempts: list[dict[str, Any]] = []
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
                            candidate, candidate_mode = recover_plain_answer(texts[-1])
                            if candidate:
                                plain_candidate = candidate
                                parse_mode = candidate_mode
                    failed_attempts.append({"attempt": attempt, "messages": qa_parse_diagnostics(dump)})
                if raw is None and not plain_candidate:
                    plain_prompt = (
                        "Answer the question from timeline_memory only. Do not call tools and do not modify memory. "
                        "Return only the concise answer text: no JSON, label, explanation, markdown, or tool call.\nQuestion: "
                        + str(sample["question"])
                    )
                    plain_run = await create_run(server, actor, agent.id, "timeline_qa_plain_retry", f"{sample['sample_id']}:plain")
                    plain_response = await loop.step(
                        [MessageCreate(role=MessageRole.user, content=plain_prompt)], max_steps=1, run_id=plain_run.id
                    )
                    dump = plain_response.model_dump(mode="json")
                    texts = assistant_texts(dump)
                    if texts:
                        candidate, candidate_mode = recover_plain_answer(texts[-1])
                        if candidate:
                            plain_candidate = candidate
                            parse_mode = "plain_retry_" + candidate_mode
                    failed_attempts.append({"attempt": "plain_retry", "messages": qa_parse_diagnostics(dump)})
                if raw is None and plain_candidate:
                    raw = {"answer": plain_candidate, "confidence": 0.5}
                if raw is None:
                    append_jsonl(
                        worker_dir / "qa_parse_failures.jsonl",
                        {
                            "sample_id": sample["sample_id"],
                            "dataset": sample["dataset"],
                            "conversation_id": conversation_id,
                            "attempts": failed_attempts,
                        },
                    )
                    raise RuntimeError(f"Letta returned no parseable answer for {sample['sample_id']}")
                usage = dump.get("usage") or {}
                pred = {
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "split": sample["split"],
                    "method": METHOD,
                    "answer": str(raw.get("answer") or "").strip(),
                    "confidence": float(raw.get("confidence") or 0.5),
                    "answer_parse_mode": parse_mode,
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
                    "baseline_reproduction_level": "official_core_memory_block_api_timeline_adapter",
                    "ingestion_mode": ingestion_mode,
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
        loaded = load_dialsim_samples(source, sample_n=args.sample_n, random_seed=args.random_seed)
        if args.sample_manifest:
            samples = select_frozen_samples(loaded, load_frozen_sample_ids(args.sample_manifest))
        else:
            samples = loaded
        return source, samples, len(loaded)
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
    resumed, resume_diagnostics = load_valid_resume_predictions(
        samples,
        args.resume_predictions,
        dataset=args.dataset,
        method=METHOD,
        endpoint=args.endpoint,
        allowed_endpoints=args.resume_allowed_endpoint,
        expected_ingestion_mode=args.ingestion_mode if args.ingestion_mode != "rolling_llm" else None,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if str(sample["sample_id"]) not in resumed:
            grouped[str(sample.get("conversation_id") or sample["sample_id"])].append(sample)
    groups = sorted(grouped.items())
    workers = min(args.workers, len(groups)) if groups else 0
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
        "requested_workers": args.workers,
        "endpoint": args.endpoint,
        "ingestion_mode": args.ingestion_mode,
        "extraction_algorithm_version": EXTRACTION_VERSION if args.ingestion_mode == "extractive" else None,
        "resume": resume_diagnostics,
        "adapter_note": (
            "Official SyncServer/CreateAgent/AgentLoop with deterministic query-independent extractive BlockUpdate ingestion; "
            "Evidence R@5 is not reported because core memory is not ranked retrieval."
            if args.ingestion_mode == "extractive"
            else "Official SyncServer/CreateAgent/AgentLoop with deterministic BlockUpdate ingestion and bounded model compaction; "
            "Evidence R@5 is not reported because core memory is not ranked retrieval."
        ),
        "timeline_chunk_max_chars": TIMELINE_CHUNK_MAX_CHARS,
        "core_memory_limit_chars": CORE_MEMORY_LIMIT_CHARS,
        "core_memory_target_chars": CORE_MEMORY_TARGET_CHARS,
        "compaction_max_tokens": COMPACTION_MAX_TOKENS,
    }
    write_text(run_dir / "experiment_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    results: list[dict[str, Any]] = []
    if groups:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    run_worker,
                    index,
                    args.dataset,
                    shard,
                    str(run_dir),
                    args.endpoint,
                    args.max_steps,
                    args.qa_max_steps,
                    args.max_tokens,
                    args.require_api,
                    args.ingestion_mode,
                )
                for index, shard in enumerate(shard_groups(groups, workers))
            ]
            for future in as_completed(futures):
                results.append(future.result())
    by_id: dict[str, dict[str, Any]] = dict(resumed)
    for result in results:
        for line in Path(result["predictions"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                pred = json.loads(line)
                by_id[str(pred["sample_id"])] = pred
    predictions = [by_id[str(sample["sample_id"])] for sample in samples if str(sample["sample_id"]) in by_id]
    os.environ["DEEPSEEK_API_KEY"] = "EMPTY"
    os.environ["DEEPSEEK_BASE_URL"] = args.endpoint
    os.environ["DEEPSEEK_MODEL"] = "qwen3-vl-8b"
    client, blocker = build_client_for_base_url(args.endpoint, require_api=args.require_api)
    judges = (
        judge_predictions(samples, predictions, client, require_api=args.require_api, max_workers=max(1, args.workers))
        if args.judge_answers
        else []
    )
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
            "new_prediction_count": len(predictions) - len(resumed),
        }
    )
    write_text(run_dir / "experiment_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if not validation["passed"] or manifest["fallback_count"] or (args.judge_answers and manifest["judge_missing_count"]):
        raise RuntimeError(f"Letta formal gate failed: {manifest}")
    print(run_dir)


if __name__ == "__main__":
    main()
