from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mem_ehr_agent.benchmark import evidence_recall_at5


DATASETS = {
    "locomo": {
        "path": "datasets/amem_original/locomo/locomo10.official.json",
        "manifest": "runs/table2_session_cards_20260719/dev_manifest.json",
    },
    "dialsim": {
        "path": "datasets/amem_original/dialsim",
        "manifest": (
            "data/processed/table2_hierarchical_dev_20260720/"
            "dialsim_dev_manifest.json"
        ),
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_candidate(
    *,
    project: Path,
    python: Path,
    output_root: Path,
    candidate_index: int,
    config: dict[str, Any],
    endpoint: str,
    embedding_model: Path,
    git_commit: str,
    max_workers: int,
) -> dict[str, Any]:
    candidate_root = output_root / f"candidate_{candidate_index:02d}"
    candidate_root.mkdir(parents=True, exist_ok=True)
    run_records = []
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(project),
            "DEEPSEEK_BASE_URL": endpoint,
            "DEEPSEEK_API_KEY": "local-vllm",
            "DEEPSEEK_MODEL": "qwen3-vl-8b",
            "DEEPSEEK_MAX_TOKENS": "700",
            "DEEPSEEK_TIMEOUT": "180",
            "MEDICAL_LLM_SEED": "20260715",
            "MEDIMEM_GIT_COMMIT": git_commit,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    for dataset, spec in DATASETS.items():
        dataset_root = candidate_root / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        before = set(dataset_root.iterdir())
        command = [
            str(python),
            "-m",
            "mem_ehr_agent.cli",
            "benchmark",
            "run",
            "--dataset",
            dataset,
            "--dataset-path",
            str(project / spec["path"]),
            "--methods",
            "medimem",
            "--sample-manifest",
            str(project / spec["manifest"]),
            "--random-seed",
            "20260720",
            "--max-workers",
            str(max_workers),
            "--output-root",
            str(dataset_root),
            "--require-api",
            "--judge-answers",
            "--locomo-top-k",
            "5",
            "--locomo-coarse-k",
            "5",
            "--timeline-card-granularity",
            "session_chunk",
            "--timeline-card-max-chars",
            "4000",
            "--timeline-retriever",
            "hierarchical_bge",
            "--timeline-embedding-model",
            str(embedding_model),
            "--timeline-semantic-rrf-weight",
            "2.0",
            "--timeline-embedding-window-tokens",
            "256",
            "--timeline-hierarchical-parent-k",
            str(config["parent_k"]),
            "--timeline-hierarchical-bundle-k",
            str(config["bundle_k"]),
            "--timeline-hierarchical-neighbor-radius",
            str(config["neighbor_radius"]),
            "--timeline-hierarchical-bundle-max-chars",
            str(config["bundle_max_chars"]),
        ]
        log_path = candidate_root / f"{dataset}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        after = set(dataset_root.iterdir())
        new_dirs = sorted(
            (path for path in after - before if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
        )
        run_dir = new_dirs[-1] if new_dirs else None
        record = {
            "dataset": dataset,
            "returncode": process.returncode,
            "run_dir": str(run_dir) if run_dir else None,
            "log": str(log_path),
            "command": command,
        }
        run_records.append(record)
        if process.returncode or run_dir is None:
            raise RuntimeError(f"Development run failed: {record}")
    return {
        "candidate_index": candidate_index,
        "config": config,
        "endpoint": endpoint,
        "runs": run_records,
    }


def summarize_candidate(record: dict[str, Any]) -> dict[str, Any]:
    dataset_metrics = {}
    for run in record["runs"]:
        dataset = run["dataset"]
        run_dir = Path(run["run_dir"])
        predictions = read_jsonl(run_dir / "predictions" / f"{dataset}.jsonl")
        samples = read_jsonl(run_dir / "samples.jsonl")
        sample_map = {str(row["sample_id"]): row for row in samples}
        prediction_map = {str(row["sample_id"]): row for row in predictions}
        if len(predictions) != 200 or len(prediction_map) != 200:
            raise RuntimeError(f"{dataset}: incomplete development predictions")
        if any(row.get("judge_correct") is None for row in predictions):
            raise RuntimeError(f"{dataset}: missing development judge results")
        accuracies = [1.0 if row["judge_correct"] else 0.0 for row in predictions]
        recalls = [
            evidence_recall_at5(sample_map[sample_id], prediction_map[sample_id])
            for sample_id in sample_map
        ]
        recall_values = [float(value) for value in recalls if value is not None]
        prompt_tokens = [
            float((row.get("usage") or {}).get("prompt_tokens") or 0)
            for row in predictions
        ]
        dataset_metrics[dataset] = {
            "sample_count": len(predictions),
            "answer_accuracy": sum(accuracies) / len(accuracies),
            "evidence_r5": (
                sum(recall_values) / len(recall_values)
                if recall_values
                else None
            ),
            "average_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
            "fallback_count": sum(bool(row.get("fallback_reason")) for row in predictions),
            "api_error_count": sum(bool(row.get("api_error")) for row in predictions),
            "judge_missing_count": sum(
                row.get("judge_correct") is None for row in predictions
            ),
        }
    average_accuracy = sum(
        dataset_metrics[dataset]["answer_accuracy"] for dataset in DATASETS
    ) / len(DATASETS)
    average_prompt_tokens = sum(
        dataset_metrics[dataset]["average_prompt_tokens"] for dataset in DATASETS
    ) / len(DATASETS)
    return {
        **record,
        "datasets": dataset_metrics,
        "average_answer_accuracy": average_accuracy,
        "evidence_r5": dataset_metrics["locomo"]["evidence_r5"],
        "average_prompt_tokens": average_prompt_tokens,
    }


def run_candidate_queue(
    candidates: list[tuple[int, dict[str, Any]]],
    *,
    endpoint: str,
    shared: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        run_candidate(
            candidate_index=index,
            config=config,
            endpoint=endpoint,
            **shared,
        )
        for index, config in candidates
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/syh/A-mem-aaai"))
    parser.add_argument(
        "--screen",
        type=Path,
        default=Path(
            "runs/table2_hierarchical_bge_20260720/offline_screen/"
            "offline_screen.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/table2_hierarchical_bge_20260720/dev_qa"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/miniconda3/envs/amem_eval/bin/python"),
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("models/bge-small-en-v1.5"),
    )
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else args.project / path

    screen = json.loads(resolve(args.screen).read_text(encoding="utf-8"))
    candidates = list(screen["selected"])
    if len(candidates) != 3:
        raise RuntimeError(f"Expected exactly three screened candidates, got {len(candidates)}")
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    endpoints = ["http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"]
    records = []
    indexed_candidates = list(enumerate(candidates, start=1))
    queues = [indexed_candidates[0::2], indexed_candidates[1::2]]
    shared = {
        "project": args.project,
        "python": args.python,
        "output_root": output_root,
        "embedding_model": resolve(args.embedding_model),
        "git_commit": args.git_commit,
        "max_workers": args.max_workers,
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                run_candidate_queue,
                queue,
                endpoint=endpoint,
                shared=shared,
            )
            for queue, endpoint in zip(queues, endpoints)
            if queue
        ]
        for future in as_completed(futures):
            records.extend(future.result())

    summaries = [summarize_candidate(record) for record in records]
    ranked = sorted(
        summaries,
        key=lambda row: (
            -float(row["average_answer_accuracy"]),
            -float(row["evidence_r5"] or 0.0),
            float(row["average_prompt_tokens"]),
            int(row["config"]["parent_k"]),
            int(row["config"]["bundle_k"]),
        ),
    )
    result = {
        "protocol": "frozen hierarchical_bge development QA selection",
        "test_sets_used": False,
        "selection_order": [
            "cross-development-set mean Answer Accuracy",
            "LoCoMo development Evidence R@5",
            "lower average prompt tokens",
            "smaller retrieval budget",
        ],
        "git_commit": args.git_commit,
        "selected": ranked[0],
        "ranked_candidates": ranked,
    }
    (output_root / "development_selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
