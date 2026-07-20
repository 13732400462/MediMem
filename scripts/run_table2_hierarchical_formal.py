from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = {
    "locomo": {
        "path": "datasets/amem_original/locomo/locomo10.official.json",
        "manifest": (
            "runs/nonmedical_timeline_formal_20260716_recovery_v2/core/"
            "locomo_benchmark_20260716_183348_550291_pid453346/"
            "sample_manifest.json"
        ),
        "seed": 20260529,
    },
    "longmemeval": {
        "path": "datasets/amem_original/longmemeval/longmemeval_s_cleaned.json",
        "manifest": (
            "runs/nonmedical_timeline_formal_20260716_final/core/"
            "longmemeval_benchmark_20260716_033920_980982_pid179090/"
            "sample_manifest.json"
        ),
        "seed": 20260529,
    },
    "dialsim": {
        "path": "datasets/amem_original/dialsim",
        "manifest": (
            "runs/nonmedical_timeline_formal_20260716_recovery_v4/core/"
            "dialsim_memory_random1000_20260716_210900_343853_pid499882/"
            "sample_manifest.json"
        ),
        "seed": 20260716,
    },
    "rhelm": {
        "path": "datasets/amem_original/rhelm/data",
        "manifest": (
            "runs/nonmedical_timeline_formal_20260717_recovery_v5/core/"
            "rhelm_benchmark_20260718_001726_592694_pid1032431/"
            "sample_manifest.json"
        ),
        "seed": 20260529,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_endpoint(endpoint: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{endpoint}/models", timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    model_ids = [str(row.get("id")) for row in payload.get("data", [])]
    if "qwen3-vl-8b" not in model_ids:
        raise RuntimeError(f"Unexpected model service at {endpoint}: {model_ids}")
    return {"endpoint": endpoint, "model_ids": model_ids}


def run_dataset(
    *,
    dataset: str,
    endpoint: str,
    project: Path,
    python: Path,
    formal_root: Path,
    embedding_model: Path,
    config: dict[str, Any],
    git_commit: str,
    max_workers: int,
) -> dict[str, Any]:
    spec = DATASETS[dataset]
    dataset_root = formal_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    before = set(dataset_root.iterdir())
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
        str(spec["seed"]),
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
    log_path = formal_root / f"{dataset}.log"
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
        "endpoint": endpoint,
        "returncode": process.returncode,
        "run_dir": str(run_dir) if run_dir else None,
        "log": str(log_path),
        "command": command,
    }
    (formal_root / f"{dataset}.result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if process.returncode or run_dir is None:
        raise RuntimeError(f"Formal dataset run failed: {record}")
    return record


def run_queue(
    datasets: list[str],
    *,
    endpoint: str,
    shared: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        run_dataset(dataset=dataset, endpoint=endpoint, **shared)
        for dataset in datasets
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/syh/A-mem-aaai"))
    parser.add_argument(
        "--development-selection",
        type=Path,
        default=Path(
            "runs/table2_hierarchical_bge_20260720/dev_qa/"
            "development_selection.json"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/table2_hierarchical_bge_20260720"),
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

    run_root = resolve(args.run_root)
    formal_root = run_root / "formal"
    formal_root.mkdir(parents=True, exist_ok=True)
    started_marker = formal_root / "FORMAL_STARTED.json"
    if started_marker.exists():
        raise RuntimeError(
            "Formal run marker already exists. Refusing an accidental second formal launch."
        )
    selection_path = resolve(args.development_selection)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config = dict(selection["selected"]["config"])
    mem0_audit_path = run_root / "mem0_audit.json"
    mem0_audit = json.loads(mem0_audit_path.read_text(encoding="utf-8"))
    if not mem0_audit.get("all_operational_gates_passed"):
        raise RuntimeError("Mem0 audit did not pass; formal shared-protocol run is blocked.")
    model_manifest_path = run_root / "services/model_artifact_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    endpoints = [
        check_endpoint("http://127.0.0.1:8001/v1"),
        check_endpoint("http://127.0.0.1:8002/v1"),
    ]
    frozen = {
        "protocol": "single frozen hierarchical_bge formal run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": args.git_commit,
        "development_selection": str(selection_path),
        "development_selection_sha256": file_sha256(selection_path),
        "mem0_audit": str(mem0_audit_path),
        "mem0_audit_sha256": file_sha256(mem0_audit_path),
        "configuration": {
            "timeline_retriever": "hierarchical_bge",
            "timeline_card_granularity": "session_chunk",
            "timeline_card_max_chars": 4000,
            "timeline_semantic_rrf_weight": 2.0,
            "timeline_embedding_window_tokens": 256,
            **config,
        },
        "qa": {
            "model": "qwen3-vl-8b",
            "max_tokens": 700,
            "seed": 20260715,
            "judge_answers": True,
            "require_api": True,
        },
        "model_artifact": {
            "path": str(model_manifest_path),
            "sha256": file_sha256(model_manifest_path),
            "artifact_sha256": model_manifest["artifact_sha256"],
        },
        "endpoints": endpoints,
        "dataset_manifests": {
            dataset: {
                "path": str(args.project / spec["manifest"]),
                "sha256": file_sha256(args.project / spec["manifest"]),
            }
            for dataset, spec in DATASETS.items()
        },
        "formal_relaunch_prohibited": True,
    }
    frozen_path = run_root / "frozen_formal_config.json"
    frozen_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    started_marker.write_text(
        json.dumps(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "frozen_config": str(frozen_path),
                "frozen_config_sha256": file_sha256(frozen_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    shared = {
        "project": args.project,
        "python": args.python,
        "formal_root": formal_root,
        "embedding_model": resolve(args.embedding_model),
        "config": config,
        "git_commit": args.git_commit,
        "max_workers": args.max_workers,
    }
    queue_specs = [
        (["locomo", "longmemeval"], "http://127.0.0.1:8001/v1"),
        (["dialsim", "rhelm"], "http://127.0.0.1:8002/v1"),
    ]
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_queue, datasets, endpoint=endpoint, shared=shared)
            for datasets, endpoint in queue_specs
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    supervisor = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": args.git_commit,
        "config": config,
        "runs": sorted(records, key=lambda row: row["dataset"]),
    }
    (formal_root / "supervisor_result.json").write_text(
        json.dumps(supervisor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (formal_root / "FORMAL_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(supervisor, ensure_ascii=False))


if __name__ == "__main__":
    main()
