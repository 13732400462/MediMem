from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {
    "locomo": 1000,
    "longmemeval": 500,
    "dialsim": 1000,
    "rhelm": 1305,
}

CORE_RUNS = {
    "locomo": (
        "runs/nonmedical_timeline_formal_20260716_recovery_v2/core/"
        "locomo_benchmark_20260716_183348_550291_pid453346"
    ),
    "longmemeval": (
        "runs/nonmedical_timeline_formal_20260716_final/core/"
        "longmemeval_benchmark_20260716_033920_980982_pid179090"
    ),
    "dialsim": (
        "runs/nonmedical_timeline_formal_20260716_recovery_v4/core/"
        "dialsim_memory_random1000_20260716_210900_343853_pid499882"
    ),
    "rhelm": (
        "runs/nonmedical_timeline_formal_20260717_recovery_v5/core/"
        "rhelm_benchmark_20260718_001726_592694_pid1032431"
    ),
}

MEM0_RUNS = {
    "locomo": (
        "runs/nonmedical_timeline_formal_20260716_recovery_v4/mem0/"
        "mem0_locomo_20260716_234932_787183_pid550030"
    ),
    "longmemeval": (
        "runs/nonmedical_timeline_formal_20260716_final/mem0/"
        "mem0_longmemeval_20260716_123140_683891_pid342335"
    ),
    "dialsim": (
        "runs/nonmedical_timeline_formal_20260716_recovery_v4/mem0/"
        "mem0_dialsim_20260717_000349_469801_pid554458"
    ),
    "rhelm": (
        "runs/nonmedical_timeline_formal_20260716_final/mem0/"
        "mem0_rhelm_20260716_152926_826927_pid394306"
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_runtime_input(sample: dict[str, Any]) -> dict[str, Any]:
    """Only fields available to the runtime retriever/answerer, never gold fields."""
    return {
        "sample_id": sample.get("sample_id"),
        "conversation_id": sample.get("conversation_id"),
        "dataset": sample.get("dataset"),
        "split": sample.get("split"),
        "context": sample.get("context"),
        "turns": sample.get("turns"),
        "question": sample.get("question"),
    }


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_model_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model",
        "model_name",
        "served_model",
        "endpoint",
        "base_url",
        "judge_model",
        "judge_endpoint",
    )
    return {key: manifest.get(key) for key in keys if manifest.get(key) is not None}


def audit_dataset(project: Path, dataset: str) -> dict[str, Any]:
    expected = EXPECTED_COUNTS[dataset]
    core_dir = project / CORE_RUNS[dataset]
    mem0_dir = project / MEM0_RUNS[dataset]
    predictions = read_jsonl(mem0_dir / "predictions" / f"mem0_{dataset}.jsonl")
    judges = read_jsonl(mem0_dir / "judge_results.jsonl")
    core_manifest = json.loads(
        (core_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    mem0_manifest = json.loads(
        (mem0_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )

    core_hashes: dict[str, str] = {}
    core_count = 0
    for row in iter_jsonl(core_dir / "samples.jsonl"):
        sample_id = str(row["sample_id"])
        core_hashes[sample_id] = canonical_hash(canonical_runtime_input(row))
        core_count += 1
    mem0_ids: set[str] = set()
    mem0_count = 0
    input_hashes_match = True
    for row in iter_jsonl(mem0_dir / "samples.jsonl"):
        sample_id = str(row["sample_id"])
        mem0_ids.add(sample_id)
        mem0_count += 1
        if core_hashes.get(sample_id) != canonical_hash(canonical_runtime_input(row)):
            input_hashes_match = False
    prediction_ids = [str(row["sample_id"]) for row in predictions]
    judge_ids = [str(row["sample_id"]) for row in judges]
    id_sets_match = (
        set(core_hashes) == mem0_ids == set(prediction_ids) == set(judge_ids)
    )
    input_hashes_match = id_sets_match and input_hashes_match

    fallback_count = sum(bool(row.get("fallback_reason")) for row in predictions)
    api_error_count = sum(bool(row.get("api_error")) for row in predictions)
    worker_failure_count = sum(bool(row.get("worker_failure")) for row in predictions)
    judge_missing_count = sum(row.get("judge_correct") is None for row in predictions)
    guard_failure_count = sum(
        not bool(row.get("guard_passed", True)) for row in predictions
    )
    leakage_count = sum(
        bool(row.get("leakage") or row.get("leakage_detected")) for row in predictions
    )
    method_identity_ok = all(
        row.get("method") == "official_mem0_timeline_adapter"
        and row.get("official_baseline") == "mem0"
        for row in predictions
    )
    endpoint_values = sorted(
        {str(row.get("endpoint")) for row in predictions if row.get("endpoint")}
    )
    expanded_ref_counts = [
        len(row.get("retrieved_evidence_refs_at5") or []) for row in predictions
    ]
    operational_passed = (
        core_count == expected
        and mem0_count == expected
        and len(predictions) == expected
        and len(set(prediction_ids)) == expected
        and len(judges) == expected
        and len(set(judge_ids)) == expected
        and id_sets_match
        and input_hashes_match
        and fallback_count == 0
        and api_error_count == 0
        and worker_failure_count == 0
        and judge_missing_count == 0
        and guard_failure_count == 0
        and leakage_count == 0
        and method_identity_ok
    )
    return {
        "expected": expected,
        "core_run": str(core_dir),
        "mem0_run": str(mem0_dir),
        "core_samples": core_count,
        "mem0_samples": mem0_count,
        "predictions": len(predictions),
        "unique_predictions": len(set(prediction_ids)),
        "judges": len(judges),
        "unique_judges": len(set(judge_ids)),
        "sample_ids_exact": id_sets_match,
        "runtime_input_hashes_exact": input_hashes_match,
        "runtime_input_set_sha256": canonical_hash(
            sorted(core_hashes.items())
        ),
        "fallback_count": fallback_count,
        "api_error_count": api_error_count,
        "worker_failure_count": worker_failure_count,
        "judge_missing_count": judge_missing_count,
        "guard_failure_count": guard_failure_count,
        "leakage_count": leakage_count,
        "method_identity_ok": method_identity_ok,
        "prediction_endpoints": endpoint_values,
        "core_model_identity": manifest_model_identity(core_manifest),
        "mem0_model_identity": manifest_model_identity(mem0_manifest),
        "retrieved_memory_count_values": sorted(
            {int(row.get("retrieved_memory_count") or 0) for row in predictions}
        ),
        "expanded_refs_at5": {
            "min": min(expanded_ref_counts) if expanded_ref_counts else 0,
            "max": max(expanded_ref_counts) if expanded_ref_counts else 0,
            "mean": (
                sum(expanded_ref_counts) / len(expanded_ref_counts)
                if expanded_ref_counts
                else 0
            ),
            "interpretation": (
                "R@5 is computed from the source refs attached to five native Mem0 "
                "memory results; the ref list may contain more than five source turns."
            ),
        },
        "operational_passed": operational_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/syh/A-mem-aaai"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/table2_hierarchical_bge_20260720/mem0_audit.json"),
    )
    args = parser.parse_args()

    source = args.project / "scripts/run_official_mem0_timeline.py"
    source_text = source.read_text(encoding="utf-8")
    static_protocol = {
        "source": str(source),
        "uses_official_memory_api": all(
            marker in source_text
            for marker in ("Memory.from_config", ".add(", ".search(")
        ),
        "memory_add_infer_false": "infer=False" in source_text,
        "uses_shared_answer_function": "answer_with_context(" in source_text,
        "uses_shared_judge_evaluator": "judge_predictions(" in source_text,
        "qwen3_vl_8b_configured": '"model": "qwen3-vl-8b"' in source_text,
        "gold_answer_passed_to_memory": 'sample["answer"]' in source_text[
            source_text.find("def run_official_mem0") :
        ],
        "gold_evidence_passed_to_memory": "sample.get(\"evidence\"" in source_text[
            source_text.find("def run_official_mem0") :
        ],
        "review_note": (
            "The two gold-field booleans are conservative source-string flags and "
            "must be interpreted with the recorded call-site excerpt."
        ),
    }
    datasets = {
        dataset: audit_dataset(args.project, dataset)
        for dataset in EXPECTED_COUNTS
    }
    all_operational_passed = all(
        item["operational_passed"] for item in datasets.values()
    )
    static_protocol_passed = (
        static_protocol["uses_official_memory_api"]
        and static_protocol["memory_add_infer_false"]
        and static_protocol["uses_shared_answer_function"]
        and static_protocol["uses_shared_judge_evaluator"]
        and static_protocol["qwen3_vl_8b_configured"]
        and not static_protocol["gold_answer_passed_to_memory"]
        and not static_protocol["gold_evidence_passed_to_memory"]
    )
    same_endpoints = sorted(
        {
            endpoint
            for item in datasets.values()
            for endpoint in item["prediction_endpoints"]
        }
    )
    report = {
        "protocol": "frozen Table 2 Mem0 comparability audit",
        "all_operational_gates_passed": (
            all_operational_passed and static_protocol_passed
        ),
        "sample_total": sum(item["expected"] for item in datasets.values()),
        "prediction_endpoints": same_endpoints,
        "static_protocol": static_protocol,
        "datasets": datasets,
        "verdict": (
            "retain_mem0_in_shared_table"
            if all_operational_passed and static_protocol_passed
            else "separate_native_memory_protocol_pending_manual_review"
        ),
        "disclosure": (
            "Mem0 uses its official native memory API through a timeline adapter. "
            "Its top-five evidence score expands the source refs belonging to the "
            "five returned native memory objects; this is disclosed rather than "
            "treated as a reason to remove a leading baseline."
        ),
    }
    output = args.project / args.output if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
