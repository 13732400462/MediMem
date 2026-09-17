from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from medimem.benchmark import evidence_recall_at5


EXPECTED = {
    "locomo": 1000,
    "longmemeval": 500,
    "dialsim": 1000,
    "rhelm": 1305,
}
EVIDENCE_DATASETS = ("locomo", "longmemeval", "rhelm")
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
HYBRID_RUNS = {
    "locomo": (
        "runs/table2_hybrid_bge_20260719/formal/locomo/"
        "locomo_benchmark_20260719_225729_692170_pid2052962"
    ),
    "longmemeval": (
        "runs/table2_hybrid_bge_20260719/formal/longmemeval/"
        "longmemeval_benchmark_20260719_230712_403832_pid2057955"
    ),
    "dialsim": (
        "runs/table2_hybrid_bge_20260719/formal/dialsim/"
        "dialsim_memory_random1000_20260719_230045_990946_pid2055359"
    ),
    "rhelm": (
        "runs/table2_hybrid_bge_20260719/formal/rhelm/"
        "rhelm_benchmark_20260720_020608_031520_pid2125837"
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_sample_evidence(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ids = []
    samples = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            ids.append(sample_id)
            samples[sample_id] = {
                "sample_id": sample_id,
                "evidence": row.get("evidence") or [],
            }
    return ids, samples


def prediction_path(run_dir: Path, dataset: str, method: str) -> Path:
    name = f"mem0_{dataset}.jsonl" if method == "mem0" else f"{dataset}.jsonl"
    return run_dir / "predictions" / name


def metric_values(
    sample_map: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float | None]], dict[str, Any]]:
    prediction_map = {str(row["sample_id"]): row for row in predictions}
    values = {}
    for sample_id, sample in sample_map.items():
        pred = prediction_map[sample_id]
        values[sample_id] = {
            "accuracy": 1.0 if bool(pred.get("judge_correct")) else 0.0,
            "r5": evidence_recall_at5(sample, pred),
        }
    accuracies = [float(row["accuracy"]) for row in values.values()]
    recalls = [
        float(row["r5"]) for row in values.values() if row["r5"] is not None
    ]
    prompt_tokens = [
        float((row.get("usage") or {}).get("prompt_tokens") or 0)
        for row in predictions
    ]
    return values, {
        "answer_accuracy": sum(accuracies) / len(accuracies),
        "evidence_r5": sum(recalls) / len(recalls) if recalls else None,
        "evidence_n": len(recalls),
        "average_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
    }


def aggregate(dataset_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        "average_answer_accuracy": sum(
            dataset_metrics[dataset]["answer_accuracy"] for dataset in EXPECTED
        )
        / len(EXPECTED),
        "average_evidence_r5": sum(
            dataset_metrics[dataset]["evidence_r5"]
            for dataset in EVIDENCE_DATASETS
        )
        / len(EVIDENCE_DATASETS),
        "average_prompt_tokens": sum(
            dataset_metrics[dataset]["average_prompt_tokens"] for dataset in EXPECTED
        )
        / len(EXPECTED),
    }


def bootstrap_average(
    current: dict[str, dict[str, dict[str, float | None]]],
    baseline: dict[str, dict[str, dict[str, float | None]]],
    *,
    metric: str,
    datasets: tuple[str, ...],
    seed: int,
    iterations: int = 10000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sampled_sources = []
    observed_sources = []
    pairing = {}
    for dataset in datasets:
        ids = sorted(current[dataset])
        pairs = [
            (current[dataset][sample_id][metric], baseline[dataset][sample_id][metric])
            for sample_id in ids
        ]
        pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
        delta = np.asarray(
            [float(left) - float(right) for left, right in pairs],
            dtype=np.float64,
        )
        observed_sources.append(float(delta.mean()))
        samples = np.empty(iterations, dtype=np.float64)
        offset = 0
        while offset < iterations:
            size = min(250, iterations - offset)
            indices = rng.integers(0, len(delta), size=(size, len(delta)))
            samples[offset : offset + size] = delta[indices].mean(axis=1)
            offset += size
        sampled_sources.append(samples)
        pairing[dataset] = {"n": len(ids), "metric_n": len(delta)}
    distribution = np.vstack(sampled_sources).mean(axis=0)
    observed = sum(observed_sources) / len(observed_sources)
    return {
        "mean_difference": observed,
        "ci95": [
            float(np.percentile(distribution, 2.5, method="linear")),
            float(np.percentile(distribution, 97.5, method="linear")),
        ],
        "probability_new_greater": float(np.mean(distribution > 0)),
        "pairing": pairing,
        "seed": seed,
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/syh/A-mem-aaai"))
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/table2_hierarchical_bge_20260720"),
    )
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    run_root = (
        args.run_root if args.run_root.is_absolute() else args.project / args.run_root
    )
    formal_root = run_root / "formal"
    supervisor = json.loads(
        (formal_root / "supervisor_result.json").read_text(encoding="utf-8")
    )
    run_dirs = {
        row["dataset"]: Path(row["run_dir"]) for row in supervisor["runs"]
    }
    if set(run_dirs) != set(EXPECTED):
        raise RuntimeError(f"Formal supervisor is incomplete: {sorted(run_dirs)}")

    sample_maps = {}
    current_values = {}
    baseline_values = {}
    hybrid_values = {}
    current_metrics = {}
    baseline_metrics = {}
    hybrid_metrics = {}
    audits = {}
    for dataset, expected in EXPECTED.items():
        run_dir = run_dirs[dataset]
        sample_ids, sample_map = read_sample_evidence(run_dir / "samples.jsonl")
        sample_maps[dataset] = sample_map
        predictions = read_jsonl(prediction_path(run_dir, dataset, "medimem"))
        judges = read_jsonl(run_dir / "judge_results.jsonl")
        blocked = read_jsonl(run_dir / "blocked_methods.jsonl")
        manifest = json.loads(
            (run_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
        )
        prediction_ids = [str(row["sample_id"]) for row in predictions]
        judge_ids = [str(row["sample_id"]) for row in judges]
        fallback_count = sum(bool(row.get("fallback_reason")) for row in predictions)
        api_error_count = sum(bool(row.get("api_error")) for row in predictions)
        worker_failure_count = sum(bool(row.get("worker_failure")) for row in predictions)
        judge_missing_count = sum(row.get("judge_correct") is None for row in predictions)
        leakage_count = sum(
            bool(row.get("leakage") or row.get("leakage_detected"))
            for row in predictions
        )
        config_ok = (
            manifest.get("git_commit") == args.git_commit
            and manifest.get("timeline_retriever") == "hierarchical_bge"
            and manifest.get("timeline_card_granularity") == "session_chunk"
        )
        ids_exact = (
            len(sample_ids) == expected
            and len(set(sample_ids)) == expected
            and len(prediction_ids) == expected
            and len(set(prediction_ids)) == expected
            and set(prediction_ids) == set(sample_ids)
            and len(judge_ids) == expected
            and len(set(judge_ids)) == expected
            and set(judge_ids) == set(sample_ids)
        )
        passed = (
            ids_exact
            and not blocked
            and fallback_count == 0
            and api_error_count == 0
            and worker_failure_count == 0
            and judge_missing_count == 0
            and leakage_count == 0
            and bool((manifest.get("validation") or {}).get("passed"))
            and bool((manifest.get("guard") or {}).get("passed"))
            and config_ok
        )
        current_values[dataset], current_metrics[dataset] = metric_values(
            sample_map, predictions
        )
        mem0_predictions = read_jsonl(
            prediction_path(args.project / MEM0_RUNS[dataset], dataset, "mem0")
        )
        baseline_values[dataset], baseline_metrics[dataset] = metric_values(
            sample_map, mem0_predictions
        )
        hybrid_predictions = read_jsonl(
            prediction_path(args.project / HYBRID_RUNS[dataset], dataset, "medimem")
        )
        hybrid_values[dataset], hybrid_metrics[dataset] = metric_values(
            sample_map, hybrid_predictions
        )
        audits[dataset] = {
            "expected": expected,
            "run_dir": str(run_dir),
            "predictions": len(predictions),
            "unique_predictions": len(set(prediction_ids)),
            "judges": len(judges),
            "unique_judges": len(set(judge_ids)),
            "sample_ids_exact": ids_exact,
            "blocked_count": len(blocked),
            "fallback_count": fallback_count,
            "api_error_count": api_error_count,
            "worker_failure_count": worker_failure_count,
            "judge_missing_count": judge_missing_count,
            "leakage_count": leakage_count,
            "validation_passed": bool(
                (manifest.get("validation") or {}).get("passed")
            ),
            "guard_passed": bool((manifest.get("guard") or {}).get("passed")),
            "config_ok": config_ok,
            "passed": passed,
        }

    current_aggregate = aggregate(current_metrics)
    baseline_aggregate = aggregate(baseline_metrics)
    hybrid_aggregate = aggregate(hybrid_metrics)
    accuracy_bootstrap = bootstrap_average(
        current_values,
        baseline_values,
        metric="accuracy",
        datasets=tuple(EXPECTED),
        seed=20260720,
    )
    evidence_bootstrap = bootstrap_average(
        current_values,
        baseline_values,
        metric="r5",
        datasets=EVIDENCE_DATASETS,
        seed=20260721,
    )

    cells = {}
    for dataset in EXPECTED:
        cells[f"{dataset}.answer_accuracy"] = {
            "medimem": current_metrics[dataset]["answer_accuracy"],
            "mem0": baseline_metrics[dataset]["answer_accuracy"],
        }
    for dataset in EVIDENCE_DATASETS:
        cells[f"{dataset}.evidence_r5"] = {
            "medimem": current_metrics[dataset]["evidence_r5"],
            "mem0": baseline_metrics[dataset]["evidence_r5"],
        }
    cells["average.answer_accuracy"] = {
        "medimem": current_aggregate["average_answer_accuracy"],
        "mem0": baseline_aggregate["average_answer_accuracy"],
    }
    cells["average.evidence_r5"] = {
        "medimem": current_aggregate["average_evidence_r5"],
        "mem0": baseline_aggregate["average_evidence_r5"],
    }
    not_lower_count = 0
    for cell in cells.values():
        cell["not_lower"] = bool(float(cell["medimem"]) >= float(cell["mem0"]))
        not_lower_count += int(cell["not_lower"])

    beats_both_averages = (
        current_aggregate["average_answer_accuracy"]
        > baseline_aggregate["average_answer_accuracy"]
        and current_aggregate["average_evidence_r5"]
        > baseline_aggregate["average_evidence_r5"]
    )
    noninferior_route = (
        not_lower_count >= 6
        and accuracy_bootstrap["ci95"][0] > -0.01
        and evidence_bootstrap["ci95"][0] > -0.01
    )
    all_operational_gates = all(row["passed"] for row in audits.values())
    adopted = all_operational_gates and (beats_both_averages or noninferior_route)

    run_candidates = [
        {
            "name": "hierarchical_bge",
            "aggregate": current_aggregate,
            "dataset_metrics": current_metrics,
            "run_dirs": {key: str(value) for key, value in run_dirs.items()},
        },
        {
            "name": "hybrid_bge",
            "aggregate": hybrid_aggregate,
            "dataset_metrics": hybrid_metrics,
            "run_dirs": {
                key: str(args.project / value) for key, value in HYBRID_RUNS.items()
            },
        },
    ]
    best_whole_run = sorted(
        run_candidates,
        key=lambda row: (
            -row["aggregate"]["average_answer_accuracy"],
            -row["aggregate"]["average_evidence_r5"],
            row["aggregate"]["average_prompt_tokens"],
        ),
    )[0]
    mem0_audit = json.loads((run_root / "mem0_audit.json").read_text(encoding="utf-8"))
    report = {
        "protocol": "Table 2 hierarchical_bge formal acceptance audit",
        "git_commit": args.git_commit,
        "all_operational_gates_passed": all_operational_gates,
        "mem0_audit_verdict": mem0_audit["verdict"],
        "mem0_retained": mem0_audit["verdict"] == "retain_mem0_in_shared_table",
        "current": {
            "datasets": current_metrics,
            "aggregate": current_aggregate,
        },
        "mem0": {
            "datasets": baseline_metrics,
            "aggregate": baseline_aggregate,
        },
        "hybrid_bge_fallback": {
            "datasets": hybrid_metrics,
            "aggregate": hybrid_aggregate,
        },
        "core_quality_cells": cells,
        "not_lower_cell_count": not_lower_count,
        "paired_bootstrap": {
            "answer_accuracy_average": accuracy_bootstrap,
            "evidence_r5_average": evidence_bootstrap,
        },
        "acceptance": {
            "beats_both_averages": beats_both_averages,
            "six_of_nine_and_noninferior": noninferior_route,
            "adopted_for_main_text": adopted,
        },
        "best_complete_medimem_run": best_whole_run,
        "paper_action": (
            "update_main_text_table2"
            if adopted
            else "move_full_table2_to_appendix_and_report_generalization_boundary"
        ),
        "no_dataset_level_splicing": True,
        "datasets": audits,
    }
    (run_root / "formal_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    marker = "ADOPTED" if adopted else "NOT_ADOPTED"
    (run_root / marker).write_text(
        f"{report['paper_action']}\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

