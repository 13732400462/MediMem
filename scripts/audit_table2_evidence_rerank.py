from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mem_ehr_agent.benchmark import (
    evidence_recall_at5,
    normalize_answer,
)
from mem_ehr_agent.metrics import token_f1


EXPECTED = {
    "locomo": 1000,
    "longmemeval": 500,
    "dialsim": 1000,
    "rhelm": 1305,
}
EVIDENCE_DATASETS = ("locomo", "longmemeval", "rhelm")
TABLE_COLUMNS = (
    "locomo_answer_acc",
    "locomo_evidence_r5",
    "longmemeval_answer_acc",
    "longmemeval_evidence_r5",
    "dialsim_answer_acc",
    "dialsim_qa_f1",
    "rhelm_answer_acc",
    "rhelm_evidence_r5",
    "average_answer_acc",
    "average_evidence_r5",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_values(
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, float | None]:
    sample_map = {str(row["sample_id"]): row for row in samples}
    prediction_map = {str(row["sample_id"]): row for row in predictions}
    accuracies = [
        1.0 if bool(prediction_map[sample_id].get("judge_correct")) else 0.0
        for sample_id in sample_map
    ]
    recalls = [
        evidence_recall_at5(sample, prediction_map[sample_id])
        for sample_id, sample in sample_map.items()
    ]
    recall_values = [float(value) for value in recalls if value is not None]
    qa_f1_values = [
        token_f1(
            normalize_answer(prediction_map[sample_id].get("answer")),
            normalize_answer(sample.get("answer")),
        )
        for sample_id, sample in sample_map.items()
    ]
    return {
        "answer_accuracy": sum(accuracies) / len(accuracies),
        "evidence_r5": (
            sum(recall_values) / len(recall_values)
            if recall_values
            else None
        ),
        "qa_f1": sum(qa_f1_values) / len(qa_f1_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/syh/A-mem-aaai"))
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/table2_evidence_rerank_20260723"),
    )
    parser.add_argument(
        "--current-table",
        type=Path,
        default=Path("mem_ehr_agent/outputs/nonmedical_timeline_main_table_20260719.csv"),
    )
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else args.project / path

    run_root = resolve(args.run_root)
    formal_root = run_root / "formal"
    supervisor = json.loads(
        (formal_root / "supervisor_result.json").read_text(encoding="utf-8")
    )
    run_dirs = {
        row["dataset"]: Path(row["run_dir"])
        for row in supervisor["runs"]
    }
    if set(run_dirs) != set(EXPECTED):
        raise RuntimeError(f"Formal supervisor is incomplete: {sorted(run_dirs)}")

    dataset_metrics: dict[str, dict[str, float | None]] = {}
    dataset_audits = {}
    for dataset, expected in EXPECTED.items():
        run_dir = run_dirs[dataset]
        samples = read_jsonl(run_dir / "samples.jsonl")
        predictions = read_jsonl(run_dir / "predictions" / f"{dataset}.jsonl")
        judges = read_jsonl(run_dir / "judge_results.jsonl")
        blocked = read_jsonl(run_dir / "blocked_methods.jsonl")
        manifest = json.loads(
            (run_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
        )
        sample_ids = [str(row["sample_id"]) for row in samples]
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
        needs_review_count = sum(bool(row.get("needs_review")) for row in predictions)
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
        config_ok = (
            manifest.get("git_commit") == args.git_commit
            and manifest.get("timeline_retriever") == "evidence_rerank"
            and manifest.get("timeline_card_granularity") == "session_chunk"
        )
        passed = (
            ids_exact
            and not blocked
            and fallback_count == 0
            and api_error_count == 0
            and worker_failure_count == 0
            and judge_missing_count == 0
            and leakage_count == 0
            and needs_review_count == 0
            and bool((manifest.get("validation") or {}).get("passed"))
            and bool((manifest.get("guard") or {}).get("passed"))
            and config_ok
        )
        dataset_metrics[dataset] = metric_values(samples, predictions)
        dataset_audits[dataset] = {
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
            "needs_review_count": needs_review_count,
            "validation_passed": bool((manifest.get("validation") or {}).get("passed")),
            "guard_passed": bool((manifest.get("guard") or {}).get("passed")),
            "config_ok": config_ok,
            "passed": passed,
        }

    candidate = {
        "locomo_answer_acc": dataset_metrics["locomo"]["answer_accuracy"],
        "locomo_evidence_r5": dataset_metrics["locomo"]["evidence_r5"],
        "longmemeval_answer_acc": dataset_metrics["longmemeval"]["answer_accuracy"],
        "longmemeval_evidence_r5": dataset_metrics["longmemeval"]["evidence_r5"],
        "dialsim_answer_acc": dataset_metrics["dialsim"]["answer_accuracy"],
        "dialsim_qa_f1": dataset_metrics["dialsim"]["qa_f1"],
        "rhelm_answer_acc": dataset_metrics["rhelm"]["answer_accuracy"],
        "rhelm_evidence_r5": dataset_metrics["rhelm"]["evidence_r5"],
        "average_answer_acc": sum(
            float(dataset_metrics[dataset]["answer_accuracy"])
            for dataset in EXPECTED
        )
        / len(EXPECTED),
        "average_evidence_r5": sum(
            float(dataset_metrics[dataset]["evidence_r5"])
            for dataset in EVIDENCE_DATASETS
        )
        / len(EVIDENCE_DATASETS),
    }
    table = read_table(resolve(args.current_table))
    current_row = next(row for row in table if row["method"] == "MediMem")
    baseline_rows = [row for row in table if row["method"] != "MediMem"]
    cells = {}
    for column in TABLE_COLUMNS:
        comparable_baselines = [
            float(row[column])
            for row in baseline_rows
            if str(row.get(column) or "").strip()
        ]
        old_value = float(current_row[column])
        new_value = float(candidate[column])
        baseline_best = max(comparable_baselines)
        old_gap = max(0.0, baseline_best - old_value)
        new_gap = max(0.0, baseline_best - new_value)
        cells[column] = {
            "baseline_best": baseline_best,
            "old_medimem": old_value,
            "new_medimem": new_value,
            "old_gap": old_gap,
            "new_gap": new_gap,
            "gap_change": new_gap - old_gap,
        }
    old_overall_gap = sum(row["old_gap"] for row in cells.values()) / len(cells)
    new_overall_gap = sum(row["new_gap"] for row in cells.values()) / len(cells)
    all_operational_gates = all(row["passed"] for row in dataset_audits.values())
    adopted = all_operational_gates and new_overall_gap < old_overall_gap
    report = {
        "protocol": "Table 2 evidence_rerank relative-gap formal audit",
        "git_commit": args.git_commit,
        "all_operational_gates_passed": all_operational_gates,
        "dataset_metrics": dataset_metrics,
        "candidate_table_row": candidate,
        "relative_gap": {
            "columns": cells,
            "old_overall_gap": old_overall_gap,
            "new_overall_gap": new_overall_gap,
            "strictly_smaller": new_overall_gap < old_overall_gap,
            "column_count": len(cells),
            "unweighted": True,
        },
        "adopted_for_paper": adopted,
        "paper_action": (
            "update_existing_table2_and_analysis"
            if adopted
            else "archive_without_paper_change"
        ),
        "no_dataset_level_splicing": True,
        "datasets": dataset_audits,
    }
    (run_root / "formal_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_root / ("ADOPTED" if adopted else "NOT_ADOPTED")).write_text(
        f"{report['paper_action']}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
