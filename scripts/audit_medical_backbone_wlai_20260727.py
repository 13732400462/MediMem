from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from medimem.metrics import (
    build_leakage_audit,
    build_source_metrics,
    evaluate_predictions,
)


NEW_MODELS = {
    "llama-3.1-70b": "Llama-3.1-70B",
    "deepseek-v3": "DeepSeek-V3",
    "gpt-4.1": "GPT-4.1",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
}
METHOD_FILES = {
    "Direct": ("baselines.jsonl", "direct_deepseek"),
    "CoT": ("baselines.jsonl", "baseline_single_cot_agent"),
    "A-MEM": ("baselines.jsonl", "baseline_amem_adapter"),
    "CliCARE": ("baselines.jsonl", "official_clincare_adapter"),
    "MediMem": ("full_medimem_round_001.jsonl", None),
}
SOURCES = (
    "medmcqa",
    "medqa",
    "medical_meadow_wikidoc",
    "pmoa_tts",
    "pmc_patients",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_of(case: dict[str, Any]) -> str:
    return str(
        (case.get("data_quality_flags") or {}).get("source_dataset") or ""
    ).strip().lower()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/syh/A-mem-aaai/data/processed/table1_medical_5x500_20260720.jsonl"),
    )
    parser.add_argument(
        "--qwen30b-canonical-root",
        type=Path,
        default=Path(
            "/home/syh/A-mem-aaai/runs/table1_5models_5x500_dualgpu_fixed_20260720_232622/"
            "final_audit_20260722/canonical_predictions/30b"
        ),
    )
    parser.add_argument("--repair-model-id")
    parser.add_argument("--repair-method")
    parser.add_argument("--repair-prediction", type=Path)
    args = parser.parse_args()
    repair_args = (
        args.repair_model_id,
        args.repair_method,
        args.repair_prediction,
    )
    if any(value is not None for value in repair_args) and not all(
        value is not None for value in repair_args
    ):
        raise RuntimeError(
            "Repair overlay requires --repair-model-id, --repair-method, "
            "and --repair-prediction together"
        )
    repair_key = (
        (args.repair_model_id, args.repair_method)
        if args.repair_prediction is not None
        else None
    )
    repair_rows = (
        read_jsonl(args.repair_prediction)
        if args.repair_prediction is not None
        else []
    )
    if args.repair_prediction is not None and len(repair_rows) != 1:
        raise RuntimeError("Repair prediction must contain exactly one JSONL row")
    out = args.run_root / "audit"
    out.mkdir(parents=True, exist_ok=False)
    truth = read_jsonl(args.dataset)
    truth_ids = [str(case["case_id"]) for case in truth]
    source_counts = Counter(source_of(case) for case in truth)
    if (
        len(truth) != 2500
        or len(set(truth_ids)) != 2500
        or source_counts != Counter({source: 500 for source in SOURCES})
    ):
        raise RuntimeError(f"Frozen dataset mismatch: n={len(truth)} sources={source_counts}")
    supervisor = json.loads(
        (args.run_root / "supervisor_result.json").read_text(encoding="utf-8")
    )
    run_by_model = {row["model"]: Path(row["run_dir"]) for row in supervisor["runs"]}
    if set(run_by_model) != set(NEW_MODELS):
        raise RuntimeError(f"Model grid mismatch: {sorted(run_by_model)}")

    audit_rows = []
    metric_rows = []
    failures = []
    truth_id_set = set(truth_ids)
    for model_id, label in NEW_MODELS.items():
        run_dir = run_by_model[model_id]
        progress = read_jsonl(run_dir / "progress.jsonl")
        failed_progress = [row for row in progress if row.get("ok") is not True]
        for method, (filename, internal_method) in METHOD_FILES.items():
            path = run_dir / "predictions" / filename
            predictions = read_jsonl(path)
            if internal_method is not None:
                predictions = [
                    row for row in predictions if row.get("method") == internal_method
                ]
            raw_fallback_count = sum(
                bool(row.get("fallback_reason") or row.get("llm_error"))
                for row in predictions
            )
            applied_repairs: list[str] = []
            if repair_key == (model_id, method):
                repair = repair_rows[0]
                repair_case_id = str(repair.get("case_id") or "")
                prediction_ids = {str(row.get("case_id")) for row in predictions}
                if repair_case_id not in prediction_ids:
                    raise RuntimeError(
                        f"Repair case is absent from target cell: {repair_case_id}"
                    )
                if repair.get("fallback_reason") or repair.get("llm_error"):
                    raise RuntimeError(
                        f"Repair prediction still uses fallback: {repair_case_id}"
                    )
                predictions = [
                    repair if str(row.get("case_id")) == repair_case_id else row
                    for row in predictions
                ]
                applied_repairs.append(repair_case_id)
            by_id = {str(row.get("case_id")): row for row in predictions}
            ids_exact = (
                len(predictions) == 2500
                and len(by_id) == 2500
                and set(by_id) == truth_id_set
            )
            ordered = [by_id[case_id] for case_id in truth_ids if case_id in by_id]
            fallback_count = sum(
                bool(row.get("fallback_reason") or row.get("llm_error"))
                for row in predictions
            )
            leakage = build_leakage_audit(truth, ordered) if ids_exact else {}
            passed = (
                ids_exact
                and fallback_count == 0
                and not failed_progress
                and int(leakage.get("critical_leakage_count", 0) or 0) == 0
                and int(leakage.get("needs_review_count", 0) or 0) == 0
            )
            audit = {
                "model": label,
                "model_id": model_id,
                "method": method,
                "run_dir": str(run_dir),
                "source_file": str(path),
                "source_sha256": sha256(path),
                "predictions": len(predictions),
                "unique_predictions": len(by_id),
                "ids_exact": ids_exact,
                "fallback_count": fallback_count,
                "raw_fallback_count": raw_fallback_count,
                "repair_count": len(applied_repairs),
                "repair_case_ids": ";".join(applied_repairs),
                "repair_source_file": (
                    str(args.repair_prediction) if applied_repairs else ""
                ),
                "repair_source_sha256": (
                    sha256(args.repair_prediction) if applied_repairs else ""
                ),
                "progress_failed": len(failed_progress),
                "critical_leakage_count": int(
                    leakage.get("critical_leakage_count", 0) or 0
                ),
                "needs_review_count": int(leakage.get("needs_review_count", 0) or 0),
                "passed": passed,
            }
            audit_rows.append(audit)
            if not passed:
                failures.append(audit)
                continue
            evaluated = evaluate_predictions(truth, ordered)
            source_metric_method = (
                "full_medimem_merged" if method == "MediMem" else internal_method
            )
            source_rows = build_source_metrics(
                truth, evaluated["case_rows"], include_overall=False
            )
            for row in source_rows:
                if row.get("method") != source_metric_method:
                    continue
                metric_rows.append(
                    {
                        "model": label,
                        "model_id": model_id,
                        "method": method,
                        "source": row["source"],
                        "n": row["n"],
                        "top1": row["primary_diagnosis_top1_accuracy"],
                        "diagnosis_f1": row["diagnosis_list_f1"],
                    }
                )

    qwen_files = {
        "Direct": "direct.jsonl",
        "CoT": "cot.jsonl",
        "A-MEM": "a_mem.jsonl",
        "CliCARE": "clicare.jsonl",
        "MediMem": "medimem.jsonl",
    }
    for method, filename in qwen_files.items():
        path = args.qwen30b_canonical_root / filename
        predictions = read_jsonl(path)
        by_id = {str(row.get("case_id")): row for row in predictions}
        ids_exact = (
            len(predictions) == 2500
            and len(by_id) == 2500
            and set(by_id) == truth_id_set
        )
        ordered = [by_id[case_id] for case_id in truth_ids if case_id in by_id]
        leakage = build_leakage_audit(truth, ordered) if ids_exact else {}
        passed = (
            ids_exact
            and int(leakage.get("critical_leakage_count", 0) or 0) == 0
            and int(leakage.get("needs_review_count", 0) or 0) == 0
        )
        audit = {
            "model": "Qwen3-VL-30B",
            "model_id": "qwen3-vl-30b-fp8",
            "method": method,
            "run_dir": str(args.qwen30b_canonical_root),
            "source_file": str(path),
            "source_sha256": sha256(path),
            "predictions": len(predictions),
            "unique_predictions": len(by_id),
            "ids_exact": ids_exact,
            "fallback_count": sum(
                bool(row.get("fallback_reason") or row.get("llm_error"))
                for row in predictions
            ),
            "progress_failed": 0,
            "critical_leakage_count": int(
                leakage.get("critical_leakage_count", 0) or 0
            ),
            "needs_review_count": int(leakage.get("needs_review_count", 0) or 0),
            "passed": passed,
            "reused": True,
        }
        audit["passed"] = bool(audit["passed"] and audit["fallback_count"] == 0)
        audit_rows.append(audit)
        if not audit["passed"]:
            failures.append(audit)
            continue
        evaluated = evaluate_predictions(truth, ordered)
        for row in build_source_metrics(
            truth, evaluated["case_rows"], include_overall=False
        ):
            metric_rows.append(
                {
                    "model": "Qwen3-VL-30B",
                    "model_id": "qwen3-vl-30b-fp8",
                    "method": method,
                    "source": row["source"],
                    "n": row["n"],
                    "top1": row["primary_diagnosis_top1_accuracy"],
                    "diagnosis_f1": row["diagnosis_list_f1"],
                }
            )

    write_csv(out / "cell_audit.csv", audit_rows)
    write_csv(out / "metrics_by_source.csv", metric_rows)
    summary = {
        "protocol": "Section 4.2 five-source medical backbone audit",
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "expected_cells": 25,
        "audited_cells": len(audit_rows),
        "passed_cells": sum(bool(row["passed"]) for row in audit_rows),
        "failure_count": len(failures),
        "failures": failures,
        "repair_overlay": (
            {
                "model_id": args.repair_model_id,
                "method": args.repair_method,
                "source_file": str(args.repair_prediction),
                "source_sha256": sha256(args.repair_prediction),
                "case_ids": [str(row.get("case_id")) for row in repair_rows],
            }
            if args.repair_prediction is not None
            else None
        ),
        "passed": len(audit_rows) == 25 and not failures,
    }
    (out / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summary["passed"]:
        raise RuntimeError(f"Medical backbone audit failed: {len(failures)} cells")
    (out / "AUDIT_PASSED").write_text("passed\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

