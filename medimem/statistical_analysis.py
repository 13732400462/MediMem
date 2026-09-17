from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_jsonl
from .metrics import evaluate_predictions


STRATUM_BY_SOURCE = {
    "pmoa_tts": "core_longitudinal",
    "pmc_patients": "core_longitudinal",
    "medmcqa": "medical_qa_transfer",
    "medqa": "medical_qa_transfer",
    "medical_meadow_wikidoc": "knowledge_transfer",
    "chatdoctor_healthcaremagic": "dialogue_stress",
}


def canonical_method(method: str) -> str:
    method = str(method)
    if method.startswith("full_medimem_"):
        return "full_medimem"
    if "_medimem_" in method and method.startswith("ablate_"):
        return method.split("_medimem_", 1)[0]
    return method


def source_name(case: dict[str, Any]) -> str:
    flags = case.get("data_quality_flags") or {}
    source = str(flags.get("source_dataset") or flags.get("source") or "").strip().lower()
    if source:
        return source
    case_id = str(case.get("case_id") or "").lower()
    return next((name for name in STRATUM_BY_SOURCE if case_id.startswith(name)), "unknown")


def case_stratum(case: dict[str, Any]) -> str:
    return STRATUM_BY_SOURCE.get(source_name(case), "other")


def load_predictions(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        if path.is_dir():
            run_dir = path.parent if path.name == "predictions" else path
        else:
            run_dir = path.parent.parent if path.parent.name == "predictions" else path.parent
        manifest_path = run_dir / "experiment_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            budget = manifest.get("completion_token_budget")
            replicate_id = f"seed={manifest.get('run_seed') or run_dir.name};budget={budget or 'unbounded'}"
        else:
            budget = None
            replicate_id = run_dir.name
        for file in files:
            for pred in read_jsonl(file):
                normalized = dict(pred)
                normalized["method"] = canonical_method(str(pred.get("method") or ""))
                normalized["replicate_id"] = replicate_id
                normalized["completion_token_budget"] = budget
                predictions.append(normalized)
    # One prediction per method/case is required for paired statistics.
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pred in predictions:
        key = (str(pred.get("method")), str(pred.get("case_id")), str(pred.get("replicate_id")))
        if key in unique:
            raise ValueError(f"Duplicate prediction for method/case: {key}")
        unique[key] = pred
    return list(unique.values())


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else math.nan


def pdo(row: dict[str, Any], weight_top1: float) -> float | None:
    top1 = row.get("primary_correct")
    diagnosis_f1 = row.get("diagnosis_f1")
    if top1 is None or diagnosis_f1 is None:
        return None
    return weight_top1 * float(top1) + (1.0 - weight_top1) * float(diagnosis_f1)


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    if metric.startswith("pdo_"):
        return pdo(row, float(metric.split("_", 1)[1]))
    value = row.get(metric)
    return None if value is None else float(value)


def paired_values(
    rows: list[dict[str, Any]], target: str, baseline: str, metric: str
) -> tuple[list[float], list[float]]:
    by_key = {
        (str(row["method"]), str(row["case_id"]), str(row.get("replicate_id") or "default")): row
        for row in rows
    }
    sample_keys = sorted(
        {(case_id, replicate) for method, case_id, replicate in by_key if method == target}
        & {(case_id, replicate) for method, case_id, replicate in by_key if method == baseline}
    )
    target_values: list[float] = []
    baseline_values: list[float] = []
    for case_id, replicate in sample_keys:
        target_value = metric_value(by_key[(target, case_id, replicate)], metric)
        baseline_value = metric_value(by_key[(baseline, case_id, replicate)], metric)
        if target_value is None or baseline_value is None:
            continue
        target_values.append(target_value)
        baseline_values.append(baseline_value)
    return target_values, baseline_values


def paired_bootstrap_ci(
    target: list[float], baseline: list[float], *, resamples: int, seed: int
) -> tuple[float, float, float]:
    if len(target) != len(baseline) or not target:
        return math.nan, math.nan, math.nan
    differences = [a - b for a, b in zip(target, baseline)]
    observed = mean(differences)
    rng = random.Random(seed)
    boot = [mean(differences[rng.randrange(len(differences))] for _ in differences) for _ in range(resamples)]
    return observed, percentile(boot, 0.025), percentile(boot, 0.975)


def paired_permutation_p(target: list[float], baseline: list[float], *, resamples: int, seed: int) -> float:
    if len(target) != len(baseline) or not target:
        return math.nan
    differences = [a - b for a, b in zip(target, baseline)]
    observed = abs(mean(differences))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(resamples):
        permuted = abs(mean(value if rng.random() < 0.5 else -value for value in differences))
        exceed += permuted >= observed
    return (exceed + 1) / (resamples + 1)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    valid = sorted((p, idx) for idx, p in enumerate(p_values) if not math.isnan(p))
    running = 0.0
    count = len(valid)
    for rank, (p_value, idx) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[idx] = running
    return adjusted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_run(
    *,
    dataset_path: str | Path,
    prediction_paths: Iterable[str | Path],
    output_dir: str | Path,
    target_method: str = "full_medimem",
    compare_methods: Iterable[str] | None = None,
    resamples: int = 10_000,
    seed: int = 20260706,
) -> Path:
    cases = read_jsonl(dataset_path)
    predictions = load_predictions(prediction_paths)
    case_by_id = {str(case["case_id"]): case for case in cases}
    evaluation = evaluate_predictions(cases, predictions)
    rows = list(evaluation["case_rows"])
    for row in rows:
        case = case_by_id[str(row["case_id"])]
        row["source"] = source_name(case)
        row["stratum"] = case_stratum(case)

    methods = sorted({str(row["method"]) for row in rows})
    baselines = list(compare_methods or [method for method in methods if method != target_method])
    strata = ["core_longitudinal", "medical_qa_transfer", "knowledge_transfer", "dialogue_stress"]
    metrics = ["primary_correct", "diagnosis_f1", "pdo_0.5", "pdo_0.65", "pdo_0.8"]
    budgets = sorted(
        {str(row.get("completion_token_budget") or "unbounded") for row in rows},
        key=lambda value: (value == "unbounded", int(value) if value.isdigit() else 0),
    )

    summary_rows: list[dict[str, Any]] = []
    for stratum in strata:
        for budget in budgets:
            stratum_rows = [
                row
                for row in rows
                if row["stratum"] == stratum
                and str(row.get("completion_token_budget") or "unbounded") == budget
            ]
            for method in methods:
                method_rows = [row for row in stratum_rows if row["method"] == method]
                applicable = [row for row in method_rows if row.get("diagnosis_metric_applicable")]
                if not method_rows:
                    continue
                summary_rows.append(
                    {
                        "stratum": stratum,
                        "completion_token_budget": budget,
                        "method": method,
                        "n": len(method_rows),
                        "diagnosis_n": len(applicable),
                        "coverage": len(applicable) / len(method_rows),
                        "top1": mean(float(row["primary_correct"]) for row in applicable),
                        "diagnosis_f1": mean(float(row["diagnosis_f1"]) for row in applicable),
                        "pdo_top1_0.50": mean(pdo(row, 0.5) for row in applicable),
                        "pdo_top1_0.65": mean(pdo(row, 0.65) for row in applicable),
                        "pdo_top1_0.80": mean(pdo(row, 0.8) for row in applicable),
                        "avg_tokens": mean(float(row.get("tokens") or 0) for row in method_rows),
                        "avg_prompt_tokens": mean(float(row.get("prompt_tokens") or 0) for row in method_rows),
                        "avg_completion_tokens": mean(float(row.get("completion_tokens") or 0) for row in method_rows),
                        "avg_model_calls": mean(float(row.get("model_calls") or 0) for row in method_rows),
                        "avg_latency_s": mean(float(row.get("latency_ms") or 0) / 1000.0 for row in method_rows),
                        "p95_latency_s": percentile(
                            [float(row.get("latency_ms") or 0) / 1000.0 for row in method_rows], 0.95
                        ),
                    }
                )

    comparison_rows: list[dict[str, Any]] = []
    for stratum in strata:
        for budget in budgets:
            stratum_rows = [
                row
                for row in rows
                if row["stratum"] == stratum
                and str(row.get("completion_token_budget") or "unbounded") == budget
            ]
            for baseline in baselines:
                for metric in metrics:
                    target_values, baseline_values = paired_values(stratum_rows, target_method, baseline, metric)
                    if not target_values:
                        continue
                    delta, ci_low, ci_high = paired_bootstrap_ci(
                        target_values, baseline_values, resamples=resamples, seed=seed
                    )
                    p_value = paired_permutation_p(target_values, baseline_values, resamples=resamples, seed=seed + 1)
                    comparison_rows.append(
                        {
                            "stratum": stratum,
                            "completion_token_budget": budget,
                            "target": target_method,
                            "baseline": baseline,
                            "metric": metric,
                            "paired_n": len(target_values),
                            "target_mean": mean(target_values),
                            "baseline_mean": mean(baseline_values),
                            "delta": delta,
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "permutation_p": p_value,
                        }
                    )
    adjusted = holm_adjust([float(row["permutation_p"]) for row in comparison_rows])
    for row, value in zip(comparison_rows, adjusted):
        row["holm_p"] = value

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "stratified_metrics.csv", summary_rows)
    write_csv(output / "paired_statistics.csv", comparison_rows)
    (output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "prediction_paths": [str(path) for path in prediction_paths],
                "target_method": target_method,
                "compare_methods": baselines,
                "resamples": resamples,
                "seed": seed,
                "stratum_by_source": STRATUM_BY_SOURCE,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output
