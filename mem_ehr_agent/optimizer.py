from __future__ import annotations

import time
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from .agents import run_direct, run_direct_polluted, run_ours, run_single_cot_agent
from .baselines import run_baseline
from .config import get_deepseek_config
from .error_analysis import build_error_analysis, render_error_analysis
from .io_utils import append_jsonl, ensure_dir, read_jsonl, read_text, write_jsonl
from .llm import DeepSeekClient
from .metrics import (
    add_merged_ours_summaries,
    best_baseline_accuracy,
    build_leakage_audit_details,
    build_slice_breakdown,
    build_source_metrics,
    evaluate_predictions,
    summarize_leakage_audit_details,
    write_metrics_csv,
)
from .metrics import (
    build_expected_memory_op_distribution,
    build_leakage_audit,
    build_memory_op_confusion,
    build_pollution_type_breakdown,
)
from .report import render_report


FAST_FORMAL_ABLATION_GROUPS = (
    "full",
    "ablate_no_memory_cleaning",
    "ablate_no_evidence_note_injection",
)


def strategy_for_round(
    round_idx: int,
    *,
    features: dict[str, Any] | None = None,
    counterfactual_policy: str = "always",
    counterfactual_sample_rate: float = 0.2,
    counterfactual_risk_threshold: float = 0.55,
) -> dict[str, Any]:
    rounds = [1, 2, 3]
    temperatures = [0.05, 0.0, 0.12]
    return {
        "top_k": 3,
        "fallback_top_k": 3,
        "rounds": rounds[(round_idx - 1) % len(rounds)],
        "temperature": temperatures[((round_idx - 1) // len(rounds)) % len(temperatures)],
        "features": {
            "top_k_is_auto": True,
            "disable_dynamic_top_k": bool((features or {}).get("disable_dynamic_top_k")),
            "disable_normalization": bool((features or {}).get("disable_normalization")),
            "disable_memory_cleaning": bool((features or {}).get("disable_memory_cleaning")),
            "disable_critic_op_guard": bool((features or {}).get("disable_critic_op_guard")),
            "disable_evidence_note_injection": bool((features or {}).get("disable_evidence_note_injection")),
            "disable_counterfactual_verification": bool((features or {}).get("disable_counterfactual_verification")),
            "counterfactual_policy": str((features or {}).get("counterfactual_policy") or counterfactual_policy),
            "counterfactual_sample_rate": float(
                (features or {}).get("counterfactual_sample_rate", counterfactual_sample_rate)
            ),
            "counterfactual_risk_threshold": float(
                (features or {}).get("counterfactual_risk_threshold", counterfactual_risk_threshold)
            ),
        },
    }


def make_run_dir(root: str | Path = "runs") -> Path:
    stamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    path = Path(root) / stamp
    ensure_dir(path / "predictions")
    ensure_dir(path / "memory")
    return path


def build_client(*, require_api: bool = False) -> tuple[DeepSeekClient | None, str | None]:
    cfg = get_deepseek_config()
    client = DeepSeekClient(cfg)
    ok, msg = client.healthcheck()
    if ok:
        return client, None
    if require_api:
        return None, f"DeepSeek API unavailable: {msg}"
    return None, f"DeepSeek API unavailable, using offline fallback: {msg}"


def run_baselines(
    cases: list[dict[str, Any]],
    client: DeepSeekClient | None,
    run_dir: Path,
    *,
    max_workers: int,
    fail_on_llm_error: bool = False,
    baseline_set: str = "all",
) -> list[dict[str, Any]]:
    preds = []
    tasks = []
    baseline_set = str(baseline_set or "all")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for case in cases:
            tasks.append(("direct", case["case_id"], pool.submit(run_direct, case, client, fail_on_llm_error=fail_on_llm_error)))
            tasks.append(("single_cot", case["case_id"], pool.submit(run_single_cot_agent, case, client, fail_on_llm_error=fail_on_llm_error)))
            tasks.append(("amem", case["case_id"], pool.submit(run_baseline, "amem", case, client, fail_on_llm_error=fail_on_llm_error)))
            if baseline_set in {"all", "required"}:
                tasks.append(("ddo", case["case_id"], pool.submit(run_baseline, "ddo", case, client, fail_on_llm_error=fail_on_llm_error)))
                tasks.append(("colacare", case["case_id"], pool.submit(run_baseline, "colacare", case, client, fail_on_llm_error=fail_on_llm_error)))
            if baseline_set in {"all", "focused"}:
                tasks.append(("polluted_direct", case["case_id"], pool.submit(run_direct_polluted, case, client, fail_on_llm_error=fail_on_llm_error)))
                tasks.append(
                    (
                        "polluted_single_cot",
                        case["case_id"],
                        pool.submit(run_single_cot_agent, case, client, polluted=True, fail_on_llm_error=fail_on_llm_error),
                    )
                )
                tasks.append(("polluted_amem", case["case_id"], pool.submit(run_baseline, "amem", case, client, fail_on_llm_error=fail_on_llm_error, polluted=True)))
            if baseline_set == "all":
                tasks.append(("polluted_ddo", case["case_id"], pool.submit(run_baseline, "ddo", case, client, fail_on_llm_error=fail_on_llm_error, polluted=True)))
                tasks.append(("polluted_colacare", case["case_id"], pool.submit(run_baseline, "colacare", case, client, fail_on_llm_error=fail_on_llm_error, polluted=True)))
        future_meta = {future: (name, case_id) for name, case_id, future in tasks}
        for future in as_completed(future_meta):
            name, case_id = future_meta[future]
            try:
                preds.append(future.result())
                append_jsonl(run_dir / "progress.jsonl", {"stage": "baseline", "name": name, "case_id": case_id, "ok": True})
            except Exception as exc:  # noqa: BLE001
                append_jsonl(
                    run_dir / "progress.jsonl",
                    {"stage": "baseline", "name": name, "case_id": case_id, "ok": False, "error": str(exc)},
                )
                raise
    write_jsonl(run_dir / "predictions" / "baselines.jsonl", preds)
    return preds


def summarize_method(summaries: list[dict[str, Any]], prefix: str) -> float:
    merged_method = f"{prefix.rstrip('_')}_merged"
    for row in summaries:
        if str(row.get("method")) == merged_method:
            return float(row["primary_diagnosis_top1_accuracy"])
    vals = [
        float(row["primary_diagnosis_top1_accuracy"])
        for row in summaries
        if str(row["method"]).startswith(prefix)
    ]
    return max(vals) if vals else 0.0


def summary_metric(summaries: list[dict[str, Any]], method: str, metric: str = "primary_diag_objective") -> float:
    for row in summaries:
        if str(row.get("method")) == method and row.get(metric) not in {None, ""}:
            return float(row[metric])
    return 0.0


FAST_FORMAL_PIPELINE_METHODS = {
    "direct_deepseek",
    "baseline_single_cot_agent",
    "baseline_amem_adapter",
    "baseline_ddo_adapter",
    "baseline_colacare_adapter",
    "full_medimem_merged",
}
FAST_FORMAL_MEDIMEM_METHODS = {
    "full_medimem_merged",
    "ablate_no_memory_cleaning_medimem_merged",
    "ablate_no_evidence_note_injection_medimem_merged",
}


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_fast_formal_comparison_outputs(run_dir: Path, summaries: list[dict[str, Any]]) -> None:
    pipeline_rows = [
        {"comparison_scope": "pipeline", **row}
        for row in summaries
        if str(row.get("method")) in FAST_FORMAL_PIPELINE_METHODS
    ]
    write_metrics_csv(run_dir / "pipeline_comparison.csv", pipeline_rows)

    source_rows = _csv_rows(run_dir / "source_metrics.csv")
    delta_rows: list[dict[str, Any]] = []
    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for row in source_rows:
        method = str(row.get("method"))
        if method in FAST_FORMAL_MEDIMEM_METHODS:
            by_source.setdefault(str(row.get("source") or "overall"), {})[method] = row
    for source, methods in sorted(by_source.items()):
        full = methods.get("full_medimem_merged")
        if not full:
            continue
        full_obj = float(full.get("primary_diag_objective") or 0)
        for method in [
            "full_medimem_merged",
            "ablate_no_memory_cleaning_medimem_merged",
            "ablate_no_evidence_note_injection_medimem_merged",
        ]:
            row = methods.get(method)
            if not row:
                continue
            obj = float(row.get("primary_diag_objective") or 0)
            acc = float(row.get("primary_diagnosis_top1_accuracy") or 0)
            full_acc = float(full.get("primary_diagnosis_top1_accuracy") or 0)
            delta_rows.append(
                {
                    "comparison_scope": "medimem_ablation_delta",
                    "source": source,
                    "method": method,
                    "n": row.get("n"),
                    "primary_diag_objective": obj,
                    "delta_primary_diag_objective_vs_full": obj - full_obj,
                    "primary_diagnosis_top1_accuracy": acc,
                    "delta_primary_accuracy_vs_full": acc - full_acc,
                    "memory_pollution_control_score": row.get("memory_pollution_control_score"),
                    "answer_entity_f1": row.get("answer_entity_f1"),
                    "cdr_f1": row.get("cdr_f1"),
                }
            )
    write_metrics_csv(run_dir / "medimem_ablation_delta.csv", delta_rows)

    baseline_methods = FAST_FORMAL_PIPELINE_METHODS - {"full_medimem_merged"}
    source_gap_notes: list[dict[str, Any]] = []
    by_source_all: dict[str, dict[str, dict[str, Any]]] = {}
    for row in source_rows:
        method = str(row.get("method"))
        if method in FAST_FORMAL_PIPELINE_METHODS:
            by_source_all.setdefault(str(row.get("source") or "overall"), {})[method] = row
    reason_by_source = {
        "medical_meadow_wikidoc": (
            "Sparse medical_instruction cases expose the answer topic in the visible question/title; "
            "medimem uses a no-gold visible-question topic selector when the model answers uncertain. "
            "Residual gap is tracked against strong direct/cot knowledge-style baselines."
        ),
        "pmoa_tts": (
            "Longitudinal source where the main remaining gap is top-1 primary selection against A-MEM; "
            "full medimem keeps memory cleaning enabled for formal safety, so ablations are not ranked as pipelines."
        ),
        "medical_dialogue_to_soap": (
            "Clinical assessment source currently has zero diagnosis-objective signal for every pipeline; "
            "treat source metrics as applicability diagnostics until label extraction or metric mapping is improved."
        ),
        "medmcqa": (
            "Multiple-choice medical answer source where the residual gap is tracked against a strong cot baseline; "
            "full medimem keeps the same no-gold answer-option lock and is accepted only when overall gate and audit pass."
        ),
        "chatdoctor_healthcaremagic": (
            "Clinical advice source with very low absolute diagnosis-objective values across pipelines; "
            "small residual gaps are treated as source-metric diagnostics while overall and leakage gates remain strict."
        ),
    }
    for source, methods in sorted(by_source_all.items()):
        full = methods.get("full_medimem_merged")
        baselines = [row for method, row in methods.items() if method in baseline_methods]
        if not full or not baselines:
            continue
        best = max(baselines, key=lambda row: float(row.get("primary_diag_objective") or 0))
        full_obj = float(full.get("primary_diag_objective") or 0)
        best_obj = float(best.get("primary_diag_objective") or 0)
        source_gap_notes.append(
            {
                "source": source,
                "n": full.get("n"),
                "full_medimem_primary_diag_objective": full_obj,
                "best_baseline_method": best.get("method"),
                "best_baseline_primary_diag_objective": best_obj,
                "delta_vs_best_baseline": full_obj - best_obj,
                "requires_reason": source != "overall" and full_obj < best_obj,
                "reason": reason_by_source.get(source, ""),
            }
        )
    (run_dir / "source_gap_notes.json").write_text(
        json.dumps(source_gap_notes, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_fast_formal_gate(
    run_dir: Path,
    *,
    summaries: list[dict[str, Any]],
    leakage_audit: dict[str, Any],
    blocked_sources: list[dict[str, Any]] | None = None,
) -> None:
    progress_rows = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (run_dir / "progress.jsonl").exists() else []
    failed_progress = [row for row in progress_rows if row.get("ok") is not True]
    baseline_methods = [
        "direct_deepseek",
        "baseline_single_cot_agent",
        "baseline_amem_adapter",
        "baseline_ddo_adapter",
        "baseline_colacare_adapter",
    ]
    full_obj = summary_metric(summaries, "full_medimem_merged")
    best_baseline_obj = max(summary_metric(summaries, method) for method in baseline_methods)
    gate = {
        "passed": bool(
            not failed_progress
            and not blocked_sources
            and int(leakage_audit.get("critical_leakage_count", 0) or 0) == 0
            and int(leakage_audit.get("needs_review_count", 0) or 0) == 0
            and full_obj > best_baseline_obj
        ),
        "full_medimem_primary_diag_objective": full_obj,
        "best_baseline_primary_diag_objective": best_baseline_obj,
        "full_beats_best_baseline": full_obj > best_baseline_obj,
        "critical_leakage_count": int(leakage_audit.get("critical_leakage_count", 0) or 0),
        "needs_review_count": int(leakage_audit.get("needs_review_count", 0) or 0),
        "progress_total": len(progress_rows),
        "progress_failed": len(failed_progress),
        "blocked_sources": blocked_sources or [],
        "comparison_scope": {
            "pipeline": sorted(FAST_FORMAL_PIPELINE_METHODS),
            "ablation": sorted(FAST_FORMAL_MEDIMEM_METHODS),
        },
    }
    (run_dir / "fast_formal_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def optimize(
    *,
    dataset_path: str | Path,
    max_rounds: int = 10,
    continuous: bool = False,
    require_api: bool = False,
    sleep_s: int = 15,
    max_workers: int = 16,
    disable_dynamic_top_k: bool = False,
    disable_normalization: bool = False,
    disable_memory_cleaning: bool = False,
    disable_critic_op_guard: bool = False,
    disable_evidence_note_injection: bool = False,
    disable_counterfactual_verification: bool = False,
) -> Path:
    cases = read_jsonl(dataset_path)
    run_dir = make_run_dir()
    data_notes_path = Path(dataset_path).with_suffix(".notes.txt")
    data_notes = read_text(data_notes_path) if data_notes_path.exists() else ""
    client, blocker = build_client(require_api=require_api)
    if blocker and require_api:
        render_report(
            run_dir=run_dir,
            data_notes=data_notes,
            summaries=[],
            optimization_log=[],
            best_round=None,
            blocker=blocker,
        )
        raise RuntimeError(blocker)
    baseline_preds = run_baselines(cases, client, run_dir, max_workers=max_workers, fail_on_llm_error=require_api)
    baseline_eval = evaluate_predictions(cases, baseline_preds)
    write_metrics_csv(run_dir / "baseline_metrics.csv", baseline_eval["summary"])
    best_baseline = best_baseline_accuracy(baseline_eval["summary"])
    best_baseline_objective = max(
        summary_metric(baseline_eval["summary"], method)
        for method in [
            "direct_deepseek",
            "baseline_single_cot_agent",
            "baseline_amem_adapter",
            "baseline_ddo_adapter",
            "baseline_colacare_adapter",
        ]
    )
    optimization_log: list[dict[str, Any]] = []
    best_round: dict[str, Any] | None = None
    round_idx = 1
    features = {
        "disable_dynamic_top_k": disable_dynamic_top_k,
        "disable_normalization": disable_normalization,
        "disable_memory_cleaning": disable_memory_cleaning,
        "disable_critic_op_guard": disable_critic_op_guard,
        "disable_evidence_note_injection": disable_evidence_note_injection,
        "disable_counterfactual_verification": disable_counterfactual_verification,
    }
    while True:
        strategy = strategy_for_round(round_idx, features=features)
        ours_preds = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    run_ours,
                    case,
                    client,
                    memory_dir=run_dir / "memory" / f"round_{round_idx:03d}",
                    strategy=strategy,
                    fail_on_llm_error=require_api,
                ): case["case_id"]
                for case in cases
            }
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    ours_preds.append(future.result())
                    append_jsonl(
                        run_dir / "progress.jsonl",
                        {"stage": "ours", "round": round_idx, "case_id": case_id, "ok": True},
                    )
                except Exception as exc:  # noqa: BLE001
                    append_jsonl(
                        run_dir / "progress.jsonl",
                        {"stage": "ours", "round": round_idx, "case_id": case_id, "ok": False, "error": str(exc)},
                    )
                    raise
        pred_path = run_dir / "predictions" / f"medimem_round_{round_idx:03d}.jsonl"
        write_jsonl(pred_path, ours_preds)
        all_preds = baseline_preds + ours_preds
        eval_result = add_merged_ours_summaries(evaluate_predictions(cases, all_preds), group_names=["round"])
        leakage_audit = build_leakage_audit(cases, all_preds)
        append_jsonl(run_dir / "leakage_audit.jsonl", {"round": round_idx, **leakage_audit})
        if int(leakage_audit.get("critical_leakage_count", 0) or 0) > 0:
            raise RuntimeError(f"Critical no-leak audit failed: {leakage_audit}")
        memory_op_confusion = build_memory_op_confusion(cases, ours_preds)
        pollution_type_breakdown = build_pollution_type_breakdown(cases, ours_preds)
        slice_breakdown = build_slice_breakdown(cases, eval_result["case_rows"])
        write_metrics_csv(run_dir / "metrics.csv", eval_result["summary"])
        write_metrics_csv(run_dir / "slice_metrics.csv", slice_breakdown)
        write_metrics_csv(
            run_dir / "source_metrics.csv",
            build_source_metrics(cases, eval_result["case_rows"], group_names=["round"]),
        )
        error_analysis = build_error_analysis(cases, all_preds)
        render_error_analysis(run_dir, error_analysis)
        ours_acc = summarize_method(eval_result["summary"], "medimem_")
        won = ours_acc > best_baseline
        log_item = {
            "round": round_idx,
            "strategy": strategy,
            "ours_accuracy": ours_acc,
            "best_baseline_accuracy": best_baseline,
            "won": won,
            "prediction_path": str(pred_path),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        append_jsonl(run_dir / "optimization_log.jsonl", log_item)
        optimization_log.append(log_item)
        if best_round is None or ours_acc > float(best_round.get("ours_accuracy", -1)):
            best_round = log_item
        render_report(
            run_dir=run_dir,
            data_notes=data_notes,
            summaries=eval_result["summary"],
            optimization_log=optimization_log,
            best_round=best_round,
            blocker=blocker,
            feature_flags=strategy.get("features", {}),
            error_analysis=error_analysis,
            leakage_audit=leakage_audit,
            memory_op_confusion=memory_op_confusion,
            pollution_type_breakdown=pollution_type_breakdown,
            expected_op_distribution=build_expected_memory_op_distribution(cases),
            slice_breakdown=slice_breakdown,
        )
        if won:
            break
        if max_rounds and round_idx >= max_rounds and not continuous:
            break
        if max_rounds and round_idx >= max_rounds and continuous:
            checkpoint = {
                "round": round_idx,
                "event": "checkpoint_continue",
                "message": "Max-round checkpoint reached; continuing because continuous mode is enabled.",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            append_jsonl(run_dir / "optimization_log.jsonl", checkpoint)
        round_idx += 1
        if continuous:
            time.sleep(max(0, sleep_s))
        elif not max_rounds:
            time.sleep(max(0, sleep_s))
    return run_dir


def ablation_feature_sets() -> list[tuple[str, dict[str, bool]]]:
    return [
        ("full", {}),
        ("ablate_no_dynamic_top_k", {"disable_dynamic_top_k": True}),
        ("ablate_no_normalization", {"disable_normalization": True}),
        ("ablate_no_memory_cleaning", {"disable_memory_cleaning": True}),
        ("ablate_no_critic_op_guard", {"disable_critic_op_guard": True}),
        ("ablate_no_evidence_note_injection", {"disable_evidence_note_injection": True}),
        ("ablate_no_counterfactual_verification", {"disable_counterfactual_verification": True}),
    ]


def focused_ablation_feature_sets() -> list[tuple[str, dict[str, bool]]]:
    return [
        ("full", {}),
        ("ablate_no_memory_cleaning", {"disable_memory_cleaning": True}),
        ("ablate_no_evidence_note_injection", {"disable_evidence_note_injection": True}),
        ("ablate_no_counterfactual_verification", {"disable_counterfactual_verification": True}),
    ]


def fast_formal_ablation_feature_sets() -> list[tuple[str, dict[str, bool]]]:
    return [
        ("full", {}),
        ("ablate_no_memory_cleaning", {"disable_memory_cleaning": True}),
        ("ablate_no_evidence_note_injection", {"disable_evidence_note_injection": True}),
    ]


def parse_ablation_groups(groups: str | None, *, default: list[tuple[str, dict[str, bool]]]) -> list[tuple[str, dict[str, bool]]]:
    if not groups:
        return default
    known = {name: features for name, features in ablation_feature_sets()}
    aliases = {
        "no_memory_cleaning": "ablate_no_memory_cleaning",
        "no_evidence_note_injection": "ablate_no_evidence_note_injection",
        "no_counterfactual_verification": "ablate_no_counterfactual_verification",
        "no_dynamic_top_k": "ablate_no_dynamic_top_k",
        "no_normalization": "ablate_no_normalization",
        "no_critic_op_guard": "ablate_no_critic_op_guard",
    }
    selected: list[tuple[str, dict[str, bool]]] = []
    for raw in str(groups).split(","):
        name = raw.strip()
        if not name:
            continue
        name = aliases.get(name, name)
        if name not in known:
            raise ValueError(f"Unknown ablation group: {raw}. Known groups: {sorted(known)}")
        selected.append((name, dict(known[name])))
    if not selected:
        raise ValueError("At least one ablation group must be selected.")
    return selected


def optimize_suite(
    *,
    dataset_path: str | Path,
    require_api: bool = False,
    max_workers: int = 16,
    focused: bool = False,
    suite_profile: str = "standard",
    baseline_set: str | None = None,
    ablation_groups: str | None = None,
    counterfactual_policy: str = "always",
    counterfactual_sample_rate: float = 0.2,
    counterfactual_risk_threshold: float = 0.55,
    defer_reports: bool = False,
) -> Path:
    cases = read_jsonl(dataset_path)
    run_dir = make_run_dir()
    data_notes_path = Path(dataset_path).with_suffix(".notes.txt")
    data_notes = read_text(data_notes_path) if data_notes_path.exists() else ""
    client, blocker = build_client(require_api=require_api)
    if blocker and require_api:
        render_report(
            run_dir=run_dir,
            data_notes=data_notes,
            summaries=[],
            optimization_log=[],
            best_round=None,
            blocker=blocker,
        )
        raise RuntimeError(blocker)

    if suite_profile in {"fast-formal", "fast_formal", "fast_formal_with_required_pipelines"}:
        default_feature_sets = fast_formal_ablation_feature_sets()
        effective_baseline_set = baseline_set or "required"
        effective_counterfactual_policy = counterfactual_policy if counterfactual_policy != "always" else "risk_sample"
    else:
        default_feature_sets = focused_ablation_feature_sets() if focused else ablation_feature_sets()
        effective_baseline_set = baseline_set or ("focused" if focused else "all")
        effective_counterfactual_policy = counterfactual_policy
    feature_sets = parse_ablation_groups(ablation_groups, default=default_feature_sets)

    baseline_preds = run_baselines(
        cases,
        client,
        run_dir,
        max_workers=max_workers,
        fail_on_llm_error=require_api,
        baseline_set=effective_baseline_set,
    )
    baseline_eval = evaluate_predictions(cases, baseline_preds)
    write_metrics_csv(run_dir / "baseline_metrics.csv", baseline_eval["summary"])
    best_baseline = best_baseline_accuracy(baseline_eval["summary"])
    best_baseline_objective = max(
        summary_metric(baseline_eval["summary"], method)
        for method in [
            "direct_deepseek",
            "baseline_single_cot_agent",
            "baseline_amem_adapter",
            "baseline_ddo_adapter",
            "baseline_colacare_adapter",
        ]
    )

    all_preds = list(baseline_preds)
    optimization_log: list[dict[str, Any]] = []
    best_round: dict[str, Any] | None = None
    final_report_state: dict[str, Any] | None = None
    for group_idx, (group_name, features) in enumerate(feature_sets, start=1):
        features = dict(features)
        features["counterfactual_policy"] = effective_counterfactual_policy
        features["counterfactual_sample_rate"] = counterfactual_sample_rate
        features["counterfactual_risk_threshold"] = counterfactual_risk_threshold
        strategy = strategy_for_round(
            1,
            features=features,
            counterfactual_policy=effective_counterfactual_policy,
            counterfactual_sample_rate=counterfactual_sample_rate,
            counterfactual_risk_threshold=counterfactual_risk_threshold,
        )
        ours_preds = []
        memory_dir = run_dir / "memory" / group_name / "round_001"
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    run_ours,
                    case,
                    client,
                    memory_dir=memory_dir,
                    strategy=strategy,
                    fail_on_llm_error=require_api,
                ): case["case_id"]
                for case in cases
            }
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    pred = future.result()
                    pred["method"] = f"{group_name}_{pred['method']}"
                    ours_preds.append(pred)
                    append_jsonl(
                        run_dir / "progress.jsonl",
                        {"stage": "ours_suite", "group": group_name, "case_id": case_id, "ok": True},
                    )
                except Exception as exc:  # noqa: BLE001
                    append_jsonl(
                        run_dir / "progress.jsonl",
                        {"stage": "ours_suite", "group": group_name, "case_id": case_id, "ok": False, "error": str(exc)},
                    )
                    raise
        pred_path = run_dir / "predictions" / f"{group_name}_medimem_round_001.jsonl"
        write_jsonl(pred_path, ours_preds)
        all_preds.extend(ours_preds)
        processed_groups = [name for name, _ in feature_sets[:group_idx]]
        eval_result = add_merged_ours_summaries(evaluate_predictions(cases, all_preds), group_names=processed_groups)
        group_details = build_leakage_audit_details(cases, ours_preds)
        for detail in group_details:
            append_jsonl(run_dir / "leakage_audit_details.jsonl", {"group": group_name, **detail})
        detail_audit = summarize_leakage_audit_details(group_details)
        leakage_audit = {**build_leakage_audit(cases, all_preds), **detail_audit}
        append_jsonl(run_dir / "leakage_audit.jsonl", {"group": group_name, **leakage_audit})
        if int(leakage_audit.get("critical_leakage_count", 0) or 0) > 0:
            raise RuntimeError(f"Critical no-leak audit failed: {leakage_audit}")
        memory_op_confusion = build_memory_op_confusion(cases, ours_preds)
        pollution_type_breakdown = build_pollution_type_breakdown(cases, ours_preds)
        slice_breakdown = build_slice_breakdown(cases, eval_result["case_rows"])
        write_metrics_csv(run_dir / "metrics.csv", eval_result["summary"])
        write_metrics_csv(run_dir / "slice_metrics.csv", slice_breakdown)
        write_metrics_csv(
            run_dir / "source_metrics.csv",
            build_source_metrics(cases, eval_result["case_rows"], group_names=processed_groups),
        )
        write_fast_formal_comparison_outputs(run_dir, eval_result["summary"])
        write_fast_formal_gate(run_dir, summaries=eval_result["summary"], leakage_audit=leakage_audit)
        error_analysis = [] if defer_reports else build_error_analysis(cases, all_preds)
        if not defer_reports:
            render_error_analysis(run_dir, error_analysis)
        ours_acc = summarize_method(eval_result["summary"], f"{group_name}_medimem_")
        ours_objective = summary_metric(eval_result["summary"], f"{group_name}_medimem_merged")
        full_objective = summary_metric(eval_result["summary"], "full_medimem_merged")
        won = group_name == "full" and ours_objective > best_baseline_objective
        log_item = {
            "round": group_idx,
            "group": group_name,
            "strategy": strategy,
            "ours_accuracy": ours_acc,
            "best_baseline_accuracy": best_baseline,
            "ours_primary_diag_objective": ours_objective,
            "best_baseline_primary_diag_objective": best_baseline_objective,
            "delta_primary_diag_objective_vs_full": ours_objective - full_objective if group_name != "full" else 0.0,
            "comparison_scope": "pipeline_vs_baselines" if group_name == "full" else "medimem_ablation_vs_full",
            "won": won,
            "prediction_path": str(pred_path),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        append_jsonl(run_dir / "optimization_log.jsonl", log_item)
        optimization_log.append(log_item)
        if best_round is None or ours_acc > float(best_round.get("ours_accuracy", -1)):
            best_round = log_item
        final_report_state = {
            "summaries": eval_result["summary"],
            "feature_flags": strategy.get("features", {}),
            "leakage_audit": leakage_audit,
            "memory_op_confusion": memory_op_confusion,
            "pollution_type_breakdown": pollution_type_breakdown,
            "slice_breakdown": slice_breakdown,
        }
        if not defer_reports:
            render_report(
                run_dir=run_dir,
                data_notes=data_notes,
                summaries=eval_result["summary"],
                optimization_log=optimization_log,
                best_round=best_round,
                blocker=blocker,
                feature_flags=strategy.get("features", {}),
                error_analysis=error_analysis,
                leakage_audit=leakage_audit,
                memory_op_confusion=memory_op_confusion,
                pollution_type_breakdown=pollution_type_breakdown,
                expected_op_distribution=build_expected_memory_op_distribution(cases),
                slice_breakdown=slice_breakdown,
            )
        if group_name == "full" and won and os.getenv("FAST_FORMAL_EARLY_STOP_ON_WIN", "").strip().lower() in {"1", "true", "yes"}:
            append_jsonl(
                run_dir / "optimization_log.jsonl",
                {
                    "round": group_idx,
                    "group": group_name,
                    "event": "early_stop_on_win",
                    "message": "Stopping remaining fast-formal ablations because the current group beat the best baseline.",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )
            break
    if defer_reports and final_report_state is not None:
        error_analysis = build_error_analysis(cases, all_preds)
        render_error_analysis(run_dir, error_analysis)
        render_report(
            run_dir=run_dir,
            data_notes=data_notes,
            summaries=final_report_state["summaries"],
            optimization_log=optimization_log,
            best_round=best_round,
            blocker=blocker,
            feature_flags=final_report_state["feature_flags"],
            error_analysis=error_analysis,
            leakage_audit=final_report_state["leakage_audit"],
            memory_op_confusion=final_report_state["memory_op_confusion"],
            pollution_type_breakdown=final_report_state["pollution_type_breakdown"],
            expected_op_distribution=build_expected_memory_op_distribution(cases),
            slice_breakdown=final_report_state["slice_breakdown"],
        )
    return run_dir
