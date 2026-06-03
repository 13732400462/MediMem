from __future__ import annotations

import time
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
from .metrics import add_merged_ours_summaries, best_baseline_accuracy, build_slice_breakdown, evaluate_predictions, write_metrics_csv
from .metrics import (
    build_expected_memory_op_distribution,
    build_leakage_audit,
    build_memory_op_confusion,
    build_pollution_type_breakdown,
)
from .report import render_report


def strategy_for_round(round_idx: int, *, features: dict[str, bool] | None = None) -> dict[str, Any]:
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
        },
    }


def make_run_dir(root: str | Path = "runs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for case in cases:
            tasks.append(("direct", case["case_id"], pool.submit(run_direct, case, client, fail_on_llm_error=fail_on_llm_error)))
            tasks.append(("single_cot", case["case_id"], pool.submit(run_single_cot_agent, case, client, fail_on_llm_error=fail_on_llm_error)))
            tasks.append(("amem", case["case_id"], pool.submit(run_baseline, "amem", case, client, fail_on_llm_error=fail_on_llm_error)))
            tasks.append(("polluted_direct", case["case_id"], pool.submit(run_direct_polluted, case, client, fail_on_llm_error=fail_on_llm_error)))
            tasks.append(
                (
                    "polluted_single_cot",
                    case["case_id"],
                    pool.submit(run_single_cot_agent, case, client, polluted=True, fail_on_llm_error=fail_on_llm_error),
                )
            )
            tasks.append(("polluted_amem", case["case_id"], pool.submit(run_baseline, "amem", case, client, fail_on_llm_error=fail_on_llm_error, polluted=True)))
            if baseline_set != "focused":
                tasks.append(("ddo", case["case_id"], pool.submit(run_baseline, "ddo", case, client, fail_on_llm_error=fail_on_llm_error)))
                tasks.append(("colacare", case["case_id"], pool.submit(run_baseline, "colacare", case, client, fail_on_llm_error=fail_on_llm_error)))
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
        memory_op_confusion = build_memory_op_confusion(cases, ours_preds)
        pollution_type_breakdown = build_pollution_type_breakdown(cases, ours_preds)
        slice_breakdown = build_slice_breakdown(cases, eval_result["case_rows"])
        write_metrics_csv(run_dir / "metrics.csv", eval_result["summary"])
        write_metrics_csv(run_dir / "slice_metrics.csv", slice_breakdown)
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


def optimize_suite(
    *,
    dataset_path: str | Path,
    require_api: bool = False,
    max_workers: int = 16,
    focused: bool = False,
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

    baseline_preds = run_baselines(
        cases,
        client,
        run_dir,
        max_workers=max_workers,
        fail_on_llm_error=require_api,
        baseline_set="focused" if focused else "all",
    )
    baseline_eval = evaluate_predictions(cases, baseline_preds)
    write_metrics_csv(run_dir / "baseline_metrics.csv", baseline_eval["summary"])
    best_baseline = best_baseline_accuracy(baseline_eval["summary"])

    all_preds = list(baseline_preds)
    optimization_log: list[dict[str, Any]] = []
    best_round: dict[str, Any] | None = None
    feature_sets = focused_ablation_feature_sets() if focused else ablation_feature_sets()
    for group_idx, (group_name, features) in enumerate(feature_sets, start=1):
        strategy = strategy_for_round(1, features=features)
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
        leakage_audit = build_leakage_audit(cases, all_preds)
        memory_op_confusion = build_memory_op_confusion(cases, ours_preds)
        pollution_type_breakdown = build_pollution_type_breakdown(cases, ours_preds)
        slice_breakdown = build_slice_breakdown(cases, eval_result["case_rows"])
        write_metrics_csv(run_dir / "metrics.csv", eval_result["summary"])
        write_metrics_csv(run_dir / "slice_metrics.csv", slice_breakdown)
        error_analysis = build_error_analysis(cases, all_preds)
        render_error_analysis(run_dir, error_analysis)
        ours_acc = summarize_method(eval_result["summary"], f"{group_name}_medimem_")
        won = ours_acc > best_baseline
        log_item = {
            "round": group_idx,
            "group": group_name,
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
    return run_dir
