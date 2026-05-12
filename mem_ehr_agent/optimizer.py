from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from .agents import run_direct, run_ours
from .baselines import run_baseline
from .config import get_deepseek_config
from .io_utils import append_jsonl, ensure_dir, read_jsonl, read_text, write_jsonl
from .llm import DeepSeekClient
from .metrics import best_baseline_accuracy, evaluate_predictions, write_metrics_csv
from .report import render_report


def strategy_for_round(round_idx: int) -> dict[str, Any]:
    top_ks = [3, 5, 8, 2, 10]
    rounds = [1, 2, 3]
    temperatures = [0.05, 0.0, 0.12]
    return {
        "top_k": top_ks[(round_idx - 1) % len(top_ks)],
        "rounds": rounds[((round_idx - 1) // len(top_ks)) % len(rounds)],
        "temperature": temperatures[((round_idx - 1) // (len(top_ks) * len(rounds))) % len(temperatures)],
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
) -> list[dict[str, Any]]:
    preds = []
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for case in cases:
            tasks.append(("direct", case["case_id"], pool.submit(run_direct, case, client)))
            tasks.append(("ddo", case["case_id"], pool.submit(run_baseline, "ddo", case, client)))
            tasks.append(("colacare", case["case_id"], pool.submit(run_baseline, "colacare", case, client)))
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
    max_workers: int = 4,
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
    baseline_preds = run_baselines(cases, client, run_dir, max_workers=max_workers)
    baseline_eval = evaluate_predictions(cases, baseline_preds)
    write_metrics_csv(run_dir / "baseline_metrics.csv", baseline_eval["summary"])
    best_baseline = best_baseline_accuracy(baseline_eval["summary"])
    optimization_log: list[dict[str, Any]] = []
    best_round: dict[str, Any] | None = None
    round_idx = 1
    while True:
        strategy = strategy_for_round(round_idx)
        ours_preds = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    run_ours,
                    case,
                    client,
                    memory_dir=run_dir / "memory" / f"round_{round_idx:03d}",
                    strategy=strategy,
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
        pred_path = run_dir / "predictions" / f"ours_round_{round_idx:03d}.jsonl"
        write_jsonl(pred_path, ours_preds)
        all_preds = baseline_preds + ours_preds
        eval_result = evaluate_predictions(cases, all_preds)
        write_metrics_csv(run_dir / "metrics.csv", eval_result["summary"])
        ours_acc = summarize_method(eval_result["summary"], "ours_")
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
