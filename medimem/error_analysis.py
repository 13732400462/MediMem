from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import write_text
from .metrics import diagnosis_match, list_f1, token_f1


def _by_case_method(predictions: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for pred in predictions:
        out.setdefault(str(pred.get("case_id")), {})[str(pred.get("method"))] = pred
    return out


def _case_line(case: dict[str, Any], pred: dict[str, Any] | None, gold_primary: str) -> str:
    if not pred:
        return f"- {case['case_id']}：缺少预测结果；gold={gold_primary}"
    diag_f1 = list_f1(pred.get("diagnosis_list", []), case.get("labels", {}).get("diagnosis_list", []))
    cdr_scores = [
        token_f1(pred.get("primary_diagnosis", ""), qa.get("answer", ""))
        for qa in case.get("qa_tasks", [])
        if qa.get("type") == "CDR"
    ]
    cdr_f1 = sum(cdr_scores) / len(cdr_scores) if cdr_scores else 0.0
    return (
        f"- {case['case_id']}：gold={gold_primary}；pred={pred.get('primary_diagnosis')}；"
        f"diagnosis_f1={diag_f1:.3f}；cdr_f1={cdr_f1:.3f}"
    )


def build_error_analysis(
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    ours_prefix: str = "ours_",
    baseline_method: str = "baseline_amem_adapter",
) -> dict[str, Any]:
    pred_by_case = _by_case_method(predictions)
    buckets: dict[str, list[str]] = {
        "ours_wrong_amem_right": [],
        "ours_right_amem_wrong": [],
        "both_wrong": [],
        "low_diagnosis_f1": [],
        "low_cdr_f1": [],
    }
    for case in cases:
        case_preds = pred_by_case.get(case["case_id"], {})
        ours = next((pred for method, pred in case_preds.items() if method.startswith(ours_prefix)), None)
        amem = case_preds.get(baseline_method)
        if not ours:
            continue
        gold_primary = str(case.get("labels", {}).get("primary_diagnosis", ""))
        ours_ok = diagnosis_match(str(ours.get("primary_diagnosis", "")), gold_primary)
        amem_ok = bool(amem and diagnosis_match(str(amem.get("primary_diagnosis", "")), gold_primary))
        if not ours_ok and amem_ok:
            buckets["ours_wrong_amem_right"].append(_case_line(case, ours, gold_primary))
        elif ours_ok and not amem_ok:
            buckets["ours_right_amem_wrong"].append(_case_line(case, ours, gold_primary))
        elif not ours_ok and not amem_ok:
            buckets["both_wrong"].append(_case_line(case, ours, gold_primary))
        diag_f1 = list_f1(ours.get("diagnosis_list", []), case.get("labels", {}).get("diagnosis_list", []))
        if diag_f1 < 0.5:
            buckets["low_diagnosis_f1"].append(_case_line(case, ours, gold_primary))
        cdr_scores = [
            token_f1(ours.get("primary_diagnosis", ""), qa.get("answer", ""))
            for qa in case.get("qa_tasks", [])
            if qa.get("type") == "CDR"
        ]
        cdr_f1 = sum(cdr_scores) / len(cdr_scores) if cdr_scores else 0.0
        if cdr_f1 < 0.35:
            buckets["low_cdr_f1"].append(_case_line(case, ours, gold_primary))
    return {"buckets": buckets}


def render_error_analysis(run_dir: str | Path, analysis: dict[str, Any]) -> str:
    titles = {
        "ours_wrong_amem_right": "Ours 错 / A-MEM 对",
        "ours_right_amem_wrong": "Ours 对 / A-MEM 错",
        "both_wrong": "双方都错",
        "low_diagnosis_f1": "诊断列表 F1 偏低",
        "low_cdr_f1": "CDR F1 偏低",
    }
    lines = ["# 错误样本分析", ""]
    for key, title in titles.items():
        items = analysis.get("buckets", {}).get(key, [])
        lines.extend([f"## {title}", f"- 数量：{len(items)}"])
        lines.extend(items[:20] or ["- 暂无"])
        lines.append("")
    text = "\n".join(lines)
    write_text(Path(run_dir) / "error_analysis_zh.md", text)
    return text
