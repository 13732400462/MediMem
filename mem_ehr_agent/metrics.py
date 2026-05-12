from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir


def norm(text: str) -> str:
    text = canonical_for_metric(text)
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_for_metric(text: str) -> str:
    low = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()
    mapping = {
        "sle": "systemic lupus erythematosus",
        "nxg": "necrobiotic xanthogranuloma",
        "rdd": "rosai dorfman destombes disease",
        "extra nodal rdd": "rosai dorfman destombes disease",
        "e granulosus": "echinococcosis",
        "invasive ductal carcinoma": "breast cancer",
        "bilateral poland syndrome": "poland syndrome",
    }
    for key, value in mapping.items():
        if low == key or key in low:
            return value
    return str(text)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = norm(pred).split()
    gold_tokens = norm(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = defaultdict(int)
    gold_counts = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    for token in gold_tokens:
        gold_counts[token] += 1
    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def diagnosis_match(pred: str, gold: str) -> bool:
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return False
    substring_match = (len(p) >= 4 and p in g) or (len(g) >= 4 and g in p)
    return p == g or substring_match or token_f1(p, g) >= 0.72


def list_f1(preds: list[str], golds: list[str]) -> float:
    if not preds and not golds:
        return 1.0
    if not preds or not golds:
        return 0.0
    matched_gold: set[int] = set()
    tp = 0
    for pred in preds:
        best_idx = None
        best_score = 0.0
        for idx, gold in enumerate(golds):
            if idx in matched_gold:
                continue
            score = token_f1(pred, gold)
            if diagnosis_match(pred, gold):
                score = max(score, 0.9)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= 0.5:
            matched_gold.add(best_idx)
            tp += 1
    precision = tp / len(preds)
    recall = tp / len(golds)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def pollution_suppression(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected = [op for op in case.get("expected_memory_ops", []) if op.get("op") in {"Discard", "Invalidate"}]
    if not expected:
        return 1.0
    actual = pred.get("memory_ops") or []
    hits = 0
    for exp in expected:
        if any(op.get("op") == exp.get("op") and str(op.get("target")) == str(exp.get("target")) for op in actual):
            hits += 1
    return hits / len(expected)


def counterfactual_score(case: dict[str, Any], pred: dict[str, Any]) -> float:
    # Cheap proxy for CPG in V0: evidence-rich high confidence should drop if the counterfactual
    # explicitly removes the diagnosis-bearing evidence. Full LLM re-query can be added later.
    if not case.get("counterfactuals"):
        return 0.0
    evidence_text = " ".join(pred.get("evidence") or [])
    primary = pred.get("primary_diagnosis", "")
    if diagnosis_match(primary, case.get("labels", {}).get("primary_diagnosis", "")) and token_f1(evidence_text, primary) > 0:
        return 0.4
    return 0.1


def evaluate_predictions(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    case_by_id = {case["case_id"]: case for case in cases}
    rows = []
    for pred in predictions:
        case = case_by_id[pred["case_id"]]
        labels = case["labels"]
        primary_ok = diagnosis_match(pred.get("primary_diagnosis", ""), labels.get("primary_diagnosis", ""))
        diag_f1 = list_f1(pred.get("diagnosis_list", []), labels.get("diagnosis_list", []))
        cdr_f1s = [
            token_f1(pred.get("primary_diagnosis", ""), qa.get("answer", ""))
            for qa in case.get("qa_tasks", [])
            if qa.get("type") == "CDR"
        ]
        rows.append(
            {
                "case_id": pred["case_id"],
                "method": pred["method"],
                "primary_correct": 1.0 if primary_ok else 0.0,
                "diagnosis_f1": diag_f1,
                "cdr_f1": sum(cdr_f1s) / len(cdr_f1s) if cdr_f1s else 0.0,
                "pollution_suppression": pollution_suppression(case, pred),
                "counterfactual_score": counterfactual_score(case, pred),
                "tokens": float((pred.get("usage") or {}).get("total_tokens", 0) or 0),
            }
        )
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    summaries = []
    for method, vals in sorted(by_method.items()):
        n = len(vals)
        summaries.append(
            {
                "method": method,
                "n": n,
                "primary_diagnosis_top1_accuracy": avg(vals, "primary_correct"),
                "diagnosis_list_f1": avg(vals, "diagnosis_f1"),
                "cdr_f1": avg(vals, "cdr_f1"),
                "memory_pollution_suppression": avg(vals, "pollution_suppression"),
                "counterfactual_robustness_proxy": avg(vals, "counterfactual_score"),
                "avg_tokens": avg(vals, "tokens"),
            }
        )
    return {"case_rows": rows, "summary": summaries}


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / max(1, len(rows))


def write_metrics_csv(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if not summaries:
        p.write_text("", encoding="utf-8")
        return
    fieldnames = list(summaries[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def best_baseline_accuracy(summaries: list[dict[str, Any]]) -> float:
    vals = [
        float(s["primary_diagnosis_top1_accuracy"])
        for s in summaries
        if str(s["method"]).startswith("baseline_")
    ]
    return max(vals) if vals else 0.0
