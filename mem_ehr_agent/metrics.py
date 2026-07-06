from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir
from .medical_terms import canonicalize_diagnosis
from .task_profiles import MEDICAL_ANSWER_ENTITY, case_task_profile

COUNTERFACTUAL_CPG_THRESHOLD = 0.40
FORBIDDEN_VISIBLE_FIELDS = (
    "pollution_type",
    "staleness_type",
    "expected_op",
    "revised_claim",
    "preserved_facts",
    "target_diagnosis",
)


def is_medimem_method(method: Any) -> bool:
    text = str(method or "")
    return (
        text.startswith("medimem_")
        or text.startswith("full_medimem_")
        or "_medimem_" in text
        or text.startswith("ours_")
        or text.startswith("full_ours_")
        or "_ours_" in text
    )


@lru_cache(maxsize=200_000)
def norm(text: str) -> str:
    text = canonicalize_diagnosis(text)
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def diagnosis_match_any(pred: str, golds: list[str]) -> bool:
    return any(diagnosis_match(pred, gold) for gold in golds if str(gold).strip())


def list_f1(preds: list[str], golds: list[str]) -> float:
    preds = unique_normalized_items(preds)
    golds = unique_normalized_items(golds)
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


def list_f1_with_aliases(preds: list[str], golds: list[str], aliases: list[str]) -> float:
    preds = unique_normalized_items(preds)
    golds = unique_normalized_items(golds)
    alias_group = [str(item) for item in aliases if str(item).strip()]
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
            gold_aliases = [gold]
            if diagnosis_match(gold, alias_group[0] if alias_group else "") or idx == 0:
                gold_aliases.extend(alias_group)
            score = max(token_f1(pred, item) for item in gold_aliases if str(item).strip())
            if diagnosis_match_any(pred, gold_aliases):
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


def unique_normalized_items(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if is_obvious_non_diagnosis_item(text):
            continue
        key = norm(text)
        if text and key and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def is_obvious_non_diagnosis_item(text: str) -> bool:
    low = norm(text)
    if not low:
        return True
    noisy_phrases = (
        "anti infection treatment",
        "bone scan",
        "case discussion",
        "cerebral infarction suspected",
        "cerebral infarction treatment",
        "cystic mass",
        "cystic lesion",
        "ct scan",
        "denied risk factors",
        "disease progression",
        "depression on scalp",
        "elevation in leads",
        "fdg pet",
        "genetic testing",
        "genomic alterations",
        "high tumor mutational burden",
        "infection control",
        "kidney yang deficiency",
        "microvascular occlusion",
        "molecular diagnostics",
        "microsatellite stable",
        "mutation",
        "non enhancing mediastinal mass",
        "mediastinal mass",
        "mild tracheal compression",
        "mismatch repair deficient",
        "pet ct",
        "resection of",
        "right frontal lobe lesion",
        "right lobectomy",
        "round depression",
        "surgical removal",
        "post operative recovery",
        "scan normalization",
        "st segment depression",
        "st segment elevation",
        "stable disease",
        "treatment",
        "tumor markers",
        "whole body bone scan",
    )
    if any(phrase in low for phrase in noisy_phrases):
        return True
    if re.search(r"\bleads?\s+[ivx0-9]", low):
        return True
    return False


def pollution_suppression(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected = [op for op in case.get("expected_memory_ops", []) if op.get("op") in {"Discard", "Invalidate"}]
    if not expected:
        return 1.0
    actual = pred.get("memory_ops") or []
    hits = 0
    for exp in expected:
        if any(memory_op_matches(exp, op) for op in actual):
            hits += 1
    return hits / len(expected)


def hard_pollution_suppression(case: dict[str, Any], pred: dict[str, Any]) -> float:
    return pollution_suppression(case, pred)


def stale_memory_expected_ops(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        op
        for op in case.get("expected_memory_ops", [])
        if op.get("op") in {"Revise", "Invalidate", "Discard", "Keep", "Flag"}
    ]


def memory_op_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if actual.get("op") != expected.get("op"):
        return False
    expected_ids = {
        str(value)
        for value in (
            expected.get("target"),
            expected.get("seed_id"),
            expected.get("runtime_memory_id"),
        )
        if str(value or "").strip()
    }
    actual_ids = {
        str(value)
        for value in (
            actual.get("target"),
            actual.get("seed_id"),
            actual.get("revised_memory_id"),
        )
        if str(value or "").strip()
    }
    actual_ids.update(str(value) for value in actual.get("touched_memory_ids") or [] if str(value or "").strip())
    return bool(expected_ids & actual_ids)


def actual_op_for_target(pred: dict[str, Any], target: str) -> dict[str, Any] | None:
    for op in pred.get("memory_ops") or []:
        op_ids = {str(op.get("target") or ""), str(op.get("seed_id") or ""), str(op.get("revised_memory_id") or "")}
        op_ids.update(str(value) for value in op.get("touched_memory_ids") or [])
        if str(target) in op_ids:
            return op
    return None


ACTION_QUALITY: dict[str, dict[str, float]] = {
    "Revise": {"Revise": 1.0, "Flag": 0.45, "Invalidate": 0.35, "Discard": 0.0, "Keep": 0.0, "MISSING": 0.0},
    "Invalidate": {"Invalidate": 1.0, "Flag": 0.65, "Revise": 0.55, "Discard": 0.45, "Keep": 0.0, "MISSING": 0.0},
    "Discard": {"Discard": 1.0, "Invalidate": 0.45, "Flag": 0.35, "Revise": 0.25, "Keep": 0.0, "MISSING": 0.0},
    "Keep": {"Keep": 1.0, "MISSING": 1.0, "Flag": 0.45, "Revise": 0.25, "Invalidate": 0.0, "Discard": 0.0},
    "Flag": {"Flag": 1.0, "Revise": 0.65, "Invalidate": 0.55, "Keep": 0.35, "MISSING": 0.0, "Discard": 0.0},
}


def action_quality(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected = stale_memory_expected_ops(case)
    if not expected:
        return 1.0
    scores = []
    for exp in expected:
        actual = actual_op_for_target(pred, str(exp.get("target")))
        actual_op = str(actual.get("op")) if actual else "MISSING"
        scores.append(ACTION_QUALITY.get(str(exp.get("op")), {}).get(actual_op, 0.0))
    return sum(scores) / len(scores)


def stale_memory_action_accuracy(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected = stale_memory_expected_ops(case)
    if not expected:
        return 1.0
    actual = pred.get("memory_ops") or []
    hits = 0
    for exp in expected:
        if exp.get("op") == "Keep" and actual_op_for_target(pred, str(exp.get("target"))) is None:
            hits += 1
        elif any(memory_op_matches(exp, op) for op in actual):
            hits += 1
    return hits / len(expected)


def revision_accuracy(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected = [op for op in stale_memory_expected_ops(case) if op.get("op") == "Revise"]
    if not expected:
        return 1.0
    actual = pred.get("memory_ops") or []
    hits = 0
    for exp in expected:
        for op in actual:
            if memory_op_matches(exp, op) and op.get("revised_memory_id") and op.get("revised_claim"):
                hits += 1
                break
    return hits / len(expected)


def fact_preservation_soft_for_expected(expected: dict[str, Any], actual: dict[str, Any] | None) -> float:
    expected_facts = [str(f).strip() for f in expected.get("preserved_facts", []) if str(f).strip()]
    if not expected_facts:
        return 1.0
    if actual is None:
        return 0.0
    actual_facts = [str(f).strip() for f in actual.get("preserved_facts", []) if str(f).strip()]
    if not actual_facts:
        return 0.0
    scores = []
    for expected_fact in expected_facts:
        scores.append(max(token_f1(expected_fact, actual_fact) for actual_fact in actual_facts))
    return sum(scores) / len(scores)


def fact_preservation_soft(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected = [
        op
        for op in stale_memory_expected_ops(case)
        if op.get("should_preserve_fact") and op.get("preserved_facts")
    ]
    if not expected:
        return 1.0
    scores = [
        fact_preservation_soft_for_expected(exp, actual_op_for_target(pred, str(exp.get("target"))))
        for exp in expected
    ]
    return sum(scores) / len(scores)


def revision_quality(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected = [op for op in stale_memory_expected_ops(case) if op.get("op") == "Revise"]
    if not expected:
        return 1.0
    scores = []
    for exp in expected:
        actual = actual_op_for_target(pred, str(exp.get("target")))
        if actual is None:
            scores.append(0.0)
            continue
        preserved_soft = fact_preservation_soft_for_expected(exp, actual)
        score = 0.0
        if actual.get("op") == "Revise":
            score += 0.4
        if actual.get("revised_memory_id"):
            score += 0.2
        if actual.get("revised_claim") or actual.get("revision_note"):
            score += 0.2
        if preserved_soft >= 0.5:
            score += 0.2
        scores.append(score)
    return sum(scores) / len(scores)


def fact_preservation_rate(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected = [
        op
        for op in stale_memory_expected_ops(case)
        if op.get("should_preserve_fact") and op.get("preserved_facts")
    ]
    if not expected:
        return 1.0
    actual = pred.get("memory_ops") or []
    hits = 0
    for exp in expected:
        expected_facts = {str(f).strip().lower() for f in exp.get("preserved_facts", []) if str(f).strip()}
        for op in actual:
            if not memory_op_matches(exp, op):
                continue
            actual_facts = {str(f).strip().lower() for f in op.get("preserved_facts", []) if str(f).strip()}
            if expected_facts and expected_facts.issubset(actual_facts):
                hits += 1
                break
    return hits / len(expected)


def over_deletion_rate(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected_revisions = [op for op in stale_memory_expected_ops(case) if op.get("op") == "Revise"]
    if not expected_revisions:
        return 0.0
    actual = pred.get("memory_ops") or []
    over_deleted = 0
    for exp in expected_revisions:
        if any(
            op.get("op") in {"Discard", "Invalidate"} and str(op.get("target")) == str(exp.get("target"))
            for op in actual
        ):
            over_deleted += 1
    return over_deleted / len(expected_revisions)


def stale_memory_challenge_score(case: dict[str, Any], pred: dict[str, Any]) -> float:
    action = stale_memory_action_accuracy(case, pred)
    revision = revision_accuracy(case, pred)
    preservation = fact_preservation_rate(case, pred)
    over_delete = over_deletion_rate(case, pred)
    return 0.4 * action + 0.3 * revision + 0.2 * preservation + 0.1 * (1 - over_delete)


def memory_pollution_control_score(case: dict[str, Any], pred: dict[str, Any]) -> float:
    expected = stale_memory_expected_ops(case)
    if expected and not pred.get("memory_ops") and any(op.get("op") != "Keep" for op in expected):
        return 0.0
    action = action_quality(case, pred)
    revision = revision_quality(case, pred)
    preservation = fact_preservation_soft(case, pred)
    over_delete = over_deletion_rate(case, pred)
    if action == 0 and revision == 0 and preservation == 0:
        return 0.0
    return 0.35 * action + 0.25 * revision + 0.25 * preservation + 0.15 * (1 - over_delete)


def memory_op_scoring_applicable(pred: dict[str, Any]) -> bool:
    method = str(pred.get("method", ""))
    return bool(pred.get("memory_ops")) or is_medimem_method(method)


def counterfactual_metrics(pred: dict[str, Any]) -> dict[str, float]:
    verification = pred.get("counterfactual_verification") or {}
    if not verification.get("enabled", False):
        return {
            "counterfactual_probability_gap": 0.0,
            "counterfactual_pass": 0.0,
            "counterfactual_robustness_score": 0.0,
            "counterfactual_revision": 0.0,
            "counterfactual_evaluated": 0.0,
            "counterfactual_skipped": 1.0 if verification.get("policy") == "risk_sample" else 0.0,
        }
    threshold = float(verification.get("threshold") or COUNTERFACTUAL_CPG_THRESHOLD)
    cpg = max(0.0, float(verification.get("cpg") or 0.0))
    score = min(cpg / threshold, 1.0) if threshold > 0 else 0.0
    return {
        "counterfactual_probability_gap": cpg,
        "counterfactual_pass": 1.0 if verification.get("passed") else 0.0,
        "counterfactual_robustness_score": score,
        "counterfactual_revision": 1.0 if pred.get("counterfactual_revision_triggered") else 0.0,
        "counterfactual_evaluated": 1.0,
        "counterfactual_skipped": 0.0,
    }


def counterfactual_score(case: dict[str, Any], pred: dict[str, Any]) -> float:
    # Backward-compatible field name: now backed by true CPG, not the V0 evidence-overlap proxy.
    if not case.get("counterfactuals"):
        return 0.0
    return counterfactual_metrics(pred)["counterfactual_robustness_score"]


def diagnosis_metrics_applicable(case: dict[str, Any]) -> bool:
    flags = case.get("data_quality_flags") or {}
    if flags.get("diagnosis_metric_applicable") is False:
        return False
    primary = str((case.get("labels") or {}).get("primary_diagnosis") or "").strip()
    if not primary:
        return False
    if len(primary) > 120 or len(primary.split()) > 16:
        return False
    return True


def answer_entity_f1(case: dict[str, Any], pred: dict[str, Any]) -> float | None:
    labels = case.get("labels") or {}
    golds = [str(labels.get("primary_diagnosis", ""))]
    golds.extend(str(item) for item in labels.get("label_aliases", []) if str(item).strip())
    golds = [gold for gold in golds if gold.strip()]
    if not golds:
        return None
    items = [str(pred.get("primary_diagnosis") or "")]
    items.extend(str(item) for item in pred.get("diagnosis_list", []) if str(item).strip())
    items.extend(str(item) for item in pred.get("evidence", []) if str(item).strip())
    items = [item for item in items if item.strip()]
    if not items:
        return 0.0
    return max(token_f1(item, gold) for item in items for gold in golds)


def evaluate_predictions(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    case_by_id = {case["case_id"]: case for case in cases}
    rows = []
    for pred in predictions:
        case = case_by_id[pred["case_id"]]
        labels = case["labels"]
        primary_golds = [str(labels.get("primary_diagnosis", ""))]
        primary_golds.extend(str(item) for item in labels.get("label_aliases", []) if str(item).strip())
        diagnosis_applicable = diagnosis_metrics_applicable(case)
        primary_ok = diagnosis_match_any(pred.get("primary_diagnosis", ""), primary_golds) if diagnosis_applicable else None
        gold_diagnosis_list = list(labels.get("diagnosis_list", []))
        diag_f1 = (
            list_f1_with_aliases(pred.get("diagnosis_list", []), gold_diagnosis_list, list(labels.get("label_aliases", [])))
            if diagnosis_applicable
            else None
        )
        diagnosis_items = [pred.get("primary_diagnosis", "")] + [str(item) for item in pred.get("diagnosis_list", [])]
        cdr_f1s = []
        for qa in case.get("qa_tasks", []):
            if qa.get("type") != "CDR":
                continue
            answer = qa.get("answer", "")
            cdr_f1s.append(max(token_f1(item, answer) for item in diagnosis_items) if diagnosis_items else 0.0)
        cdr_value = (sum(cdr_f1s) / len(cdr_f1s) if cdr_f1s else 0.0) if diagnosis_applicable else None
        score_memory_ops = memory_op_scoring_applicable(pred)
        cf = counterfactual_metrics(pred)
        rows.append(
            {
                "case_id": pred["case_id"],
                "method": pred["method"],
                "replicate_id": pred.get("replicate_id", "default"),
                "completion_token_budget": pred.get("completion_token_budget"),
                "task_profile": case_task_profile(case),
                "diagnosis_metric_applicable": 1.0 if diagnosis_applicable else 0.0,
                "primary_correct": (1.0 if primary_ok else 0.0) if primary_ok is not None else None,
                "diagnosis_f1": diag_f1,
                "cdr_f1": cdr_value,
                "answer_entity_f1": answer_entity_f1(case, pred) if case_task_profile(case) == MEDICAL_ANSWER_ENTITY else None,
                "hard_pollution_suppression": hard_pollution_suppression(case, pred) if score_memory_ops else None,
                "pollution_suppression": pollution_suppression(case, pred) if score_memory_ops else None,
                "stale_memory_action_accuracy": stale_memory_action_accuracy(case, pred) if score_memory_ops else None,
                "revision_accuracy": revision_accuracy(case, pred) if score_memory_ops else None,
                "fact_preservation_rate": fact_preservation_rate(case, pred) if score_memory_ops else None,
                "over_deletion_rate": over_deletion_rate(case, pred) if score_memory_ops else None,
                "stale_memory_challenge_score": stale_memory_challenge_score(case, pred) if score_memory_ops else None,
                "action_quality": action_quality(case, pred) if score_memory_ops else None,
                "revision_quality": revision_quality(case, pred) if score_memory_ops else None,
                "fact_preservation_soft": fact_preservation_soft(case, pred) if score_memory_ops else None,
                "memory_pollution_control_score": memory_pollution_control_score(case, pred) if score_memory_ops else None,
                "counterfactual_score": counterfactual_score(case, pred),
                "counterfactual_probability_gap": cf["counterfactual_probability_gap"],
                "counterfactual_pass": cf["counterfactual_pass"],
                "counterfactual_robustness_score": cf["counterfactual_robustness_score"],
                "counterfactual_revision": cf["counterfactual_revision"],
                "counterfactual_evaluated": cf["counterfactual_evaluated"],
                "counterfactual_skipped": cf["counterfactual_skipped"],
                "tokens": float((pred.get("usage") or {}).get("total_tokens", 0) or 0),
                "prompt_tokens": float((pred.get("usage") or {}).get("prompt_tokens", 0) or 0),
                "completion_tokens": float((pred.get("usage") or {}).get("completion_tokens", 0) or 0),
                "model_calls": float((pred.get("usage") or {}).get("calls", 0) or 0),
                "latency_ms": float((pred.get("usage") or {}).get("latency_ms", 0) or 0),
            }
        )
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    summaries = [summarize_case_rows(method, vals) for method, vals in sorted(by_method.items())]
    return {"case_rows": rows, "summary": summaries}


def summarize_case_rows(method: str, vals: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(vals)
    primary_acc = avg(vals, "primary_correct")
    diagnosis_f1 = avg(vals, "diagnosis_f1")
    return {
        "method": method,
        "n": n,
        "diagnosis_metric_coverage": avg(vals, "diagnosis_metric_applicable"),
        "primary_diagnosis_top1_accuracy": primary_acc,
        "diagnosis_list_f1": diagnosis_f1,
        "answer_entity_f1": avg(vals, "answer_entity_f1"),
        "primary_diag_objective": (
            0.65 * primary_acc + 0.35 * diagnosis_f1 if primary_acc is not None and diagnosis_f1 is not None else None
        ),
        "cdr_f1": avg(vals, "cdr_f1"),
        "hard_pollution_suppression": avg(vals, "hard_pollution_suppression"),
        "memory_pollution_suppression": avg(vals, "pollution_suppression"),
        "stale_memory_action_accuracy": avg(vals, "stale_memory_action_accuracy"),
        "revision_accuracy": avg(vals, "revision_accuracy"),
        "fact_preservation_rate": avg(vals, "fact_preservation_rate"),
        "over_deletion_rate": avg(vals, "over_deletion_rate"),
        "stale_memory_challenge_score": avg(vals, "stale_memory_challenge_score"),
        "action_quality": avg(vals, "action_quality"),
        "revision_quality": avg(vals, "revision_quality"),
        "fact_preservation_soft": avg(vals, "fact_preservation_soft"),
        "memory_pollution_control_score": avg(vals, "memory_pollution_control_score"),
        "safety_efficiency_score": safety_efficiency_score(avg(vals, "memory_pollution_control_score"), avg(vals, "tokens")),
        "counterfactual_probability_gap": avg(vals, "counterfactual_probability_gap"),
        "counterfactual_pass_rate": avg(vals, "counterfactual_pass"),
        "counterfactual_robustness_score": avg(vals, "counterfactual_robustness_score"),
        "counterfactual_revision_rate": avg(vals, "counterfactual_revision"),
        "counterfactual_robustness_proxy": avg(vals, "counterfactual_score"),
        "counterfactual_coverage_rate": avg(vals, "counterfactual_evaluated"),
        "counterfactual_skipped_rate": avg(vals, "counterfactual_skipped"),
        "avg_tokens": avg(vals, "tokens"),
        "avg_prompt_tokens": avg(vals, "prompt_tokens"),
        "avg_completion_tokens": avg(vals, "completion_tokens"),
        "avg_model_calls": avg(vals, "model_calls"),
        "avg_latency_s": (avg(vals, "latency_ms") or 0.0) / 1000.0,
    }


def add_merged_ours_summaries(
    eval_result: dict[str, Any],
    *,
    group_names: list[str] | None = None,
) -> dict[str, Any]:
    """Add aggregate rows across dynamic top_k variants, e.g. full_medimem_topk*."""
    case_rows = list(eval_result.get("case_rows", []))
    summaries = list(eval_result.get("summary", []))
    groups = group_names or ["full"]
    existing = {str(row.get("method")) for row in summaries}
    for group in groups:
        prefix = f"{group}_medimem_"
        method = f"{group}_medimem_merged"
        if method in existing:
            continue
        vals = [row for row in case_rows if str(row.get("method", "")).startswith(prefix)]
        if vals:
            summaries.append(summarize_case_rows(method, vals))
            existing.add(method)
    for group in groups:
        prefix = f"{group}_ours_"
        method = f"{group}_ours_merged"
        if method in existing:
            continue
        vals = [row for row in case_rows if str(row.get("method", "")).startswith(prefix)]
        if vals:
            summaries.append(summarize_case_rows(method, vals))
            existing.add(method)
    if "medimem_merged" not in existing:
        vals = [
            row
            for row in case_rows
            if str(row.get("method", "")).startswith("medimem_")
            or str(row.get("method", "")).startswith("round_medimem_")
        ]
        if vals:
            summaries.append(summarize_case_rows("medimem_merged", vals))
            existing.add("medimem_merged")
    if "ours_merged" not in existing:
        vals = [
            row
            for row in case_rows
            if str(row.get("method", "")).startswith("ours_")
            or str(row.get("method", "")).startswith("round_ours_")
        ]
        if vals:
            summaries.append(summarize_case_rows("ours_merged", vals))
    updated = dict(eval_result)
    updated["summary"] = summaries
    return updated


def source_dataset_for_case(case: dict[str, Any]) -> str:
    flags = case.get("data_quality_flags") or {}
    source = str(flags.get("source_dataset") or "").strip()
    if source:
        return source
    for ref in case.get("source_refs") or []:
        if str(ref.get("role") or "") == "primary_source" and ref.get("dataset"):
            return str(ref.get("dataset"))
    for ref in case.get("source_refs") or []:
        if ref.get("dataset"):
            return str(ref.get("dataset"))
    return "unknown"


def build_source_metrics(
    cases: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    *,
    dataset: str = "medical_pooled",
    group_names: list[str] | None = None,
    include_overall: bool = True,
) -> list[dict[str, Any]]:
    """Summarize pooled medical rows by their original source_dataset."""
    case_source = {str(case.get("case_id")): source_dataset_for_case(case) for case in cases}
    source_order: list[str] = []
    for case in cases:
        source = source_dataset_for_case(case)
        if source not in source_order:
            source_order.append(source)
    if include_overall:
        source_order = ["overall"] + source_order

    rows: list[dict[str, Any]] = []
    for source in source_order:
        if source == "overall":
            vals = list(case_rows)
        else:
            vals = [row for row in case_rows if case_source.get(str(row.get("case_id"))) == source]
        if not vals:
            continue
        by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in vals:
            by_method[str(row.get("method"))].append(row)
        eval_result = {"case_rows": vals, "summary": [summarize_case_rows(method, rows_) for method, rows_ in sorted(by_method.items())]}
        eval_result = add_merged_ours_summaries(eval_result, group_names=group_names)
        for summary in eval_result["summary"]:
            rows.append({"dataset": dataset, "source": source, **summary})
    return rows


def build_slice_breakdown(
    cases: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    *,
    method_prefixes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    case_by_id = {str(case.get("case_id")): case for case in cases}
    prefixes = method_prefixes or {
        "full_medimem_merged": "full_medimem_",
        "full_ours_merged": "full_ours_",
        "baseline_amem_adapter": "baseline_amem_adapter",
        "baseline_polluted_amem_adapter": "baseline_polluted_amem_adapter",
    }
    slices = build_case_slices(cases)
    rows: list[dict[str, Any]] = []
    for slice_name, case_ids in slices.items():
        case_id_set = set(case_ids)
        for display_method, prefix in prefixes.items():
            vals = [
                row
                for row in case_rows
                if str(row.get("case_id")) in case_id_set
                and (
                    str(row.get("method")) == prefix
                    or str(row.get("method", "")).startswith(prefix)
                )
            ]
            if not vals:
                continue
            summary = summarize_case_rows(display_method, vals)
            summary["slice"] = slice_name
            rows.append(summary)
    return rows


def build_case_slices(cases: list[dict[str, Any]]) -> dict[str, list[str]]:
    slices: dict[str, list[str]] = {}
    ordered = sorted(cases, key=lambda case: str(case.get("case_id")))
    ids = [str(case.get("case_id")) for case in ordered]
    if ids:
        slices["all"] = ids
        slices["prefix_50"] = ids[:50]
        if len(ids) > 50:
            slices["after_50"] = ids[50:]
        slices["prefix_100"] = ids[:100]
    for case in ordered:
        cid = str(case.get("case_id"))
        flags = case.get("data_quality_flags") or {}
        event_count = int(flags.get("event_count") or len(case.get("events", [])) or 0)
        diagnosis_event_count = int(flags.get("diagnosis_event_count") or 0)
        slices.setdefault("high_complexity", [] if event_count < 60 else []).append(cid) if event_count >= 60 else None
        slices.setdefault("low_complexity", [] if event_count >= 60 else []).append(cid) if event_count < 60 else None
        slices.setdefault("high_diagnosis_events", [] if diagnosis_event_count < 3 else []).append(cid) if diagnosis_event_count >= 3 else None
        slices.setdefault("low_diagnosis_events", [] if diagnosis_event_count >= 3 else []).append(cid) if diagnosis_event_count < 3 else None
        if flags.get("has_follow_up"):
            slices.setdefault("has_follow_up", []).append(cid)
        else:
            slices.setdefault("no_follow_up", []).append(cid)
        if flags.get("label_fragment_like"):
            slices.setdefault("label_quality_fragment_like", []).append(cid)
        species = str(flags.get("species_context") or "").strip().lower()
        if species and species != "human":
            slices.setdefault(f"species_{species}", []).append(cid)
    for case in ordered:
        cid = str(case.get("case_id"))
        poison_types = {str(p.get("pollution_type")) for p in case.get("poison_records", []) if p.get("pollution_type")}
        for pollution_type in poison_types:
            slices.setdefault(f"pollution_{pollution_type}", []).append(cid)
    return {name: case_ids for name, case_ids in slices.items() if case_ids}


def avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


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


def safety_efficiency_score(mpcs: float | None, avg_tokens: float | None) -> float | None:
    if mpcs is None or avg_tokens is None:
        return None
    if avg_tokens <= 0 or mpcs <= 0:
        return 0.0
    return mpcs / (avg_tokens / 1000.0)


def _contains(text: Any, needle: str) -> bool:
    return bool(needle) and needle.lower() in str(text).lower()


def _audit_norm(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def _audit_context(text: str, needle: str, *, width: int = 120) -> str:
    low = text.lower()
    idx = low.find(needle.lower())
    if idx < 0:
        return str(text or "")[: width * 2]
    start = max(0, idx - width)
    end = min(len(text), idx + len(needle) + width)
    return text[start:end]


def _is_prompt_guard_language_false_positive(alias: str, context: str) -> bool:
    alias_norm = _audit_norm(alias)
    if alias_norm not in {"diagnosis", "diagnoses", "diagnostic", "interpretation"}:
        return False
    context_norm = _audit_norm(context)
    guard_patterns = (
        r"\bdo not use (?:the )?prior interpretation as (?:a )?diagnosis\b",
        r"\bdo not treat (?:a )?prior interpretation as (?:a )?diagnosis\b",
        r"\bunsafe for direct diagnosis\b",
        r"\bprior interpretation as (?:a )?diagnosis\b",
        r"\bstale interpretation as (?:a )?diagnosis\b",
    )
    return any(re.search(pattern, context_norm) for pattern in guard_patterns)


def _audit_aliases(case: dict[str, Any]) -> list[str]:
    labels = case.get("labels") or {}
    aliases = [labels.get("primary_diagnosis")] + list(labels.get("label_aliases") or [])
    out: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        text = str(alias or "").strip()
        key = _audit_norm(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _audit_source_type(case: dict[str, Any]) -> str:
    flags = case.get("data_quality_flags") or {}
    return str(flags.get("source_type") or "").strip()


def _audit_source_dataset(case: dict[str, Any]) -> str:
    return source_dataset_for_case(case)


def _audit_option_norms(case: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for item in case.get("answer_options") or []:
        out.add(_audit_norm(item.get("label")))
        out.add(_audit_norm(item.get("text")))
    return {item for item in out if item}


def _audit_verdict(case: dict[str, Any], field: str, alias: str, kind: str, context: str = "") -> str:
    alias_norm = _audit_norm(alias)
    source_type = _audit_source_type(case)
    if kind in {"runtime_gold_marker", "primary_selection_source", "source_real_false", "forbidden_visible_field"}:
        return "critical"
    if len(alias_norm) < 4:
        return "short_label_false_positive"
    if source_type == "medical_mcqa" and alias_norm in _audit_option_norms(case):
        return "benign_visible_option"
    if field in {"runtime_visible", "counterfactual_intervention"} and source_type == "medical_instruction":
        question_or_title_text = " ".join(
            str(event.get("text") or "")
            for event in case.get("events", [])
            if re.search(
                r"\b(?:Clinical question or presentation|Initial clinical task/source title)\b",
                str(event.get("text") or ""),
                flags=re.I,
            )
        )
        if alias_norm in _audit_norm(question_or_title_text):
            return "benign_visible_question_topic"
    if field == "runtime_visible" and source_type in {"longitudinal_case", "patient_summary"}:
        return "benign_visible_source_text"
    if field == "counterfactual_intervention" and source_type in {"longitudinal_case", "patient_summary"}:
        return "benign_visible_source_text"
    if kind == "prediction_marker" and _audit_norm(alias) in {"correct answer", "correct option"}:
        return "benign_model_answer_phrase"
    if kind == "prediction_marker":
        return "needs_review"
    if field == "prompt_memory_ops":
        if _is_prompt_guard_language_false_positive(alias, context):
            return "benign_prompt_guard_language"
        return "needs_review"
    if field in {"runtime_visible", "counterfactual_intervention"}:
        return "needs_review"
    return "benign_visible_text"


def build_leakage_audit_details(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_by_id = {str(case.get("case_id")): case for case in cases}
    details: list[dict[str, Any]] = []
    marker_re = re.compile(
        r"\b(?:assessment entity candidate|reference answer evidence|correct answer|correct option|doctor assessment)\b",
        flags=re.I,
    )
    for case in cases:
        case_id = str(case.get("case_id"))
        source = _audit_source_dataset(case)
        runtime_visible = {
            "events": [{"type": event.get("type"), "text": event.get("text")} for event in case.get("events", [])],
            "memory_seed": case.get("memory_seed", []),
            "poison_records": [
                {
                    "text": poison.get("text"),
                    "supporting_evidence": poison.get("supporting_evidence", []),
                }
                for poison in case.get("poison_records", [])
            ],
        }
        runtime_text = json.dumps(runtime_visible, ensure_ascii=False)
        marker_match = marker_re.search(runtime_text)
        if marker_match:
            details.append(
                {
                    "case_id": case_id,
                    "source": source,
                    "method": "",
                    "field": "runtime_visible",
                    "kind": "runtime_gold_marker",
                    "alias": marker_match.group(0),
                    "verdict": "critical",
                    "context": _audit_context(runtime_text, marker_match.group(0)),
                }
            )
        if str(case.get("source_real", True)).lower() == "false":
            details.append(
                {
                    "case_id": case_id,
                    "source": source,
                    "method": "",
                    "field": "source_real",
                    "kind": "source_real_false",
                    "alias": "source_real=false",
                    "verdict": "critical",
                    "context": "",
                }
            )
        for alias in _audit_aliases(case):
            if _contains(runtime_text, alias):
                details.append(
                    {
                        "case_id": case_id,
                        "source": source,
                        "method": "",
                        "field": "runtime_visible",
                        "kind": "runtime_gold_mention",
                        "alias": alias,
                        "verdict": _audit_verdict(case, "runtime_visible", alias, "runtime_gold_mention"),
                        "context": _audit_context(runtime_text, alias),
                    }
                )
    for pred in predictions:
        case = case_by_id.get(str(pred.get("case_id")))
        if not case:
            continue
        case_id = str(case.get("case_id"))
        source = _audit_source_dataset(case)
        method = str(pred.get("method") or "")
        selection_source = str((pred.get("primary_selection_pass") or {}).get("source") or "")
        if selection_source in {"assessment_entity_candidate", "reference_answer_evidence"}:
            details.append(
                {
                    "case_id": case_id,
                    "source": source,
                    "method": method,
                    "field": "primary_selection_pass.source",
                    "kind": "primary_selection_source",
                    "alias": selection_source,
                    "verdict": "critical",
                    "context": json.dumps(pred.get("primary_selection_pass") or {}, ensure_ascii=False)[:240],
                }
            )
        pred_visible = json.dumps(
            {
                "diagnosis_candidates": pred.get("diagnosis_candidates"),
                "evidence": pred.get("evidence"),
                "reasoning_summary": pred.get("reasoning_summary"),
                "primary_selection_pass": pred.get("primary_selection_pass"),
                "counterfactual_verification": pred.get("counterfactual_verification"),
            },
            ensure_ascii=False,
        )
        for field_name in FORBIDDEN_VISIBLE_FIELDS:
            if re.search(rf'"{re.escape(field_name)}"\s*:', pred_visible):
                details.append(
                    {
                        "case_id": case_id,
                        "source": source,
                        "method": method,
                        "field": "prediction_visible",
                        "kind": "forbidden_visible_field",
                        "alias": field_name,
                        "verdict": "critical",
                        "context": _audit_context(pred_visible, field_name),
                    }
                )
        marker_match = marker_re.search(pred_visible)
        if marker_match:
            details.append(
                {
                    "case_id": case_id,
                    "source": source,
                    "method": method,
                    "field": "prediction_visible",
                    "kind": "prediction_marker",
                    "alias": marker_match.group(0),
                    "verdict": _audit_verdict(case, "prediction_visible", marker_match.group(0), "prediction_marker"),
                    "context": _audit_context(pred_visible, marker_match.group(0)),
                }
            )
        prompt_ops = pred.get("prompt_memory_ops")
        prompt_text = json.dumps(prompt_ops, ensure_ascii=False) if prompt_ops is not None else ""
        for field_name in FORBIDDEN_VISIBLE_FIELDS:
            if prompt_text and re.search(rf'"{re.escape(field_name)}"\s*:', prompt_text):
                details.append(
                    {
                        "case_id": case_id,
                        "source": source,
                        "method": method,
                        "field": "prompt_memory_ops",
                        "kind": "forbidden_visible_field",
                        "alias": field_name,
                        "verdict": "critical",
                        "context": _audit_context(prompt_text, field_name),
                    }
                )
        verification = pred.get("counterfactual_verification") or {}
        counterfactual_text = json.dumps({"intervention": verification.get("intervention")}, ensure_ascii=False)
        for alias in _audit_aliases(case):
            if prompt_text and _contains(prompt_text, alias):
                context = _audit_context(prompt_text, alias)
                details.append(
                    {
                        "case_id": case_id,
                        "source": source,
                        "method": method,
                        "field": "prompt_memory_ops",
                        "kind": "prompt_memory_ops_gold_mention",
                        "alias": alias,
                        "verdict": _audit_verdict(
                            case,
                            "prompt_memory_ops",
                            alias,
                            "prompt_memory_ops_gold_mention",
                            context,
                        ),
                        "context": context,
                    }
                )
            if _contains(counterfactual_text, alias):
                details.append(
                    {
                        "case_id": case_id,
                        "source": source,
                        "method": method,
                        "field": "counterfactual_intervention",
                        "kind": "counterfactual_runtime_gold_mention",
                        "alias": alias,
                        "verdict": _audit_verdict(case, "counterfactual_intervention", alias, "counterfactual_runtime_gold_mention"),
                        "context": _audit_context(counterfactual_text, alias),
                    }
                )
    return details


def summarize_leakage_audit_details(details: list[dict[str, Any]]) -> dict[str, int]:
    verdict_counts = Counter(str(item.get("verdict") or "unknown") for item in details)
    kind_counts = Counter(str(item.get("kind") or "unknown") for item in details)
    return {
        "detail_count": len(details),
        "critical_leakage_count": int(verdict_counts.get("critical", 0)),
        "needs_review_count": int(verdict_counts.get("needs_review", 0)),
        "benign_visible_text_count": int(verdict_counts.get("benign_visible_text", 0)),
        "benign_visible_source_text_count": int(verdict_counts.get("benign_visible_source_text", 0)),
        "benign_visible_question_topic_count": int(verdict_counts.get("benign_visible_question_topic", 0)),
        "benign_visible_option_count": int(verdict_counts.get("benign_visible_option", 0)),
        "benign_model_answer_phrase_count": int(verdict_counts.get("benign_model_answer_phrase", 0)),
        "benign_prompt_guard_language_count": int(verdict_counts.get("benign_prompt_guard_language", 0)),
        "short_label_false_positive_count": int(verdict_counts.get("short_label_false_positive", 0)),
        "runtime_gold_mentions": int(kind_counts.get("runtime_gold_mention", 0)),
        "runtime_gold_markers": int(kind_counts.get("runtime_gold_marker", 0)),
        "prompt_memory_ops_gold_mentions": int(kind_counts.get("prompt_memory_ops_gold_mention", 0)),
        "counterfactual_runtime_gold_mentions": int(kind_counts.get("counterfactual_runtime_gold_mention", 0)),
        "leaked_primary_selection_sources": int(kind_counts.get("primary_selection_source", 0)),
        "leaked_prediction_candidate_mentions": int(kind_counts.get("prediction_marker", 0)),
        "forbidden_visible_field_mentions": int(kind_counts.get("forbidden_visible_field", 0)),
        "source_real_false": int(kind_counts.get("source_real_false", 0)),
    }


def build_leakage_audit(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, int]:
    detailed = summarize_leakage_audit_details(build_leakage_audit_details(cases, predictions))
    case_by_id = {case.get("case_id"): case for case in cases}
    runtime_gold_mentions = 0
    runtime_gold_markers = 0
    poison_expected_op = 0
    poison_revised_claim = 0
    poison_expected_memory_ops = 0
    source_real_false = 0
    prompt_memory_ops_gold_mentions = 0
    counterfactual_runtime_gold_mentions = 0
    leaked_primary_selection_sources = 0
    leaked_prediction_candidate_mentions = 0
    forbidden_visible_field_mentions = int(detailed.get("forbidden_visible_field_mentions", 0) or 0)
    marker_re = re.compile(
        r"\b(?:assessment entity candidate|reference answer evidence|correct answer|correct option|doctor assessment)\b",
        flags=re.I,
    )
    for case in cases:
        primary = str(case.get("labels", {}).get("primary_diagnosis") or "")
        runtime_visible = {
            "events": [{"type": event.get("type"), "text": event.get("text")} for event in case.get("events", [])],
            "memory_seed": case.get("memory_seed", []),
            "poison_records": [
                {
                    "text": poison.get("text"),
                    "supporting_evidence": poison.get("supporting_evidence", []),
                }
                for poison in case.get("poison_records", [])
            ],
        }
        runtime_text = json.dumps(runtime_visible, ensure_ascii=False)
        if _contains(runtime_text, primary):
            runtime_gold_mentions += 1
        if marker_re.search(runtime_text):
            runtime_gold_markers += 1
        if str(case.get("source_real", True)).lower() == "false":
            source_real_false += 1
        for poison in case.get("poison_records", []):
            poison_expected_op += int("expected_op" in poison)
            poison_revised_claim += int("revised_claim" in poison)
            poison_expected_memory_ops += int("expected_memory_ops" in poison)
    for pred in predictions:
        selection_source = str((pred.get("primary_selection_pass") or {}).get("source") or "")
        if selection_source in {"assessment_entity_candidate", "reference_answer_evidence"}:
            leaked_primary_selection_sources += 1
        pred_visible = json.dumps(
            {
                "diagnosis_candidates": pred.get("diagnosis_candidates"),
                "evidence": pred.get("evidence"),
                "reasoning_summary": pred.get("reasoning_summary"),
                "primary_selection_pass": pred.get("primary_selection_pass"),
                "counterfactual_verification": pred.get("counterfactual_verification"),
            },
            ensure_ascii=False,
        )
        if marker_re.search(pred_visible):
            leaked_prediction_candidate_mentions += 1
        prompt_ops = pred.get("prompt_memory_ops")
        if prompt_ops is None:
            continue
        case = case_by_id.get(pred.get("case_id"))
        if not case:
            continue
        primary = str(case.get("labels", {}).get("primary_diagnosis") or "")
        if _contains(json.dumps(prompt_ops, ensure_ascii=False), primary):
            prompt_memory_ops_gold_mentions += 1
        verification = pred.get("counterfactual_verification") or {}
        runtime_counterfactual = {
            "intervention": verification.get("intervention"),
        }
        if _contains(json.dumps(runtime_counterfactual, ensure_ascii=False), primary):
            counterfactual_runtime_gold_mentions += 1
    critical_leakage_count = runtime_gold_markers + leaked_primary_selection_sources + forbidden_visible_field_mentions
    return {
        "critical_leakage_count": critical_leakage_count,
        "needs_review_count": int(detailed.get("needs_review_count", 0) or 0),
        "runtime_gold_mentions": runtime_gold_mentions,
        "runtime_gold_markers": runtime_gold_markers,
        "prompt_memory_ops_gold_mentions": prompt_memory_ops_gold_mentions,
        "counterfactual_runtime_gold_mentions": counterfactual_runtime_gold_mentions,
        "leaked_primary_selection_sources": leaked_primary_selection_sources,
        "leaked_prediction_candidate_mentions": leaked_prediction_candidate_mentions,
        "forbidden_visible_field_mentions": forbidden_visible_field_mentions,
        "poison_expected_op": poison_expected_op,
        "poison_revised_claim": poison_revised_claim,
        "poison_expected_memory_ops": poison_expected_memory_ops,
        "source_real_false": source_real_false,
    }


def build_memory_op_confusion(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, int]:
    pred_by_case = {pred.get("case_id"): pred for pred in predictions if is_medimem_method(pred.get("method", ""))}
    confusion: dict[str, int] = defaultdict(int)
    for case in cases:
        pred = pred_by_case.get(case.get("case_id"))
        if not pred:
            continue
        for expected in stale_memory_expected_ops(case):
            expected_op = str(expected.get("op"))
            target = str(expected.get("target"))
            actual = actual_op_for_target(pred, target)
            actual_op = str(actual.get("op")) if actual else "MISSING"
            confusion[f"{expected_op}->{actual_op}"] += 1
    return dict(sorted(confusion.items()))


def build_pollution_type_breakdown(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    pred_by_case = {pred.get("case_id"): pred for pred in predictions if pred.get("memory_ops")}
    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)
    for case in cases:
        pred = pred_by_case.get(case.get("case_id"))
        if not pred:
            continue
        poison_by_id = {str(p.get("poison_id")): p for p in case.get("poison_records", [])}
        for expected in stale_memory_expected_ops(case):
            target = str(expected.get("target"))
            pollution_type = str(expected.get("memory_issue_type") or poison_by_id.get(target, {}).get("pollution_type") or "unknown")
            single_case = {"expected_memory_ops": [expected]}
            buckets[pollution_type].append(
                {
                    "action_quality": action_quality(single_case, pred),
                    "revision_quality": revision_quality(single_case, pred),
                    "fact_preservation_soft": fact_preservation_soft(single_case, pred),
                    "over_deletion_rate": over_deletion_rate(single_case, pred),
                    "memory_pollution_control_score": memory_pollution_control_score(single_case, pred),
                }
            )
    return {
        pollution_type: {
            "n": float(len(rows)),
            "action_quality": avg(rows, "action_quality"),
            "revision_quality": avg(rows, "revision_quality"),
            "fact_preservation_soft": avg(rows, "fact_preservation_soft"),
            "over_deletion_rate": avg(rows, "over_deletion_rate"),
            "memory_pollution_control_score": avg(rows, "memory_pollution_control_score"),
        }
        for pollution_type, rows in sorted(buckets.items())
    }


def build_expected_memory_op_distribution(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    poison_by_case = {
        case.get("case_id"): {str(p.get("poison_id")): p for p in case.get("poison_records", [])}
        for case in cases
    }
    distribution: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for case in cases:
        poison_by_id = poison_by_case.get(case.get("case_id"), {})
        for expected in stale_memory_expected_ops(case):
            target = str(expected.get("target"))
            pollution_type = str(expected.get("memory_issue_type") or poison_by_id.get(target, {}).get("pollution_type") or "unknown")
            distribution[pollution_type][str(expected.get("op"))] += 1
    return {
        pollution_type: dict(sorted(counts.items()))
        for pollution_type, counts in sorted(distribution.items())
    }


def best_baseline_accuracy(summaries: list[dict[str, Any]]) -> float:
    polluted_vals = [
        float(s["primary_diagnosis_top1_accuracy"])
        for s in summaries
        if str(s["method"]).startswith("baseline_polluted_") and s.get("primary_diagnosis_top1_accuracy") is not None
    ]
    if polluted_vals:
        return max(polluted_vals)
    vals = [
        float(s["primary_diagnosis_top1_accuracy"])
        for s in summaries
        if str(s["method"]).startswith("baseline_") and s.get("primary_diagnosis_top1_accuracy") is not None
    ]
    return max(vals) if vals else 0.0


METRIC_GATE_MAIN_COLUMNS = (
    "primary_diagnosis_top1_accuracy",
    "diagnosis_list_f1",
    "cdr_f1",
    "counterfactual_robustness_proxy",
)

METRIC_GATE_SAFETY_MINIMUMS = {
    "stale_memory_action_accuracy": 0.95,
    "revision_accuracy": 0.95,
}


def read_metrics_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_cell(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in {None, "", "N/A"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def amem_reference_value(rows: list[dict[str, Any]], metric: str) -> float | None:
    vals = [
        _float_cell(row, metric)
        for row in rows
        if str(row.get("method")) in {"baseline_amem_adapter", "baseline_polluted_amem_adapter"}
    ]
    vals = [val for val in vals if val is not None]
    return max(vals) if vals else None


def ours_gate_row(rows: list[dict[str, Any]], method: str | None = None) -> dict[str, Any] | None:
    if method:
        exact = [row for row in rows if str(row.get("method")) == method]
        if exact:
            return exact[0]
    merged = [row for row in rows if str(row.get("method")) == "full_medimem_merged"]
    if merged:
        return merged[0]
    merged = [row for row in rows if str(row.get("method")) == "full_ours_merged"]
    if merged:
        return merged[0]
    preferred = [
        row
        for row in rows
        if str(row.get("method", "")).startswith("full_medimem_")
        or str(row.get("method", "")).startswith("medimem_")
        or str(row.get("method", "")).startswith("full_ours_")
        or str(row.get("method", "")).startswith("ours_")
    ]
    if not preferred:
        return None
    return max(preferred, key=lambda row: _float_cell(row, "primary_diagnosis_top1_accuracy") or 0.0)


def build_metric_gate(
    rows: list[dict[str, Any]],
    *,
    method: str | None = None,
    relative_margin: float = 0.10,
    absolute_margin: float = 0.10,
) -> dict[str, Any]:
    ours = ours_gate_row(rows, method=method)
    if ours is None:
        return {"passed": False, "method": method, "failures": ["No ours method row found."], "checks": []}

    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    for metric in METRIC_GATE_MAIN_COLUMNS:
        ref = amem_reference_value(rows, metric)
        actual = _float_cell(ours, metric)
        if ref is None or actual is None:
            failures.append(f"{metric}: missing reference or actual value")
            checks.append({"metric": metric, "actual": actual, "reference": ref, "target": None, "passed": False})
            continue
        target = max(ref * (1 + relative_margin), ref + absolute_margin)
        passed = actual >= target
        if not passed:
            failures.append(f"{metric}: actual={actual:.3f} target={target:.3f} reference={ref:.3f}")
        checks.append({"metric": metric, "actual": actual, "reference": ref, "target": target, "passed": passed})

    for metric, target in METRIC_GATE_SAFETY_MINIMUMS.items():
        actual = _float_cell(ours, metric)
        passed = actual is not None and actual >= target
        if not passed:
            failures.append(f"{metric}: actual={actual} target={target:.3f}")
        checks.append({"metric": metric, "actual": actual, "reference": None, "target": target, "passed": passed})

    over_delete = _float_cell(ours, "over_deletion_rate")
    over_delete_passed = over_delete == 0.0
    if not over_delete_passed:
        failures.append(f"over_deletion_rate: actual={over_delete} target=0.000")
    checks.append({"metric": "over_deletion_rate", "actual": over_delete, "reference": None, "target": 0.0, "passed": over_delete_passed})

    return {
        "passed": not failures,
        "method": ours.get("method"),
        "failures": failures,
        "checks": checks,
    }


def render_metric_gate(gate: dict[str, Any]) -> str:
    lines = [f"method={gate.get('method')}", f"passed={gate.get('passed')}"]
    for check in gate.get("checks", []):
        target = check.get("target")
        reference = check.get("reference")
        target_text = "N/A" if target is None else f"{float(target):.3f}"
        ref_text = "N/A" if reference is None else f"{float(reference):.3f}"
        actual = check.get("actual")
        actual_text = "N/A" if actual is None else f"{float(actual):.3f}"
        lines.append(
            f"- {check.get('metric')}: actual={actual_text} reference={ref_text} target={target_text} passed={check.get('passed')}"
        )
    if gate.get("failures"):
        lines.append("failures:")
        lines.extend(f"- {failure}" for failure in gate["failures"])
    return "\n".join(lines)
