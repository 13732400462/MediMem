from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir
from .medical_terms import canonicalize_diagnosis

COUNTERFACTUAL_CPG_THRESHOLD = 0.40


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
        if any(op.get("op") == exp.get("op") and str(op.get("target")) == str(exp.get("target")) for op in actual):
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
    return actual.get("op") == expected.get("op") and str(actual.get("target")) == str(expected.get("target"))


def actual_op_for_target(pred: dict[str, Any], target: str) -> dict[str, Any] | None:
    for op in pred.get("memory_ops") or []:
        if str(op.get("target")) == str(target):
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
        }
    threshold = float(verification.get("threshold") or COUNTERFACTUAL_CPG_THRESHOLD)
    cpg = max(0.0, float(verification.get("cpg") or 0.0))
    score = min(cpg / threshold, 1.0) if threshold > 0 else 0.0
    return {
        "counterfactual_probability_gap": cpg,
        "counterfactual_pass": 1.0 if verification.get("passed") else 0.0,
        "counterfactual_robustness_score": score,
        "counterfactual_revision": 1.0 if pred.get("counterfactual_revision_triggered") else 0.0,
    }


def counterfactual_score(case: dict[str, Any], pred: dict[str, Any]) -> float:
    # Backward-compatible field name: now backed by true CPG, not the V0 evidence-overlap proxy.
    if not case.get("counterfactuals"):
        return 0.0
    return counterfactual_metrics(pred)["counterfactual_robustness_score"]


def evaluate_predictions(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    case_by_id = {case["case_id"]: case for case in cases}
    rows = []
    for pred in predictions:
        case = case_by_id[pred["case_id"]]
        labels = case["labels"]
        primary_ok = diagnosis_match(pred.get("primary_diagnosis", ""), labels.get("primary_diagnosis", ""))
        diag_f1 = list_f1(pred.get("diagnosis_list", []), labels.get("diagnosis_list", []))
        diagnosis_items = [pred.get("primary_diagnosis", "")] + [str(item) for item in pred.get("diagnosis_list", [])]
        cdr_f1s = []
        for qa in case.get("qa_tasks", []):
            if qa.get("type") != "CDR":
                continue
            answer = qa.get("answer", "")
            cdr_f1s.append(max(token_f1(item, answer) for item in diagnosis_items) if diagnosis_items else 0.0)
        score_memory_ops = memory_op_scoring_applicable(pred)
        cf = counterfactual_metrics(pred)
        rows.append(
            {
                "case_id": pred["case_id"],
                "method": pred["method"],
                "primary_correct": 1.0 if primary_ok else 0.0,
                "diagnosis_f1": diag_f1,
                "cdr_f1": sum(cdr_f1s) / len(cdr_f1s) if cdr_f1s else 0.0,
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
                "tokens": float((pred.get("usage") or {}).get("total_tokens", 0) or 0),
            }
        )
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    summaries = [summarize_case_rows(method, vals) for method, vals in sorted(by_method.items())]
    return {"case_rows": rows, "summary": summaries}


def summarize_case_rows(method: str, vals: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(vals)
    return {
        "method": method,
        "n": n,
        "primary_diagnosis_top1_accuracy": avg(vals, "primary_correct"),
        "diagnosis_list_f1": avg(vals, "diagnosis_f1"),
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
        "avg_tokens": avg(vals, "tokens"),
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


def build_leakage_audit(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, int]:
    runtime_gold_mentions = 0
    poison_expected_op = 0
    poison_revised_claim = 0
    poison_expected_memory_ops = 0
    source_real_false = 0
    prompt_memory_ops_gold_mentions = 0
    counterfactual_runtime_gold_mentions = 0
    for case in cases:
        primary = str(case.get("labels", {}).get("primary_diagnosis") or "")
        runtime_visible = {
            "memory_seed": case.get("memory_seed", []),
            "poison_records": [
                {
                    "text": poison.get("text"),
                    "supporting_evidence": poison.get("supporting_evidence", []),
                }
                for poison in case.get("poison_records", [])
            ],
        }
        if _contains(json.dumps(runtime_visible, ensure_ascii=False), primary):
            runtime_gold_mentions += 1
        if str(case.get("source_real", True)).lower() == "false":
            source_real_false += 1
        for poison in case.get("poison_records", []):
            poison_expected_op += int("expected_op" in poison)
            poison_revised_claim += int("revised_claim" in poison)
            poison_expected_memory_ops += int("expected_memory_ops" in poison)
    for pred in predictions:
        prompt_ops = pred.get("prompt_memory_ops")
        if prompt_ops is None:
            continue
        case = next((c for c in cases if c.get("case_id") == pred.get("case_id")), None)
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
    return {
        "runtime_gold_mentions": runtime_gold_mentions,
        "prompt_memory_ops_gold_mentions": prompt_memory_ops_gold_mentions,
        "counterfactual_runtime_gold_mentions": counterfactual_runtime_gold_mentions,
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
        actual_by_target = {str(op.get("target")): str(op.get("op")) for op in pred.get("memory_ops", [])}
        for expected in stale_memory_expected_ops(case):
            expected_op = str(expected.get("op"))
            target = str(expected.get("target"))
            actual_op = actual_by_target.get(target, "MISSING")
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
            pollution_type = str(poison_by_id.get(target, {}).get("pollution_type") or "unknown")
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
            pollution_type = str(poison_by_id.get(target, {}).get("pollution_type") or "unknown")
            distribution[pollution_type][str(expected.get("op"))] += 1
    return {
        pollution_type: dict(sorted(counts.items()))
        for pollution_type, counts in sorted(distribution.items())
    }


def best_baseline_accuracy(summaries: list[dict[str, Any]]) -> float:
    polluted_vals = [
        float(s["primary_diagnosis_top1_accuracy"])
        for s in summaries
        if str(s["method"]).startswith("baseline_polluted_")
    ]
    if polluted_vals:
        return max(polluted_vals)
    vals = [
        float(s["primary_diagnosis_top1_accuracy"])
        for s in summaries
        if str(s["method"]).startswith("baseline_")
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
