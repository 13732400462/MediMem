from __future__ import annotations

import json

from mem_ehr_agent.optimizer import write_fast_formal_gate


def _summaries() -> list[dict[str, object]]:
    methods = {
        "full_medimem_merged": 0.3,
        "direct_deepseek": 0.4,
        "baseline_single_cot_agent": 0.35,
        "baseline_static_rag": 0.45,
        "baseline_amem_adapter": 0.42,
        "baseline_ddo_adapter": 0.2,
        "baseline_colacare_adapter": 0.25,
    }
    return [{"method": method, "primary_diag_objective": value} for method, value in methods.items()]


def test_gate_checks_integrity_not_whether_full_method_wins(tmp_path) -> None:
    (tmp_path / "progress.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
    write_fast_formal_gate(tmp_path, summaries=_summaries(), leakage_audit={})
    gate = json.loads((tmp_path / "fast_formal_gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["full_beats_best_baseline"] is False
    assert gate["best_baseline_primary_diag_objective"] == 0.45


def test_gate_records_but_excludes_intentional_sanitization_ablation_leakage(tmp_path) -> None:
    (tmp_path / "progress.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
    write_fast_formal_gate(
        tmp_path,
        summaries=_summaries(),
        leakage_audit={"critical_leakage_count": 4, "needs_review_count": 1},
        intentional_leakage_ablation=True,
    )
    gate = json.loads((tmp_path / "fast_formal_gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["critical_leakage_count"] == 0
    assert gate["intentional_ablation_critical_leakage_count"] == 4
