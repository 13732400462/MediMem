from medimem.data_builder import build_case, is_fragment_like_label, write_prefix_slices
from medimem.data_sources import fallback_pmc_rows, fallback_pmoa_rows


def _norm(text):
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def test_stale_poison_records_are_linked_to_timeline():
    case = build_case(fallback_pmoa_rows()[0], fallback_pmc_rows()[0], 1)
    assert len(case["poison_records"]) >= 2
    for poison in case["poison_records"]:
        assert poison["source_event_id"]
        assert poison["supporting_evidence"]
        assert poison["pollution_type"]
        assert poison["staleness_type"]
        assert poison["valid_time_scope"]
        assert poison["claim_type"]
        assert "expected_op" not in poison
        assert "revised_claim" not in poison


def test_stale_expected_ops_mix_revision_invalidation_and_discard():
    case = build_case(fallback_pmoa_rows()[0], fallback_pmc_rows()[0], 1)
    ops = {op["op"] for op in case["expected_memory_ops"]}
    assert {"Revise", "Invalidate", "Discard"}.issubset(ops)
    assert any(op.get("should_preserve_fact") for op in case["expected_memory_ops"] if op["op"] == "Revise")


def test_expected_ops_are_not_fixed_across_cases():
    pmoa_rows = fallback_pmoa_rows()
    pmc_rows = fallback_pmc_rows()
    cases = [
        build_case(pmoa_rows[idx % len(pmoa_rows)], pmc_rows[idx % len(pmc_rows)], idx + 1)
        for idx in range(5)
    ]
    op_sets = {tuple(op["op"] for op in case["expected_memory_ops"] if op["op"] != "Write") for case in cases}
    all_ops = {op["op"] for case in cases for op in case["expected_memory_ops"]}
    assert len(op_sets) > 1
    assert {"Keep", "Flag", "Revise", "Invalidate", "Discard"}.issubset(all_ops)
    assert any(
        op.get("quality_checks", {}).get("avoid_diagnosis_leakage")
        for case in cases
        for op in case["expected_memory_ops"]
        if op["op"] == "Revise"
    )


def test_partial_truth_pollution_preserves_true_fact_but_revises_interpretation():
    case = build_case(fallback_pmoa_rows()[0], fallback_pmc_rows()[0], 1)
    partial = [p for p in case["poison_records"] if p["pollution_type"] == "partial_truth_misleading_memory"]
    assert partial
    poison = partial[0]
    assert "Misleading conclusion" in poison["text"]
    assert "rather than" not in poison["text"]


def test_runtime_poison_and_seed_do_not_directly_leak_primary_label():
    case = build_case(fallback_pmoa_rows()[0], fallback_pmc_rows()[0], 1)
    gold = _norm(case["labels"]["primary_diagnosis"])
    runtime_text = " ".join(
        [m.get("summary", "") for m in case.get("memory_seed", [])]
        + [p.get("text", "") for p in case.get("poison_records", [])]
        + [" ".join(p.get("supporting_evidence", [])) for p in case.get("poison_records", [])]
    )
    assert gold not in _norm(runtime_text)


def test_prefix_slices_are_prefix_stable(tmp_path):
    cases = [
        build_case(
            fallback_pmoa_rows()[idx % len(fallback_pmoa_rows())],
            fallback_pmc_rows()[idx % len(fallback_pmc_rows())],
            idx + 1,
        )
        for idx in range(8)
    ]
    paths = write_prefix_slices(cases, output_dir=tmp_path, stem="pool", sizes=[3, 5])
    small = paths[0].read_text(encoding="utf-8").splitlines()
    large = paths[1].read_text(encoding="utf-8").splitlines()
    assert small == large[:3]
    assert "prefix_stable=true" in paths[0].with_suffix(".notes.txt").read_text(encoding="utf-8")


def test_fragment_like_label_flagger():
    assert is_fragment_like_label("by ITS sequencing")
    assert is_fragment_like_label("no personal history of breast cancer")
    assert not is_fragment_like_label("Systemic lupus erythematosus")

