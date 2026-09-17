from __future__ import annotations

import json

from medimem.statistical_analysis import (
    canonical_method,
    holm_adjust,
    paired_bootstrap_ci,
    paired_permutation_p,
    percentile,
)


def test_canonical_method_merges_dynamic_medimem_variants() -> None:
    assert canonical_method("full_medimem_topk8_round1") == "full_medimem"
    assert canonical_method("ablate_no_memory_cleaning_medimem_topk3_round1") == "ablate_no_memory_cleaning"
    assert canonical_method("baseline_amem_adapter") == "baseline_amem_adapter"


def test_percentile_and_bootstrap_are_deterministic() -> None:
    assert percentile([0.0, 1.0], 0.5) == 0.5
    observed, low, high = paired_bootstrap_ci([1.0, 1.0], [0.0, 0.0], resamples=100, seed=7)
    assert (observed, low, high) == (1.0, 1.0, 1.0)
    assert paired_permutation_p([1.0] * 8, [0.0] * 8, resamples=1000, seed=7) < 0.02


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted[0] == 0.03
    assert adjusted[2] >= adjusted[0]
    assert adjusted[1] >= adjusted[2]

