from mem_ehr_agent.report import render_report


def test_report_renders_na_and_pollution_breakdown(tmp_path):
    text = render_report(
        run_dir=tmp_path,
        data_notes="unit test",
        summaries=[
            {
                "method": "baseline_polluted_amem_adapter",
                "n": 1,
                "primary_diagnosis_top1_accuracy": 0.5,
                "diagnosis_list_f1": 0.4,
                "cdr_f1": 0.3,
                "hard_pollution_suppression": None,
                "action_quality": None,
                "revision_quality": None,
                "fact_preservation_soft": None,
                "over_deletion_rate": None,
                "memory_pollution_control_score": None,
                "safety_efficiency_score": None,
                "counterfactual_robustness_proxy": 0.1,
                "avg_tokens": 100,
            }
        ],
        optimization_log=[],
        best_round=None,
        expected_op_distribution={"valid_historical_fact": {"Keep": 1}},
        pollution_type_breakdown={
            "valid_historical_fact": {
                "n": 1,
                "action_quality": 1.0,
                "revision_quality": 1.0,
                "fact_preservation_soft": 0.0,
                "over_deletion_rate": 0.0,
                "memory_pollution_control_score": 0.75,
            }
        },
    )
    assert "Hard Hit" in text
    assert "| baseline_polluted_amem_adapter | 1 | 0.500 | 0.400 | 0.300 |" in text
    assert "N/A" in text
    assert "## 预期操作分布" in text
    assert "valid_historical_fact | Keep=1" in text
    assert "## 污染类型分解" in text
