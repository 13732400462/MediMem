from mem_ehr_agent.error_analysis import build_error_analysis, render_error_analysis


def test_error_analysis_buckets_and_render(tmp_path):
    cases = [
        {
            "case_id": "c1",
            "labels": {"primary_diagnosis": "SLE", "diagnosis_list": ["Systemic lupus erythematosus"]},
            "qa_tasks": [{"type": "CDR", "answer": "Systemic lupus erythematosus"}],
        }
    ]
    predictions = [
        {
            "case_id": "c1",
            "method": "ours_topk3_round1",
            "primary_diagnosis": "viral syndrome",
            "diagnosis_list": ["viral syndrome"],
        },
        {
            "case_id": "c1",
            "method": "baseline_amem_adapter",
            "primary_diagnosis": "Systemic lupus erythematosus",
            "diagnosis_list": ["Systemic lupus erythematosus"],
        },
    ]
    analysis = build_error_analysis(cases, predictions)
    assert len(analysis["buckets"]["ours_wrong_amem_right"]) == 1
    text = render_error_analysis(tmp_path, analysis)
    assert "Ours 错 / A-MEM 对" in text
    assert (tmp_path / "error_analysis_zh.md").exists()
