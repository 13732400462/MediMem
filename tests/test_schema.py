from mem_ehr_agent.data_builder import build_case
from mem_ehr_agent.data_sources import fallback_pmc_rows, fallback_pmoa_rows
from mem_ehr_agent.schemas import validate_case


def test_generated_case_schema():
    case = build_case(fallback_pmoa_rows()[0], fallback_pmc_rows()[0], 1)
    assert validate_case(case) == []

