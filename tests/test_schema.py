from medimem.data_builder import build_case
from medimem.data_sources import fallback_pmc_rows, fallback_pmoa_rows
from medimem.schemas import validate_case


def test_generated_case_schema():
    case = build_case(fallback_pmoa_rows()[0], fallback_pmc_rows()[0], 1)
    assert validate_case(case) == []


