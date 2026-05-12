from mem_ehr_agent.metrics import diagnosis_match, list_f1, token_f1


def test_token_f1_overlap():
    assert token_f1("acute myeloid leukemia", "myeloid leukemia") > 0.7


def test_diagnosis_match_substring():
    assert diagnosis_match("non small cell lung cancer", "Non-small-cell lung cancer (NSCLC)")


def test_list_f1():
    score = list_f1(["acute myeloid leukemia", "anemia"], ["AML", "acute myeloid leukemia"])
    assert score > 0.4

