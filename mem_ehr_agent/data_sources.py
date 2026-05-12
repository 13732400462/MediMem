from __future__ import annotations

import itertools
import random
from typing import Any

import requests


HF_DATASET_SERVER = "https://datasets-server.huggingface.co"


def hf_first_rows(dataset: str, *, split: str = "train", config: str = "default", n: int = 20) -> list[dict[str, Any]]:
    """Fetch a small public preview through the Hugging Face Dataset Viewer API."""
    url = f"{HF_DATASET_SERVER}/rows"
    params = {"dataset": dataset, "config": config, "split": split, "offset": 0, "length": n}
    response = requests.get(url, params=params, timeout=45)
    if response.status_code >= 400:
        raise RuntimeError(f"HF rows API failed for {dataset}: HTTP {response.status_code} {response.text[:300]}")
    data = response.json()
    rows = []
    for item in data.get("rows", []):
        row = item.get("row")
        if isinstance(row, dict):
            rows.append(row)
    return rows


def fetch_pmoa_rows(n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = [("default", "train_DSR1"), ("default", "train"), ("default", "case_study_100")]
    for config, split in configs:
        try:
            rows = hf_first_rows("snoroozi/pmoa-tts", split=split, config=config, n=max(n, 20))
        except Exception:
            continue
        if rows:
            break
    return rows[:n]


def fetch_pmc_patient_rows(n: int) -> list[dict[str, Any]]:
    candidates = [
        ("aisc-team-b1/PMC-Patients", "default", "train"),
        ("zhengyun21/PMC-Patients", "default", "train"),
    ]
    for dataset, config, split in candidates:
        try:
            rows = hf_first_rows(dataset, split=split, config=config, n=max(n, 20))
        except Exception:
            continue
        if rows:
            return rows[:n]
    return []


def fallback_pmoa_rows() -> list[dict[str, Any]]:
    """Small built-in open-case style seeds used only when public APIs are unreachable."""
    return [
        {
            "case_report_id": "fallback_nsclc_skull_met",
            "pmc_id": "fallback",
            "demographics": {"age": 43, "sex": "Male", "ethnicity": "Not Specified"},
            "diagnoses": ["Non-small-cell lung cancer", "Metastatic lesion in the parietal skull bone"],
            "death_info": {"observed_time": 30, "death_event_indicator": 1},
            "textual_timeseries": [
                {"time": -728, "event": "diagnosed with non-small-cell lung cancer with a left upper lobe mass"},
                {"time": -120, "event": "progressive headache and scalp swelling"},
                {"time": 0, "event": "CT showed destructive parietal skull bone lesion"},
                {"time": 7, "event": "biopsy confirmed metastatic lung adenocarcinoma"},
            ],
        },
        {
            "case_report_id": "fallback_pdp_headache",
            "pmc_id": "fallback",
            "demographics": {"age": 56, "sex": "Male", "ethnicity": "Not Specified"},
            "diagnoses": ["Postdural puncture headache", "Subdural hematoma"],
            "death_info": {"observed_time": 4320, "death_event_indicator": 0},
            "textual_timeseries": [
                {"time": 0, "event": "admitted with knee arthralgia and underwent lumbar puncture"},
                {"time": 48, "event": "developed postural headache after the procedure"},
                {"time": 144, "event": "brain CT showed subdural hematoma"},
                {"time": 4320, "event": "follow-up CT showed complete resolution of subdural hematoma"},
            ],
        },
        {
            "case_report_id": "fallback_reactive_lymphoid_hyperplasia",
            "pmc_id": "fallback",
            "demographics": {"age": 77, "sex": "Female", "ethnicity": "Not Specified"},
            "diagnoses": ["Intrahepatic reactive lymphoid hyperplasia", "Atrial fibrillation"],
            "death_info": {"observed_time": 365, "death_event_indicator": 0},
            "textual_timeseries": [
                {"time": -30, "event": "hospitalized for atrial fibrillation and planned ablation"},
                {"time": 0, "event": "contrast CT found a low-density liver lesion"},
                {"time": 14, "event": "MRI suggested hepatic malignancy mimic"},
                {"time": 21, "event": "pathology diagnosed intrahepatic reactive lymphoid hyperplasia"},
            ],
        },
        {
            "case_report_id": "fallback_aml_venetoclax_resistance",
            "pmc_id": "fallback",
            "demographics": {"age": 65, "sex": "Male", "ethnicity": "Not Specified"},
            "diagnoses": ["Acute myeloid leukemia", "Venetoclax plus azacitidine resistance"],
            "death_info": {"observed_time": 520, "death_event_indicator": 0},
            "textual_timeseries": [
                {"time": -420, "event": "diagnosed with acute myeloid leukemia after leukocytosis and anemia"},
                {"time": -300, "event": "treated with venetoclax plus azacitidine"},
                {"time": 0, "event": "bone marrow showed persistent myeloid blasts indicating resistance"},
                {"time": 28, "event": "histone deacetylase targeting therapy achieved hematologic improvement"},
            ],
        },
        {
            "case_report_id": "fallback_septic_shock_renal_failure",
            "pmc_id": "fallback",
            "demographics": {"age": 37, "sex": "Male", "ethnicity": "Not Specified"},
            "diagnoses": ["Septic shock", "Chronic renal insufficiency stage V", "Urinary tract infection"],
            "death_info": {"observed_time": 60, "death_event_indicator": 0},
            "textual_timeseries": [
                {"time": -365, "event": "history of cystectomy with ileal bladder replacement and neurogenic bladder"},
                {"time": 0, "event": "admitted with septic shock, urinary retention, pyuria, and unstable vital signs"},
                {"time": 1, "event": "treated with vasopressor and broad-spectrum antibiotics"},
                {"time": 14, "event": "renal function returned to chronic baseline after infection control"},
            ],
        },
    ]


def fallback_pmc_rows() -> list[dict[str, Any]]:
    return [
        {
            "patient_uid": "fallback-pmc-1",
            "title": "Case report of longitudinal diagnostic correction",
            "patient": "The patient had repeated visits where an initial benign hypothesis was later revised after imaging and pathology established the final diagnosis.",
        },
        {
            "patient_uid": "fallback-pmc-2",
            "title": "Case report with treatment contraindication",
            "patient": "The patient had a documented medication intolerance that made a previously planned therapy inappropriate during later follow-up.",
        },
    ]


def paired_source_rows(n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    try:
        pmoa = fetch_pmoa_rows(n)
    except Exception as exc:
        notes.append(f"PMOA-TTS fetch failed: {exc}")
        pmoa = []
    try:
        pmc = fetch_pmc_patient_rows(n)
    except Exception as exc:
        notes.append(f"PMC-Patients fetch failed: {exc}")
        pmc = []
    if not pmoa:
        pmoa = fallback_pmoa_rows()
        notes.append("Using built-in fallback PMOA-style seeds.")
    if not pmc:
        pmc = fallback_pmc_rows()
        notes.append("Using built-in fallback PMC-style seeds.")
    if len(pmoa) < n:
        pmoa = list(itertools.islice(itertools.cycle(pmoa), n))
    if len(pmc) < n:
        pmc = list(itertools.islice(itertools.cycle(pmc), n))
    random.Random(13).shuffle(pmoa)
    return pmoa[:n], pmc[:n], notes

