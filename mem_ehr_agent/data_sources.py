from __future__ import annotations

import itertools
import json
import os
import random
from pathlib import Path
from typing import Any

import requests


DEFAULT_HF_DATASET_SERVER = "https://datasets-server.huggingface.co"
PMOA_CACHE_NAMES = ["pmoa_tts.json", "pmoa-tts.json", "snoroozi_pmoa-tts.json"]
PMC_CACHE_NAMES = ["pmc_patients.json", "PMC-Patients.json", "aisc-team-b1_PMC-Patients.json"]


def load_cached_rows(cache_dir: str | Path | None, names: list[str], n: int) -> list[dict[str, Any]]:
    if not cache_dir:
        return []
    root = Path(cache_dir)
    for name in names:
        path = root / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            raw_rows = data["rows"]
            rows = [item.get("row") if isinstance(item, dict) and "row" in item else item for item in raw_rows]
        elif isinstance(data, list):
            rows = [item.get("row") if isinstance(item, dict) and "row" in item else item for item in data]
        else:
            rows = []
        out = [row for row in rows if isinstance(row, dict)]
        if out:
            return out[:n]
    return []


def hf_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    server = os.getenv("HF_DATASET_SERVER", DEFAULT_HF_DATASET_SERVER).rstrip("/")
    url = f"{server}/{endpoint.lstrip('/')}"
    response = requests.get(url, params=params, timeout=45)
    if response.status_code >= 400:
        raise RuntimeError(f"HF {endpoint} API failed: HTTP {response.status_code} {response.text[:300]}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"HF {endpoint} API returned non-object payload")
    return data


def hf_dataset_splits(dataset: str) -> list[tuple[str, str]]:
    data = hf_json("splits", {"dataset": dataset})
    splits = []
    for item in data.get("splits", []):
        if isinstance(item, dict) and item.get("config") and item.get("split"):
            splits.append((str(item["config"]), str(item["split"])))
    return splits


def hf_first_rows(
    dataset: str,
    *,
    split: str = "train",
    config: str = "default",
    n: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch a small public preview through the Hugging Face Dataset Viewer API."""
    params = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": n}
    data = hf_json("rows", params)
    rows = []
    for item in data.get("rows", []):
        row = item.get("row")
        if isinstance(row, dict):
            rows.append(row)
    return rows


def fetch_hf_rows(
    dataset: str,
    *,
    preferred_splits: list[str],
    n: int,
    notes: list[str],
    page_size: int = 100,
) -> list[dict[str, Any]]:
    discovered = hf_dataset_splits(dataset)
    split_order: list[tuple[str, str]] = []
    for preferred in preferred_splits:
        split_order.extend(item for item in discovered if item[1] == preferred and item not in split_order)
    split_order.extend(item for item in discovered if item not in split_order)
    if not split_order:
        raise RuntimeError(f"No Hugging Face Dataset Viewer splits discovered for {dataset}")
    errors = []
    for config, split in split_order:
        rows: list[dict[str, Any]] = []
        try:
            for offset in range(0, n, page_size):
                batch = hf_first_rows(
                    dataset,
                    split=split,
                    config=config,
                    n=min(page_size, n - offset),
                    offset=offset,
                )
                if not batch:
                    break
                rows.extend(batch)
                if len(rows) >= n:
                    break
        except Exception as exc:  # noqa: BLE001 - try the next public split
            errors.append(f"{dataset}/{config}/{split}: {exc}")
            continue
        if rows:
            notes.append(f"{dataset}: fetched {len(rows[:n])} rows from config={config} split={split}.")
            return rows[:n]
    raise RuntimeError("; ".join(errors) or f"No rows fetched for {dataset}")


def fetch_pmoa_rows(
    n: int,
    notes: list[str] | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    notes = notes if notes is not None else []
    cached = load_cached_rows(cache_dir, PMOA_CACHE_NAMES, n)
    if cached:
        notes.append(f"snoroozi/pmoa-tts: loaded {len(cached)} rows from cache {cache_dir}.")
        return cached
    return fetch_hf_rows(
        "snoroozi/pmoa-tts",
        preferred_splits=["train_DSR1", "train", "case_study_100", "case_study_25k_DSR1", "case_study_25k_L33"],
        n=n,
        notes=notes,
    )


def fetch_pmc_patient_rows(
    n: int,
    notes: list[str] | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    notes = notes if notes is not None else []
    cached = load_cached_rows(cache_dir, PMC_CACHE_NAMES, n)
    if cached:
        notes.append(f"PMC-Patients: loaded {len(cached)} rows from cache {cache_dir}.")
        return cached
    candidates = [
        ("aisc-team-b1/PMC-Patients", ["train"]),
        ("zhengyun21/PMC-Patients", ["train"]),
    ]
    errors = []
    for dataset, preferred_splits in candidates:
        try:
            return fetch_hf_rows(dataset, preferred_splits=preferred_splits, n=n, notes=notes)
        except Exception as exc:  # noqa: BLE001 - zhengyun21 is known to fail Dataset Viewer generation sometimes
            errors.append(f"{dataset}: {exc}")
            notes.append(f"{dataset}: fetch failed: {exc}")
    raise RuntimeError("; ".join(errors))


def legacy_fetch_pmoa_rows(n: int) -> list[dict[str, Any]]:
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


def legacy_fetch_pmc_patient_rows(n: int) -> list[dict[str, Any]]:
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


def paired_source_rows(
    n: int,
    *,
    require_real_data: bool = False,
    cache_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    cache_dir = cache_dir or os.getenv("MEM_EHR_DATA_CACHE_DIR")
    try:
        pmoa = fetch_pmoa_rows(n, notes, cache_dir=cache_dir)
    except Exception as exc:
        notes.append(f"PMOA-TTS fetch failed: {exc}")
        pmoa = []
    try:
        pmc = fetch_pmc_patient_rows(n, notes, cache_dir=cache_dir)
    except Exception as exc:
        notes.append(f"PMC-Patients fetch failed: {exc}")
        pmc = []
    if require_real_data and (not pmoa or not pmc):
        raise RuntimeError("Real-data build required but one or more public datasets could not be fetched:\n" + "\n".join(notes))
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
