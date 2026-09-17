from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medimem.benchmark import load_dialsim_samples, load_frozen_sample_ids


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_select(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in candidates:
        strata[
            (
                str(sample.get("split") or ""),
                str(sample.get("category_name") or ""),
            )
        ].append(sample)
    rng = random.Random(seed)
    for values in strata.values():
        rng.shuffle(values)
    ordered_strata = sorted(strata)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < count and ordered_strata:
        key = ordered_strata[cursor % len(ordered_strata)]
        values = strata[key]
        if values:
            selected.append(values.pop())
            cursor += 1
            continue
        ordered_strata.remove(key)
        if ordered_strata:
            cursor %= len(ordered_strata)
    if len(selected) != count:
        raise RuntimeError(
            f"Only {len(selected)} balanced candidates were available; expected {count}."
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/syh/A-mem-aaai"))
    parser.add_argument(
        "--dialsim-path",
        type=Path,
        default=Path("datasets/amem_original/dialsim"),
    )
    parser.add_argument(
        "--dialsim-test-manifest",
        type=Path,
        default=Path(
            "runs/nonmedical_timeline_formal_20260716_recovery_v4/core/"
            "dialsim_memory_random1000_20260716_210900_343853_pid499882/"
            "sample_manifest.json"
        ),
    )
    parser.add_argument(
        "--locomo-dev-manifest",
        type=Path,
        default=Path("runs/table2_session_cards_20260719/dev_manifest.json"),
    )
    parser.add_argument(
        "--locomo-test-manifest",
        type=Path,
        default=Path(
            "runs/nonmedical_timeline_formal_20260716_recovery_v2/core/"
            "locomo_benchmark_20260716_183348_550291_pid453346/"
            "sample_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/table2_hierarchical_dev_20260720"),
    )
    parser.add_argument("--candidate-n", type=int, default=5000)
    parser.add_argument("--sample-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else args.project / path

    dialsim_path = resolve(args.dialsim_path)
    test_manifest_path = resolve(args.dialsim_test_manifest)
    locomo_dev_path = resolve(args.locomo_dev_manifest)
    locomo_test_path = resolve(args.locomo_test_manifest)
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dialsim_test_ids = set(load_frozen_sample_ids(test_manifest_path))
    candidate_seed = args.seed + 1
    candidates = load_dialsim_samples(
        dialsim_path,
        sample_n=args.candidate_n,
        random_seed=candidate_seed,
    )
    eligible = [
        sample
        for sample in candidates
        if str(sample["sample_id"]) not in dialsim_test_ids
    ]
    selected = balanced_select(eligible, count=args.sample_n, seed=args.seed)
    selected_ids = [str(sample["sample_id"]) for sample in selected]
    overlap = sorted(set(selected_ids) & dialsim_test_ids)
    if overlap:
        raise RuntimeError(f"DialSim development/test overlap: {overlap[:10]}")

    distribution = Counter(
        (
            str(sample.get("split") or ""),
            str(sample.get("category_name") or ""),
        )
        for sample in selected
    )
    dialsim_manifest = {
        "dataset": "dialsim",
        "role": "table2_development_only",
        "selection": "balanced_round_robin_by_subset_and_question_family",
        "random_seed": args.seed,
        "candidate_seed": candidate_seed,
        "candidate_n": args.candidate_n,
        "eligible_after_frozen_test_exclusion": len(eligible),
        "sample_count": len(selected_ids),
        "sample_ids": selected_ids,
        "sample_ids_sha256": canonical_hash(selected_ids),
        "runtime_input_hash_sha256": canonical_hash(
            [
                {
                    "sample_id": sample["sample_id"],
                    "conversation_id": sample.get("conversation_id"),
                    "context": sample.get("context"),
                    "turns": sample.get("turns"),
                    "question": sample.get("question"),
                }
                for sample in selected
            ]
        ),
        "excluded_test_manifest": str(test_manifest_path),
        "excluded_test_manifest_sha256": file_sha256(test_manifest_path),
        "test_overlap_count": 0,
        "stratum_counts": {
            f"{subset}::{family}": value
            for (subset, family), value in sorted(distribution.items())
        },
    }
    dialsim_output = output_root / "dialsim_dev_manifest.json"
    dialsim_output.write_text(
        json.dumps(dialsim_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    locomo_dev_ids = load_frozen_sample_ids(locomo_dev_path)
    locomo_test_ids = set(load_frozen_sample_ids(locomo_test_path))
    locomo_overlap = sorted(set(locomo_dev_ids) & locomo_test_ids)
    if locomo_overlap:
        raise RuntimeError(f"LoCoMo development/test overlap: {locomo_overlap[:10]}")

    combined = {
        "protocol": "Table 2 hierarchical-BGE frozen development sets",
        "selection_seed": args.seed,
        "development_only": True,
        "formal_test_tuning_prohibited": True,
        "locomo": {
            "manifest": str(locomo_dev_path),
            "manifest_sha256": file_sha256(locomo_dev_path),
            "sample_count": len(locomo_dev_ids),
            "sample_ids_sha256": canonical_hash(locomo_dev_ids),
            "test_manifest": str(locomo_test_path),
            "test_overlap_count": 0,
        },
        "dialsim": {
            "manifest": str(dialsim_output),
            "manifest_sha256": file_sha256(dialsim_output),
            "sample_count": len(selected_ids),
            "sample_ids_sha256": canonical_hash(selected_ids),
            "test_manifest": str(test_manifest_path),
            "test_overlap_count": 0,
        },
    }
    combined_output = output_root / "development_protocol_manifest.json"
    combined_output.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(combined, ensure_ascii=False))


if __name__ == "__main__":
    main()

