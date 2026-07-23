from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mem_ehr_agent.benchmark import (
    TimelineCrossEncoder,
    TimelineSemanticEncoder,
    build_locomo_memory_store,
    evidence_recall_at5,
    format_locomo_memory_line,
    retrieve_evidence_reranked_timeline_bundles,
)
from scripts.screen_table2_hierarchical_retrieval import load_development_samples


FROZEN_GRID = [
    {"parent_k": 16, "candidate_k": 16, "final_k": 5, "neighbor_radius": 0, "bundle_max_chars": 900, "redundancy_threshold": 0.90, "semantic_rrf_weight": 2.0},
    {"parent_k": 16, "candidate_k": 16, "final_k": 8, "neighbor_radius": 1, "bundle_max_chars": 900, "redundancy_threshold": 0.90, "semantic_rrf_weight": 2.0},
    {"parent_k": 24, "candidate_k": 24, "final_k": 5, "neighbor_radius": 0, "bundle_max_chars": 900, "redundancy_threshold": 0.85, "semantic_rrf_weight": 2.0},
    {"parent_k": 24, "candidate_k": 24, "final_k": 8, "neighbor_radius": 1, "bundle_max_chars": 900, "redundancy_threshold": 0.85, "semantic_rrf_weight": 2.0},
    {"parent_k": 24, "candidate_k": 24, "final_k": 8, "neighbor_radius": 1, "bundle_max_chars": 1400, "redundancy_threshold": 0.90, "semantic_rrf_weight": 2.0},
    {"parent_k": 32, "candidate_k": 32, "final_k": 5, "neighbor_radius": 0, "bundle_max_chars": 900, "redundancy_threshold": 0.85, "semantic_rrf_weight": 2.0},
    {"parent_k": 32, "candidate_k": 32, "final_k": 8, "neighbor_radius": 1, "bundle_max_chars": 900, "redundancy_threshold": 0.85, "semantic_rrf_weight": 2.0},
    {"parent_k": 32, "candidate_k": 32, "final_k": 8, "neighbor_radius": 1, "bundle_max_chars": 1400, "redundancy_threshold": 0.90, "semantic_rrf_weight": 2.0},
    {"parent_k": 32, "candidate_k": 32, "final_k": 10, "neighbor_radius": 1, "bundle_max_chars": 900, "redundancy_threshold": 0.95, "semantic_rrf_weight": 1.0},
    {"parent_k": 32, "candidate_k": 32, "final_k": 10, "neighbor_radius": 1, "bundle_max_chars": 1400, "redundancy_threshold": 0.95, "semantic_rrf_weight": 3.0},
]


def stable_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/home/syh/A-mem-aaai"))
    parser.add_argument(
        "--locomo-path",
        type=Path,
        default=Path("datasets/amem_original/locomo/locomo10.official.json"),
    )
    parser.add_argument(
        "--locomo-manifest",
        type=Path,
        default=Path("runs/table2_session_cards_20260719/dev_manifest.json"),
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("models/bge-small-en-v1.5"),
    )
    parser.add_argument(
        "--cross-encoder-model",
        type=Path,
        default=Path("models/bge-reranker-v2-m3"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/table2_evidence_rerank_20260723/offline_screen"),
    )
    parser.add_argument("--embedding-window-tokens", type=int, default=256)
    parser.add_argument("--card-max-chars", type=int, default=4000)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else args.project / path

    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    grid_manifest = {
        "protocol": "frozen Table 2 evidence_rerank retrieval-only development grid",
        "test_sets_used": False,
        "ranking": [
            "higher LoCoMo development Evidence R@5",
            "lower mean context characters",
            "smaller final evidence budget",
            "smaller candidate budget",
        ],
        "grid": FROZEN_GRID,
    }
    grid_path = output_root / "FROZEN_GRID.json"
    if grid_path.exists():
        existing = json.loads(grid_path.read_text(encoding="utf-8"))
        if existing != grid_manifest:
            raise RuntimeError("Frozen grid already exists with different content.")
    else:
        grid_path.write_text(
            json.dumps(grid_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    samples = load_development_samples(
        "locomo",
        resolve(args.locomo_path),
        resolve(args.locomo_manifest),
    )
    encoder = TimelineSemanticEncoder(str(resolve(args.embedding_model)))
    reranker = TimelineCrossEncoder(str(resolve(args.cross_encoder_model)))
    stores: dict[str, Any] = {}
    for sample in samples:
        conversation_id = str(sample.get("conversation_id") or sample["sample_id"])
        if conversation_id in stores:
            continue
        store_path = output_root / "stores" / f"{stable_name(conversation_id)}.memory.jsonl"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        stores[conversation_id] = build_locomo_memory_store(
            sample,
            store_path,
            card_granularity="session_chunk",
            card_max_chars=args.card_max_chars,
        )

    results: list[dict[str, Any]] = []
    for config_index, config in enumerate(FROZEN_GRID, start=1):
        recalls: list[float] = []
        context_chars: list[float] = []
        retrieved_counts: list[float] = []
        for sample in samples:
            conversation_id = str(sample.get("conversation_id") or sample["sample_id"])
            retrieved = retrieve_evidence_reranked_timeline_bundles(
                stores[conversation_id],
                sample,
                semantic_encoder=encoder,
                cross_encoder=reranker,
                embedding_window_tokens=args.embedding_window_tokens,
                parent_k=int(config["parent_k"]),
                candidate_k=int(config["candidate_k"]),
                final_k=int(config["final_k"]),
                neighbor_radius=int(config["neighbor_radius"]),
                bundle_max_chars=int(config["bundle_max_chars"]),
                redundancy_threshold=float(config["redundancy_threshold"]),
                semantic_rrf_weight=float(config["semantic_rrf_weight"]),
            )
            refs_at5 = sorted(
                {
                    str(ref)
                    for bundle in retrieved[:5]
                    for ref in bundle.get("evidence_refs", [])
                }
            )
            recall = evidence_recall_at5(
                sample,
                {"retrieved_evidence_refs_at5": refs_at5},
            )
            if recall is not None:
                recalls.append(float(recall))
            context_chars.append(
                float(
                    len(
                        "\n".join(
                            format_locomo_memory_line(bundle)
                            for bundle in retrieved
                        )
                    )
                )
            )
            retrieved_counts.append(float(len(retrieved)))
        results.append(
            {
                "config_index": config_index,
                **config,
                "sample_count": len(samples),
                "evidence_sample_count": len(recalls),
                "evidence_r5": mean(recalls),
                "mean_context_chars": mean(context_chars),
                "mean_retrieved_bundles": mean(retrieved_counts),
            }
        )

    ranked = sorted(
        results,
        key=lambda row: (
            -float(row["evidence_r5"] or 0.0),
            float(row["mean_context_chars"] or 0.0),
            int(row["final_k"]),
            int(row["candidate_k"]),
            int(row["config_index"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    families: set[tuple[int, int, int]] = set()
    for row in ranked:
        family = (
            int(row["final_k"]),
            int(row["neighbor_radius"]),
            int(row["bundle_max_chars"]),
        )
        if family in families:
            continue
        selected.append(row)
        families.add(family)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        selected.extend(row for row in ranked if row not in selected)
        selected = selected[:3]

    report = {
        **grid_manifest,
        "embedding_identity": encoder.identity,
        "cross_encoder_identity": reranker.identity,
        "development_count": len(samples),
        "selected": selected,
        "all_results": ranked,
    }
    (output_root / "offline_screen.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
