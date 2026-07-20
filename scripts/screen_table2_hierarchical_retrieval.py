from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from mem_ehr_agent.benchmark import (
    TEMPORAL_TERMS,
    TimelineSemanticEncoder,
    _hierarchical_turn_bundle,
    build_locomo_memory_store,
    evidence_recall_at5,
    extract_locomo_terms,
    format_locomo_memory_line,
    load_dialsim_samples,
    load_frozen_sample_ids,
    load_locomo_samples,
    locomo_expanded_query,
    locomo_memory_retrieval_text,
    locomo_retrieval_score,
    normalized_vector_cosine,
    retrieve_locomo_memories,
    select_frozen_samples,
    timeline_turn_refs,
    weighted_rrf_scores,
)


def stable_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_development_samples(
    dataset: str,
    dataset_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    ids = load_frozen_sample_ids(manifest_path)
    if dataset == "dialsim":
        return load_dialsim_samples(dataset_path, required_sample_ids=ids)
    return select_frozen_samples(load_locomo_samples(dataset_path), ids)


def retrieve_hierarchical_grid(
    store: Any,
    sample: dict[str, Any],
    *,
    semantic_encoder: Any,
    semantic_rrf_weight: float,
    embedding_window_tokens: int,
    parent_ks: list[int],
    neighbor_radii: list[int],
    bundle_max_chars_values: list[int],
    bundle_k: int,
) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    """Evaluate a parameter grid while encoding each bundle shape only once."""
    query = locomo_expanded_query(sample)
    maximum_parent_k = max(parent_ks)
    parents = retrieve_locomo_memories(
        store,
        sample,
        top_k=maximum_parent_k,
        coarse_k=max(maximum_parent_k, 32),
        retriever="hybrid_bge",
        semantic_encoder=semantic_encoder,
        semantic_rrf_weight=semantic_rrf_weight,
        embedding_window_tokens=embedding_window_tokens,
    )
    parent_ref_ranks: dict[str, int] = {}
    parent_session_ranks: dict[str, int] = {}
    for rank, parent in enumerate(parents, start=1):
        for ref in parent.get("evidence_refs", []):
            parent_ref_ranks.setdefault(str(ref), rank)
        session = str((parent.get("time_scope") or {}).get("session") or "")
        if session:
            parent_session_ranks.setdefault(session, rank)
    eligible_turns: list[tuple[int, int]] = []
    for turn_index, turn in enumerate(sample.get("turns") or []):
        refs = timeline_turn_refs(turn)
        session = str(turn.get("session") or turn.get("session_date") or "unknown")
        matching_ranks = [
            parent_ref_ranks[ref] for ref in refs if ref in parent_ref_ranks
        ]
        parent_rank = (
            min(matching_ranks)
            if matching_ranks
            else parent_session_ranks.get(session)
        )
        if parent_rank is not None:
            eligible_turns.append((turn_index, parent_rank))
    if not eligible_turns:
        return {
            (parent_k, neighbor_radius, bundle_max_chars): []
            for parent_k in parent_ks
            for neighbor_radius in neighbor_radii
            for bundle_max_chars in bundle_max_chars_values
        }

    query_vector = semantic_encoder.encode([query], is_query=True)[0]
    category = str(sample.get("category_name") or "").lower().replace("_", "-")
    is_multi_hop = "multi-hop" in category or "multihop" in category
    query_terms = set(extract_locomo_terms(query))
    is_temporal = "temporal" in category or bool(query_terms & TEMPORAL_TERMS)
    output: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for neighbor_radius in neighbor_radii:
        for bundle_max_chars in bundle_max_chars_values:
            candidates: list[dict[str, Any]] = []
            for turn_index, parent_rank in eligible_turns:
                bundle = _hierarchical_turn_bundle(
                    sample,
                    anchor_index=turn_index,
                    query=query,
                    neighbor_radius=neighbor_radius,
                    bundle_max_chars=bundle_max_chars,
                    parent_rank=parent_rank,
                )
                bundle["lexical_retrieval_score"] = locomo_retrieval_score(
                    query, sample, bundle
                )
                candidates.append(bundle)
            candidate_vectors = semantic_encoder.encode(
                [locomo_memory_retrieval_text(candidate) for candidate in candidates],
                is_query=False,
            )
            semantic_scores = {
                str(candidate["memory_id"]): normalized_vector_cosine(
                    query_vector, vector
                )
                for candidate, vector in zip(candidates, candidate_vectors)
            }
            for parent_k in parent_ks:
                visible = [
                    candidate
                    for candidate in candidates
                    if int(candidate["parent_rank"]) <= parent_k
                ]
                lexical_order = [
                    str(candidate["memory_id"])
                    for candidate in sorted(
                        visible,
                        key=lambda item: (
                            float(item["lexical_retrieval_score"]),
                            -int(item["parent_rank"]),
                            str(item["memory_id"]),
                        ),
                        reverse=True,
                    )
                ]
                semantic_order = sorted(
                    (str(candidate["memory_id"]) for candidate in visible),
                    key=lambda memory_id: (
                        semantic_scores[memory_id],
                        memory_id,
                    ),
                    reverse=True,
                )
                fused_scores = weighted_rrf_scores(
                    lexical_order,
                    semantic_order,
                    semantic_weight=semantic_rrf_weight,
                )
                ranked = sorted(
                    visible,
                    key=lambda item: (
                        fused_scores[str(item["memory_id"])]
                        + 0.01 / int(item["parent_rank"]),
                        float(item["lexical_retrieval_score"]),
                        semantic_scores[str(item["memory_id"])],
                        str(item["memory_id"]),
                    ),
                    reverse=True,
                )
                selected: list[dict[str, Any]] = []
                seen_sessions: set[str] = set()
                if is_multi_hop:
                    for candidate in ranked:
                        session = str(
                            (candidate.get("time_scope") or {}).get("session")
                            or ""
                        )
                        if session in seen_sessions:
                            continue
                        selected.append(candidate)
                        seen_sessions.add(session)
                        if len(selected) >= bundle_k:
                            break
                if len(selected) < bundle_k:
                    selected_ids = {
                        str(candidate["memory_id"]) for candidate in selected
                    }
                    selected.extend(
                        candidate
                        for candidate in ranked
                        if str(candidate["memory_id"]) not in selected_ids
                    )
                selected = [dict(candidate) for candidate in selected[:bundle_k]]
                if is_temporal:
                    selected.sort(
                        key=lambda item: (
                            str(
                                (item.get("time_scope") or {}).get("date") or ""
                            ),
                            str(
                                (item.get("time_scope") or {}).get("time") or ""
                            ),
                            int(item.get("anchor_turn_index") or 0),
                        )
                    )
                for candidate in selected:
                    memory_id = str(candidate["memory_id"])
                    candidate["retrieval_score"] = fused_scores[memory_id]
                    candidate["semantic_retrieval_score"] = semantic_scores[
                        memory_id
                    ]
                output[(parent_k, neighbor_radius, bundle_max_chars)] = selected
    return output


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
        "--dialsim-path",
        type=Path,
        default=Path("datasets/amem_original/dialsim"),
    )
    parser.add_argument(
        "--dialsim-manifest",
        type=Path,
        default=Path(
            "data/processed/table2_hierarchical_dev_20260720/"
            "dialsim_dev_manifest.json"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("models/bge-small-en-v1.5"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/table2_hierarchical_bge_20260720/offline_screen"),
    )
    parser.add_argument("--parent-k", default="5,8,12")
    parser.add_argument("--bundle-k", default="5,8")
    parser.add_argument("--neighbor-radius", default="0,1")
    parser.add_argument("--bundle-max-chars", default="600,900")
    parser.add_argument("--semantic-rrf-weight", type=float, default=2.0)
    parser.add_argument("--embedding-window-tokens", type=int, default=256)
    parser.add_argument("--card-max-chars", type=int, default=4000)
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else args.project / path

    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    samples_by_dataset = {
        "locomo": load_development_samples(
            "locomo", resolve(args.locomo_path), resolve(args.locomo_manifest)
        ),
        "dialsim": load_development_samples(
            "dialsim", resolve(args.dialsim_path), resolve(args.dialsim_manifest)
        ),
    }
    encoder = TimelineSemanticEncoder(str(resolve(args.embedding_model)))
    stores: dict[tuple[str, str], Any] = {}
    for dataset, samples in samples_by_dataset.items():
        for sample in samples:
            conversation_id = str(
                sample.get("conversation_id") or sample["sample_id"]
            )
            key = (dataset, conversation_id)
            if key in stores:
                continue
            path = (
                output_root
                / "stores"
                / dataset
                / f"{stable_name(conversation_id)}.memory.jsonl"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            stores[key] = build_locomo_memory_store(
                sample,
                path,
                card_granularity="session_chunk",
                card_max_chars=args.card_max_chars,
            )

    configs = [
        {
            "parent_k": parent_k,
            "bundle_k": bundle_k,
            "neighbor_radius": neighbor_radius,
            "bundle_max_chars": bundle_max_chars,
        }
        for parent_k, bundle_k, neighbor_radius, bundle_max_chars in itertools.product(
            parse_ints(args.parent_k),
            parse_ints(args.bundle_k),
            parse_ints(args.neighbor_radius),
            parse_ints(args.bundle_max_chars),
        )
    ]
    maximum_bundle_k = max(parse_ints(args.bundle_k))
    retrieval_cache: dict[tuple[str, str, int, int, int], list[dict[str, Any]]] = {}
    parent_ks = sorted({int(config["parent_k"]) for config in configs})
    neighbor_radii = sorted(
        {int(config["neighbor_radius"]) for config in configs}
    )
    bundle_max_chars_values = sorted(
        {int(config["bundle_max_chars"]) for config in configs}
    )
    for dataset, samples in samples_by_dataset.items():
        for sample in samples:
            conversation_id = str(
                sample.get("conversation_id") or sample["sample_id"]
            )
            sample_grid = retrieve_hierarchical_grid(
                stores[(dataset, conversation_id)],
                sample,
                semantic_encoder=encoder,
                semantic_rrf_weight=args.semantic_rrf_weight,
                embedding_window_tokens=args.embedding_window_tokens,
                parent_ks=parent_ks,
                neighbor_radii=neighbor_radii,
                bundle_max_chars_values=bundle_max_chars_values,
                bundle_k=maximum_bundle_k,
            )
            for (
                parent_k,
                neighbor_radius,
                bundle_max_chars,
            ), bundles in sample_grid.items():
                cache_key = (
                    dataset,
                    str(sample["sample_id"]),
                    parent_k,
                    neighbor_radius,
                    bundle_max_chars,
                )
                retrieval_cache[cache_key] = bundles

    results: list[dict[str, Any]] = []
    for config in configs:
        dataset_summaries: dict[str, dict[str, Any]] = {}
        all_context_chars: list[float] = []
        for dataset, samples in samples_by_dataset.items():
            recalls: list[float] = []
            context_chars: list[float] = []
            retrieved_counts: list[float] = []
            for sample in samples:
                bundles = retrieval_cache[
                    (
                        dataset,
                        str(sample["sample_id"]),
                        int(config["parent_k"]),
                        int(config["neighbor_radius"]),
                        int(config["bundle_max_chars"]),
                    )
                ][: int(config["bundle_k"])]
                refs_at5 = sorted(
                    {
                        str(ref)
                        for bundle in bundles[:5]
                        for ref in bundle.get("evidence_refs", [])
                    }
                )
                recall = evidence_recall_at5(
                    sample, {"retrieved_evidence_refs_at5": refs_at5}
                )
                if recall is not None:
                    recalls.append(float(recall))
                chars = len(
                    "\n".join(format_locomo_memory_line(bundle) for bundle in bundles)
                )
                context_chars.append(float(chars))
                all_context_chars.append(float(chars))
                retrieved_counts.append(float(len(bundles)))
            dataset_summaries[dataset] = {
                "sample_count": len(samples),
                "evidence_r5": mean(recalls),
                "evidence_sample_count": len(recalls),
                "mean_context_chars": mean(context_chars),
                "mean_retrieved_bundles": mean(retrieved_counts),
            }
        results.append(
            {
                **config,
                "semantic_rrf_weight": args.semantic_rrf_weight,
                "embedding_window_tokens": args.embedding_window_tokens,
                "card_max_chars": args.card_max_chars,
                "evidence_r5": dataset_summaries["locomo"]["evidence_r5"],
                "mean_context_chars": mean(all_context_chars),
                "datasets": dataset_summaries,
            }
        )

    ranked = sorted(
        results,
        key=lambda row: (
            -float(row["evidence_r5"] or 0.0),
            float(row["mean_context_chars"] or 0.0),
            int(row["parent_k"]),
            int(row["bundle_k"]),
            int(row["neighbor_radius"]),
            int(row["bundle_max_chars"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_families: set[tuple[int, int]] = set()
    for row in ranked:
        family = (int(row["bundle_k"]), int(row["neighbor_radius"]))
        if family in selected_families:
            continue
        selected.append(row)
        selected_families.add(family)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        selected.extend(row for row in ranked if row not in selected)
        selected = selected[:3]

    report = {
        "protocol": "hierarchical_bge retrieval-only development screen",
        "test_sets_used": False,
        "ranking": [
            "higher LoCoMo development Evidence R@5",
            "lower joint LoCoMo/DialSim context characters",
            "smaller parent and bundle budgets",
        ],
        "diversity_rule": (
            "Retain the best configuration from distinct "
            "(bundle_k, neighbor_radius) families before development QA."
        ),
        "encoder_identity": encoder.identity,
        "grid_size": len(results),
        "development_counts": {
            dataset: len(samples) for dataset, samples in samples_by_dataset.items()
        },
        "selected": selected,
        "all_results": ranked,
    }
    (output_root / "offline_screen.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "offline_screen.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "parent_k",
            "bundle_k",
            "neighbor_radius",
            "bundle_max_chars",
            "evidence_r5",
            "mean_context_chars",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow({key: row[key] for key in fieldnames})
    print(json.dumps({"selected": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
