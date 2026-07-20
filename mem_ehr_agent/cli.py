from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import benchmark_status, parse_int_list, parse_method_queue, parse_methods, run_locomo_parallel_benchmark, run_native_benchmark
from .data_builder import build_dataset, write_prefix_slices
from .expanded_data import build_medical_ehr_pool, parse_source_names
from .io_utils import ensure_dir, read_jsonl, write_jsonl
from .metrics import build_metric_gate, read_metrics_csv, render_metric_gate
from .optimizer import optimize, optimize_suite
from .schemas import validate_case
from .statistical_analysis import analyze_run, source_name


def cmd_data_build(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.output).parent)
    notes_path = str(Path(args.output).with_suffix(".notes.txt"))
    cases = build_dataset(
        args.n,
        args.output,
        notes_path=notes_path,
        require_real_data=args.require_real_data,
        cache_dir=args.cache_dir,
    )
    print(f"wrote {len(cases)} cases to {args.output}")
    print(f"wrote data-source notes to {notes_path}")


def cmd_data_validate(args: argparse.Namespace) -> None:
    cases = read_jsonl(args.dataset)
    failures = {}
    for case in cases:
        errors = validate_case(case)
        if errors:
            failures[case.get("case_id", "unknown")] = errors
    if failures:
        raise SystemExit(f"validation failed: {failures}")
    print(f"validated {len(cases)} cases")


def cmd_optimize(args: argparse.Namespace) -> None:
    run_dir = optimize(
        dataset_path=args.dataset,
        max_rounds=args.max_rounds,
        continuous=args.continuous,
        require_api=args.require_api,
        sleep_s=args.sleep_s,
        max_workers=args.max_workers,
        disable_dynamic_top_k=args.disable_dynamic_top_k,
        disable_normalization=args.disable_normalization,
        disable_memory_cleaning=args.disable_memory_cleaning,
        disable_critic_op_guard=args.disable_critic_op_guard,
        disable_evidence_note_injection=args.disable_evidence_note_injection,
        disable_counterfactual_verification=args.disable_counterfactual_verification,
    )
    print(f"run_dir={run_dir}")


def cmd_experiment_suite(args: argparse.Namespace) -> None:
    run_dir = optimize_suite(
        dataset_path=args.dataset,
        require_api=args.require_api,
        max_workers=args.max_workers,
        focused=args.focused,
        suite_profile=args.suite_profile,
        baseline_set=args.baseline_set,
        ablation_groups=args.ablation_groups,
        counterfactual_policy=args.counterfactual_policy,
        counterfactual_sample_rate=args.counterfactual_sample_rate,
        counterfactual_risk_threshold=args.counterfactual_risk_threshold,
        defer_reports=args.defer_reports,
        completion_token_budget=args.completion_token_budget,
        run_seed=args.run_seed,
        output_root=args.output_root,
    )
    print(f"run_dir={run_dir}")


def cmd_analyze_run(args: argparse.Namespace) -> None:
    output = analyze_run(
        dataset_path=args.dataset,
        prediction_paths=args.predictions,
        output_dir=args.output_dir,
        target_method=args.target_method,
        compare_methods=[item.strip() for item in args.compare_methods.split(",") if item.strip()]
        if args.compare_methods
        else None,
        resamples=args.resamples,
        seed=args.random_seed,
    )
    print(f"analysis_dir={output}")


def cmd_filter_sources(args: argparse.Namespace) -> None:
    selected = {item.strip().lower() for item in args.sources.split(",") if item.strip()}
    cases = [case for case in read_jsonl(args.dataset) if source_name(case) in selected]
    if not cases:
        raise SystemExit(f"No cases matched sources: {sorted(selected)}")
    write_jsonl(args.output, cases)
    print(f"wrote {len(cases)} rows to {args.output}")


def cmd_sample(args: argparse.Namespace) -> None:
    cases = read_jsonl(args.dataset)
    write_jsonl(args.output, cases[: args.n])
    print(f"wrote {min(args.n, len(cases))} rows to {args.output}")


def cmd_prefix_slices(args: argparse.Namespace) -> None:
    cases = read_jsonl(args.dataset)
    sizes = [int(item.strip()) for item in str(args.sizes).split(",") if item.strip()]
    paths = write_prefix_slices(
        cases,
        output_dir=args.output_dir,
        stem=args.stem,
        sizes=sizes,
        notes=f"source_dataset={args.dataset}",
    )
    for path in paths:
        print(f"wrote {path}")


def cmd_data_build_medical_pool(args: argparse.Namespace) -> None:
    manifest = build_medical_ehr_pool(
        per_source_n=args.per_source_n,
        output_dir=args.output_dir,
        sources=parse_source_names(args.sources),
        require_real_data=args.require_real_data,
        cache_dir=args.cache_dir,
        random_seed=args.random_seed,
    )
    print(f"wrote medical_ehr_pool manifest to {Path(args.output_dir) / 'manifest.json'}")
    print(f"pooled_path={manifest['pooled_path']}")


def cmd_benchmark_run(args: argparse.Namespace) -> None:
    run_dir = run_native_benchmark(
        dataset=args.dataset,
        methods=parse_methods(args.methods),
        dataset_path=args.dataset_path,
        limit=args.limit,
        sample_n=args.sample_n,
        random_seed=args.random_seed,
        max_workers=args.max_workers,
        output_root=args.output_root,
        require_api=args.require_api,
        locomo_top_k=args.locomo_top_k,
        locomo_coarse_k=args.locomo_coarse_k,
        top_k_sweep=parse_int_list(args.top_k_sweep),
        judge_answers=args.judge_answers,
        sample_manifest=args.sample_manifest,
        timeline_card_granularity=args.timeline_card_granularity,
        timeline_card_max_chars=args.timeline_card_max_chars,
        timeline_retriever=args.timeline_retriever,
        timeline_embedding_model=args.timeline_embedding_model,
        timeline_semantic_rrf_weight=args.timeline_semantic_rrf_weight,
        timeline_embedding_window_tokens=args.timeline_embedding_window_tokens,
        timeline_hierarchical_parent_k=args.timeline_hierarchical_parent_k,
        timeline_hierarchical_bundle_k=args.timeline_hierarchical_bundle_k,
        timeline_hierarchical_neighbor_radius=args.timeline_hierarchical_neighbor_radius,
        timeline_hierarchical_bundle_max_chars=args.timeline_hierarchical_bundle_max_chars,
    )
    print(f"run_dir={run_dir}")


def cmd_benchmark_run_locomo_parallel(args: argparse.Namespace) -> None:
    run_dir = run_locomo_parallel_benchmark(
        methods_a=parse_method_queue(args.methods_a),
        methods_b=parse_method_queue(args.methods_b),
        base_url_a=args.base_url_a,
        base_url_b=args.base_url_b,
        dataset_path=args.dataset_path,
        sample_n=args.sample_n,
        random_seed=args.random_seed,
        max_workers_per_queue=args.max_workers_per_queue,
        output_root=args.output_root,
        require_api=args.require_api,
        locomo_top_k=args.locomo_top_k,
        locomo_coarse_k=args.locomo_coarse_k,
    )
    print(f"run_dir={run_dir}")


def cmd_benchmark_status(args: argparse.Namespace) -> None:
    for row in benchmark_status():
        print(
            "{dataset}\t{method}\t{status}\t{env}\t{official_repo}".format(
                dataset=row.get("dataset"),
                method=row.get("method"),
                status=row.get("status"),
                env=row.get("env", ""),
                official_repo=row.get("official_repo", ""),
            )
        )


def cmd_metric_gate(args: argparse.Namespace) -> None:
    rows = read_metrics_csv(args.metrics)
    gate = build_metric_gate(rows, method=args.method)
    print(render_metric_gate(gate))
    if not gate["passed"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mem-ehr")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build")
    build.add_argument("--n", type=int, default=10)
    build.add_argument("--output", default="data/processed/samples.jsonl")
    build.add_argument(
        "--require-real-data",
        "--no-fallback",
        action="store_true",
        help="Fail the build if PMOA-TTS or PMC-Patients cannot be fetched from public sources.",
    )
    build.add_argument(
        "--cache-dir",
        default=None,
        help="Optional directory with pmoa_tts.json and pmc_patients.json rows, for offline real-data builds.",
    )
    build.set_defaults(func=cmd_data_build)
    pool = data_sub.add_parser("build-medical-pool")
    pool.add_argument("--per-source-n", type=int, default=1000)
    pool.add_argument("--output-dir", default="data/processed/medical_ehr_pool")
    pool.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source keys. Defaults to all configured medical pool sources.",
    )
    pool.add_argument(
        "--require-real-data",
        "--no-fallback",
        action="store_true",
        help="Fail if any selected source cannot provide the requested number of real rows.",
    )
    pool.add_argument("--cache-dir", default=None)
    pool.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Stable per-source sampling seed; samples from a larger fetched/cache pool when set.",
    )
    pool.set_defaults(func=cmd_data_build_medical_pool)
    validate = data_sub.add_parser("validate")
    validate.add_argument("--dataset", default="data/processed/samples.jsonl")
    validate.set_defaults(func=cmd_data_validate)
    sample = data_sub.add_parser("sample")
    sample.add_argument("--dataset", default="data/processed/samples.jsonl")
    sample.add_argument("--output", default="data/processed/sample_small.jsonl")
    sample.add_argument("--n", type=int, default=3)
    sample.set_defaults(func=cmd_sample)
    prefix = data_sub.add_parser("prefix-slices")
    prefix.add_argument("--dataset", required=True)
    prefix.add_argument("--output-dir", default="data/processed")
    prefix.add_argument("--stem", default="fixed_pool_prefix")
    prefix.add_argument("--sizes", default="50,100,200")
    prefix.set_defaults(func=cmd_prefix_slices)

    opt = sub.add_parser("optimize")
    opt.add_argument("--dataset", default="data/processed/samples.jsonl")
    opt.add_argument("--max-rounds", type=int, default=10, help="0 means no explicit round cap.")
    opt.add_argument("--continuous", action="store_true", help="Keep iterating after checkpoints until a win or blocker.")
    opt.add_argument("--require-api", action="store_true", help="Stop if DeepSeek API is unavailable instead of using fallback.")
    opt.add_argument("--sleep-s", type=int, default=15)
    opt.add_argument("--max-workers", type=int, default=16)
    opt.add_argument("--disable-dynamic-top-k", action="store_true", help="Ablation: use fixed fallback top_k for ours.")
    opt.add_argument("--disable-normalization", action="store_true", help="Ablation: disable diagnosis alias normalization for ours outputs.")
    opt.add_argument("--disable-memory-cleaning", action="store_true", help="Ablation: skip memory critique/discard/invalidate ops for ours.")
    opt.add_argument("--disable-critic-op-guard", action="store_true", help="Ablation: allow the LLM critic to choose memory ops without deterministic pollution-type guardrails.")
    opt.add_argument("--disable-evidence-note-injection", action="store_true", help="Ablation: omit source-aligned evidence notes from the final ours diagnosis prompt.")
    opt.add_argument("--disable-counterfactual-verification", action="store_true", help="Ablation: skip MediMem counterfactual verification and one-pass revision.")
    opt.set_defaults(func=cmd_optimize)

    suite = sub.add_parser("experiment-suite")
    suite.add_argument("--dataset", default="data/processed/samples.jsonl")
    suite.add_argument("--require-api", action="store_true", help="Stop if DeepSeek API is unavailable instead of using fallback.")
    suite.add_argument("--max-workers", type=int, default=16)
    suite.add_argument(
        "--suite-profile",
        default="standard",
        choices=["standard", "fast-formal", "fast_formal", "fast_formal_with_required_pipelines"],
        help="Use fast-formal for required pipelines plus full/no-memory-cleaning/no-evidence ablations.",
    )
    suite.add_argument(
        "--baseline-set",
        default=None,
        choices=["all", "focused", "required", "none"],
        help="all includes polluted variants; required runs direct/cot/amem/ddo/colacare only.",
    )
    suite.add_argument(
        "--ablation-groups",
        default=None,
        help="Comma-separated groups, e.g. full,no_memory_cleaning,no_evidence_note_injection,no_temporal_signal.",
    )
    suite.add_argument(
        "--counterfactual-policy",
        default="always",
        choices=["always", "risk_sample", "never"],
        help="risk_sample runs counterfactual verification only for high-risk plus sampled low-risk cases.",
    )
    suite.add_argument("--counterfactual-sample-rate", type=float, default=0.20)
    suite.add_argument("--counterfactual-risk-threshold", type=float, default=0.55)
    suite.add_argument(
        "--defer-reports",
        action="store_true",
        help="Write metrics/predictions/leakage during the run, then render markdown/error analysis once at the end.",
    )
    suite.add_argument(
        "--completion-token-budget",
        type=int,
        default=None,
        help="Optional cumulative generated-token budget per method/case (e.g. 750, 1500, 3000).",
    )
    suite.add_argument("--run-seed", type=int, default=20260706)
    suite.add_argument("--output-root", default="runs")
    suite.add_argument(
        "--focused",
        action="store_true",
        help="Run only direct/AMEM/polluted AMEM plus full ours and memory/evidence ablations.",
    )
    suite.set_defaults(func=cmd_experiment_suite)

    analysis = sub.add_parser("analyze-run")
    analysis.add_argument("--dataset", required=True)
    analysis.add_argument("--predictions", nargs="+", required=True)
    analysis.add_argument("--output-dir", required=True)
    analysis.add_argument("--target-method", default="full_medimem")
    analysis.add_argument("--compare-methods", default=None)
    analysis.add_argument("--resamples", type=int, default=10_000)
    analysis.add_argument("--random-seed", type=int, default=20260706)
    analysis.set_defaults(func=cmd_analyze_run)

    filter_sources = sub.add_parser("filter-sources")
    filter_sources.add_argument("--dataset", required=True)
    filter_sources.add_argument("--sources", required=True)
    filter_sources.add_argument("--output", required=True)
    filter_sources.set_defaults(func=cmd_filter_sources)

    gate = sub.add_parser("metric-gate")
    gate.add_argument("--metrics", required=True, help="Path to metrics.csv from an experiment run.")
    gate.add_argument("--method", default=None, help="Optional exact ours method row to validate.")
    gate.set_defaults(func=cmd_metric_gate)

    benchmark = sub.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_run = benchmark_sub.add_parser("run")
    benchmark_run.add_argument(
        "--dataset",
        required=True,
        choices=[
            "locomo",
            "dialsim",
            "longmemeval",
            "rhelm",
            "memoryos_native",
            "meminsight_native",
            "gmemory_native",
            "ddo_native",
        ],
    )
    benchmark_run.add_argument("--methods", required=True, help="Comma-separated methods, e.g. ours,amem")
    benchmark_run.add_argument("--dataset-path", default=None)
    benchmark_run.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    benchmark_run.add_argument("--sample-n", type=int, default=None, help="Optional random sample size after loading the dataset.")
    benchmark_run.add_argument("--random-seed", type=int, default=20260529, help="Random seed used with --sample-n.")
    benchmark_run.add_argument("--max-workers", type=int, default=64, help="Maximum concurrent benchmark requests.")
    benchmark_run.add_argument("--output-root", default="runs")
    benchmark_run.add_argument("--require-api", action="store_true")
    benchmark_run.add_argument("--locomo-top-k", type=int, default=8, help="Final LoCoMo memory cards passed to ours QA.")
    benchmark_run.add_argument("--locomo-coarse-k", type=int, default=32, help="Coarse LoCoMo retrieval pool before reranking.")
    benchmark_run.add_argument("--top-k-sweep", default=None, help="Comma-separated LoCoMo ours top-k sweep, e.g. 8,16,32.")
    benchmark_run.add_argument("--judge-answers", action="store_true", help="Run the frozen blinded semantic-correctness judge.")
    benchmark_run.add_argument("--sample-manifest", default=None, help="Frozen JSON sample-ID manifest; disables resampling.")
    benchmark_run.add_argument(
        "--timeline-card-granularity",
        choices=("turn", "session_chunk"),
        default="turn",
        help="MediMem timeline memory-card unit; defaults to the historical per-turn behavior.",
    )
    benchmark_run.add_argument(
        "--timeline-card-max-chars",
        type=int,
        default=4000,
        help="Maximum visible characters per session_chunk card; ignored for turn cards.",
    )
    benchmark_run.add_argument(
        "--timeline-retriever",
        choices=("lexical", "hybrid_bge", "hierarchical_bge"),
        default="lexical",
        help="Opt-in timeline-card retriever; hierarchical_bge adds session-to-turn evidence packing.",
    )
    benchmark_run.add_argument(
        "--timeline-embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Frozen sentence-transformers encoder used only by hybrid_bge.",
    )
    benchmark_run.add_argument(
        "--timeline-semantic-rrf-weight",
        type=float,
        default=1.0,
        help="Semantic rank weight in hybrid_bge reciprocal-rank fusion.",
    )
    benchmark_run.add_argument(
        "--timeline-embedding-window-tokens",
        type=int,
        default=256,
        help="Deterministic encoder-token window for long timeline cards.",
    )
    benchmark_run.add_argument(
        "--timeline-hierarchical-parent-k",
        type=int,
        default=8,
        help="Session-card candidates retained by hierarchical_bge coarse retrieval.",
    )
    benchmark_run.add_argument(
        "--timeline-hierarchical-bundle-k",
        type=int,
        default=8,
        help="Fine-grained turn bundles passed to QA by hierarchical_bge.",
    )
    benchmark_run.add_argument(
        "--timeline-hierarchical-neighbor-radius",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether each hierarchical anchor may include one adjacent turn.",
    )
    benchmark_run.add_argument(
        "--timeline-hierarchical-bundle-max-chars",
        type=int,
        default=900,
        help="Maximum visible characters in each hierarchical evidence bundle.",
    )
    benchmark_run.set_defaults(func=cmd_benchmark_run)
    locomo_parallel = benchmark_sub.add_parser("run-locomo-parallel")
    locomo_parallel.add_argument("--methods-a", default="direct,amem,memoryos,medimem")
    locomo_parallel.add_argument("--methods-b", default="meminsight,gmemory,ddo")
    locomo_parallel.add_argument("--base-url-a", default=None)
    locomo_parallel.add_argument("--base-url-b", default=None)
    locomo_parallel.add_argument("--dataset-path", default=None)
    locomo_parallel.add_argument("--sample-n", type=int, default=1000)
    locomo_parallel.add_argument("--random-seed", type=int, default=20260606)
    locomo_parallel.add_argument("--max-workers-per-queue", type=int, default=48)
    locomo_parallel.add_argument("--output-root", default="runs")
    locomo_parallel.add_argument("--require-api", action="store_true")
    locomo_parallel.add_argument("--locomo-top-k", type=int, default=8)
    locomo_parallel.add_argument("--locomo-coarse-k", type=int, default=32)
    locomo_parallel.set_defaults(func=cmd_benchmark_run_locomo_parallel)
    benchmark_status_parser = benchmark_sub.add_parser("status")
    benchmark_status_parser.set_defaults(func=cmd_benchmark_status)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
