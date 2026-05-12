from __future__ import annotations

import argparse
from pathlib import Path

from .data_builder import build_dataset
from .io_utils import ensure_dir, read_jsonl, write_jsonl
from .optimizer import optimize
from .schemas import validate_case


def cmd_data_build(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.output).parent)
    notes_path = str(Path(args.output).with_suffix(".notes.txt"))
    cases = build_dataset(args.n, args.output, notes_path=notes_path)
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
    )
    print(f"run_dir={run_dir}")


def cmd_sample(args: argparse.Namespace) -> None:
    cases = read_jsonl(args.dataset)
    write_jsonl(args.output, cases[: args.n])
    print(f"wrote {min(args.n, len(cases))} rows to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mem-ehr")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build")
    build.add_argument("--n", type=int, default=10)
    build.add_argument("--output", default="data/processed/samples.jsonl")
    build.set_defaults(func=cmd_data_build)
    validate = data_sub.add_parser("validate")
    validate.add_argument("--dataset", default="data/processed/samples.jsonl")
    validate.set_defaults(func=cmd_data_validate)
    sample = data_sub.add_parser("sample")
    sample.add_argument("--dataset", default="data/processed/samples.jsonl")
    sample.add_argument("--output", default="data/processed/sample_small.jsonl")
    sample.add_argument("--n", type=int, default=3)
    sample.set_defaults(func=cmd_sample)

    opt = sub.add_parser("optimize")
    opt.add_argument("--dataset", default="data/processed/samples.jsonl")
    opt.add_argument("--max-rounds", type=int, default=10, help="0 means no explicit round cap.")
    opt.add_argument("--continuous", action="store_true", help="Keep iterating after checkpoints until a win or blocker.")
    opt.add_argument("--require-api", action="store_true", help="Stop if DeepSeek API is unavailable instead of using fallback.")
    opt.add_argument("--sleep-s", type=int, default=15)
    opt.add_argument("--max-workers", type=int, default=4)
    opt.set_defaults(func=cmd_optimize)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
