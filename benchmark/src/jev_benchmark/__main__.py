import argparse
import asyncio
from importlib.resources import files
from pathlib import Path

from jev_guard import load_question, load_settings

from . import run_benchmark
from .report import write_comparison, write_report

GUARD_RESOURCES = files("jev_guard") / "resources"


def main():
    parser = argparse.ArgumentParser(
        description="Sequential guard comparison and offline reports"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    report = commands.add_parser("report")
    compare = commands.add_parser("compare")
    compare.add_argument("--run", action="append", type=Path, required=True)
    for command in (run, report, compare):
        command.add_argument("dataset", type=Path, help="Saved dataset.jsonl")
        command.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--backend",
        action="append",
        required=True,
        help="Profile name; repeat in execution order",
    )
    run.add_argument("--guards", type=Path, default=GUARD_RESOURCES / "guards.toml")
    run.add_argument("--question", type=Path, default=GUARD_RESOURCES / "guard-v1.json")
    run.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            write_comparison(args.dataset, args.run, args.output)
            return 0
        if args.command == "report":
            write_report(args.dataset, args.output)
            return 0
        summary = asyncio.run(
            run_benchmark(
                args.dataset,
                args.output,
                load_settings(args.guards, args.backend),
                load_question(args.question),
                limit=args.limit,
            )
        )
        return 1 if summary["errors"] or summary["pending"] else 0
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Interrupted; saved outcomes resume without replay. An in-flight call may repeat.\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
