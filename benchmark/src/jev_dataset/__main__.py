"""Run with python -m jev_dataset prepare/annotate."""

import argparse
import asyncio
import json
from importlib.resources import files
from pathlib import Path

from .adapter import prepare
from .pipeline import initialize, run_annotations

GUARD_RESOURCES = files("jev_guard") / "resources"
RESOURCES = files("jev_dataset") / "resources"


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative")
    return number


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Transform JSON logs without inference")
    prep.add_argument("run_dir", type=Path)
    prep.add_argument("--definitions", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument(
        "--policy", type=Path, default=GUARD_RESOURCES / "authorization-policy-v1.txt"
    )
    prep.add_argument("--task", nargs=2, metavar=("SUITE", "TASK"))
    prep.add_argument("--attack", nargs=2, metavar=("ATTACK", "INJECTION_TASK"))
    prep.add_argument("--limit", type=nonnegative)
    prep.add_argument(
        "--selection", type=Path, help="Complete clean/attack trajectories from TOML"
    )
    annotation = commands.add_parser(
        "annotate", help="Annotate saved requests and resume progress"
    )
    annotation.add_argument("dataset_dir", type=Path)
    annotation.add_argument("--models", type=Path, default=RESOURCES / "models.toml")
    annotation.add_argument(
        "--rubric", type=Path, default=RESOURCES / "annotation-rubric-v1.txt"
    )
    annotation.add_argument(
        "--review", action="store_true", help="Independently call the quality profile"
    )
    annotation.add_argument(
        "--review-limit",
        type=nonnegative,
        help="Review first N decision rows; requires --review",
    )
    annotation.add_argument(
        "--concurrency",
        type=positive,
        default=1,
        help="Maximum in-flight annotation calls per stage (default: 1)",
    )
    retry_options = annotation.add_mutually_exclusive_group()
    retry_options.add_argument(
        "--max-attempts",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Legacy transport-only retry limit (default: 3 attempts)",
    )
    retry_options.add_argument(
        "--retries",
        type=nonnegative,
        help="Extra dispatches per decision/annotator for transport or invalid judgments; counts interrupted calls",
    )
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            if args.output.resolve().is_relative_to(args.run_dir.resolve()):
                raise ValueError("Output must be outside the source run directory")
            rows, metadata = prepare(
                args.run_dir,
                args.definitions,
                args.policy.read_text(),
                task=args.task,
                attack=args.attack,
                limit=args.limit,
                selection=args.selection,
            )
            initialize(args.output, rows, metadata)
            print(json.dumps(metadata["summary"], sort_keys=True))
            return (
                1
                if metadata["summary"]["file_errors"]
                or metadata["summary"]["extraction_errors"]
                or metadata["summary"]["source_errors"]
                or metadata["summary"].get("selection_errors")
                else 0
            )
        if args.review_limit is not None and not args.review:
            raise ValueError("--review-limit requires --review")
        from jev_llm import load_profile

        result = asyncio.run(
            run_annotations(
                args.dataset_dir,
                args.rubric.read_text(),
                load_profile(str(args.models), "annotation"),
                review_profile=load_profile(str(args.models), "quality")
                if args.review
                else None,
                review_limit=args.review_limit,
                max_attempts=args.max_attempts,
                retries=args.retries,
                concurrency=args.concurrency,
            )
        )
        return 1 if result["stopped"] or result["failed_decisions"] else 0
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(
            130,
            "Interrupted; completed decisions and attempts are saved. Resume with the same command.\n",
        )


if __name__ == "__main__":
    raise SystemExit(main())
