"""Verify installed packages and published benchmark evidence offline."""

import argparse
import asyncio
import hashlib
import importlib.resources
import io
import json
import os
import runpy
import shutil
import socket
import sys
import tarfile
from contextlib import redirect_stdout
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise AssertionError(
        "Release verification forbids inference, networking and pricing"
    )


def run_script(path, *args):
    sys.argv = [str(path), *map(str, args)]
    with redirect_stdout(io.StringIO()):
        runpy.run_path(str(path), run_name="__main__")


def cli(*args, module="jev_benchmark"):
    sys.argv = [module, *map(str, args)]
    try:
        with redirect_stdout(io.StringIO()):
            runpy.run_module(module, run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0


def normalized(value, root):
    # Only historical/relocated root paths change; all other report bytes matter.
    value = json.dumps(value, sort_keys=True)
    for prefix in (str(root), "<DATASET_ROOT>"):
        value = value.replace(prefix, "<EVIDENCE_ROOT>")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    repo, artifacts, work = (
        p.resolve() for p in (args.repo, args.artifacts, args.work)
    )
    if work.exists() or work.is_relative_to(repo):
        raise ValueError("Use a new work directory outside both checkouts")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = (
        forbidden
    )
    import litellm
    import typesafe_sdk

    import jev_benchmark
    import jev_contracts
    import jev_guard
    from jev_benchmark import run_benchmark

    typesafe_sdk.AsyncTypeSafeClient = forbidden
    litellm.acompletion = litellm.aresponses = litellm.completion_cost = forbidden
    for module in (jev_guard, jev_contracts, jev_benchmark):
        assert "site-packages" in module.__file__, module.__file__
        assert not Path(module.__file__).resolve().is_relative_to(repo)
    work.mkdir(parents=True)
    sums = (repo / "benchmark/datasets/v1/SHA256SUMS").read_text().splitlines()
    for line in sums:
        expected, name = line.split(maxsplit=1)
        assert sha(artifacts / name) == expected, name
    extracted = {}
    for label in ("dataset", "annotation-evidence", "source"):
        root = work / label
        root.mkdir()
        archive = artifacts / f"contextual-guard-{label}-v1.tar.gz"
        with tarfile.open(archive) as tf:
            assert all(
                m.isfile() and not m.name.startswith("/") for m in tf.getmembers()
            )
            tf.extractall(root, filter="data")
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest == json.loads(
            (repo / f"benchmark/datasets/v1/{label}-manifest.json").read_text()
        )
        for item in manifest["files"]:
            assert sha(root / item["path"]) == item["sha256"], item["path"]
        extracted[label] = root
    root = extracted["dataset"]
    full = root / "runs"
    dataset = root / "dataset.jsonl"
    source = extracted["source"]
    assert (source / "selection.toml").read_bytes() == (
        repo / "benchmark/config/selection.toml"
    ).read_bytes()
    cli(
        "prepare",
        source / "runs",
        "--definitions",
        source / "definitions.json",
        "--selection",
        source / "selection.toml",
        "--output",
        work / "prepared-full",
        module="jev_dataset",
    )
    prepared = work / "prepared-full/dataset.jsonl"
    assert prepared.read_bytes() == (source / "prepared/dataset.jsonl").read_bytes()
    with prepared.open() as left, dataset.open() as right:
        for raw, labelled in zip(left, right, strict=True):
            raw, labelled = json.loads(raw), json.loads(labelled)
            for key in ("id", "source", "request", "extraction_error"):
                assert raw[key] == labelled[key], (raw["id"], key)
    run_script(
        repo / "benchmark/scripts/reproduce_reports.py",
        "--root",
        root,
        "--output",
        work / "reports",
    )
    for name in ("jev", "clef", "comparison"):
        stem = "comparison" if name == "comparison" else "report"
        for file in (stem + ".json", "reviewed-cohorts.json"):
            old = json.loads((full / name / file).read_text())
            new = json.loads((work / "reports" / name / file).read_text())
            assert normalized(old, root) == normalized(new, root), (name, file)
        for file in (stem + ".md", "reviewed-cohorts.md"):
            old = (full / name / file).read_text()
            new = (work / "reports" / name / file).read_text()
            assert normalized(old, root) == normalized(new, root), (name, file)
    guard_resources = importlib.resources.files("jev_guard") / "resources"
    question = jev_guard.load_question(guard_resources / "guard-v1.json")
    before = {p: sha(p) for p in full.rglob("*") if p.is_file()}
    for name in ("jev", "clef"):
        messages = []
        summary = asyncio.run(
            run_benchmark(
                dataset,
                full / name,
                jev_guard.load_settings(full / "settings/guards.toml", [name]),
                question,
                emit=messages.append,
            )
        )
        assert json.loads(messages[0])["planned_calls"] == {name: 0}
        assert summary["errors"] == summary["pending"] == 0
    assert all(sha(p) == digest for p, digest in before.items())
    annotation_root = extracted["annotation-evidence"]
    run_script(
        repo / "benchmark/scripts/reproduce_reviews.py",
        "--root",
        annotation_root,
        "--output",
        work / "review-export",
    )
    assert (work / "review-export/dataset.jsonl").read_bytes() == dataset.read_bytes()
    old_ledger = json.loads((dataset.parent / "human-reviews.json").read_text())
    new_ledger = json.loads((work / "review-export/human-reviews.json").read_text())
    for key in ("reviews", "events", "layers", "inherited_reviews"):
        assert old_ledger[key] == new_ledger[key], key
    initial = annotation_root / "review/initial/dataset.jsonl"
    run_script(
        repo / "benchmark/scripts/group_disagreements.py",
        initial,
        "--output",
        work / "groups",
    )
    run_script(
        repo / "benchmark/scripts/render_remaining_review.py",
        initial,
        "--reviews",
        annotation_root / "review/grouped/reviews.txt",
        "--output",
        work / "remaining",
    )
    for output, historical, keys in (
        (
            "groups",
            "grouped",
            ("levels", "summary", "case_decisions"),
        ),
        (
            "remaining",
            "remaining",
            ("batches", "summary", "case_decisions"),
        ),
    ):
        old = json.loads(
            (annotation_root / "review" / historical / "groups.json").read_text()
        )
        new = json.loads((work / output / "groups.json").read_text())
        assert all(old[key] == new[key] for key in keys), output
    for script in ("analyze.py", "build_interpretation.py"):
        run_script(
            repo / "benchmark/scripts" / script,
            "--root",
            root,
            "--output",
            work / "analysis",
        )
    for name in (
        "statistics.json",
        "examples.json",
        "case-studies.json",
        "benchmark-interpretation-counts.json",
    ):
        assert json.loads((work / "analysis" / name).read_text()) == json.loads(
            (root / "analysis" / name).read_text()
        ), name
    assert (work / "analysis/benchmark-results-interpretation.html").read_bytes() == (
        repo / "benchmark/docs/results/benchmark-results-interpretation.html"
    ).read_bytes()
    run_script(
        repo / "benchmark/scripts/derive_example.py",
        "--root",
        root,
        "--output",
        work / "derived",
    )
    original = repo / "benchmark/datasets/example"
    for path in original.rglob("*"):
        if path.is_file():
            assert (
                path.read_bytes()
                == (work / "derived" / path.relative_to(original)).read_bytes()
            )
    shutil.copytree(original, work / "example")
    sample = work / "example/dataset.jsonl"
    for name in ("jev", "clef"):
        cli("report", sample, "--output", work / "example" / name)
    cli(
        "compare",
        sample,
        "--run",
        work / "example/jev",
        "--run",
        work / "example/clef",
        "--output",
        work / "example-comparison",
    )
    inputs = repo / "benchmark/datasets/agentdojo-input"
    cli(
        "prepare",
        inputs / "runs",
        "--definitions",
        inputs / "definitions.json",
        "--output",
        work / "prepared",
        "--limit",
        "2",
        module="jev_dataset",
    )
    rows = [
        json.loads(line)
        for line in (work / "prepared/dataset.jsonl").read_text().splitlines()
    ]
    expected = [json.loads(line) for line in sample.read_text().splitlines()[:2]]
    assert [{k: r[k] for k in ("id", "source", "request")} for r in rows] == [
        {k: r[k] for k in ("id", "source", "request")} for r in expected
    ]
    run_script(
        repo / "benchmark/scripts/render_dataset_review.py",
        sample,
        "--output",
        work / "example-review.html",
    )
    run_script(repo / "examples/assess.py")
    summary = {
        "installed_guard": jev_guard.__file__,
        "installed_tooling": jev_benchmark.__file__,
        "archives_and_checksums": True,
        "report_scores_unchanged": True,
        "expected_report_changes": "Only relocated absolute paths in provenance and prose",
        "analysis_and_html_reproduced": True,
        "example_prefix_bytes_identical": True,
        "native_preparation_matches_frozen_requests": True,
        "full_source_preparation_reproduced": True,
        "review_export_and_grouping_reproduced": True,
        "completed_resume_calls": 0,
        "inference_calls": 0,
        "network_and_pricing_disabled": True,
    }
    (work / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
