"""Refresh authorized report outputs only, retaining retry history and source bytes."""

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import litellm
import typesafe_sdk

from jev_benchmark import write_json
from jev_benchmark.report import (
    build_report,
    markdown,
    write_comparison,
    write_reviewed_cohorts,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True, help="Extracted release root")
parser.add_argument("--output", type=Path, required=True, help="New report directory")
args = parser.parse_args()
ROOT = args.root.resolve()
BASE = ROOT / "runs"
DATASET = ROOT / "dataset.jsonl"
JEV, CLEF = BASE / "jev", BASE / "clef"


def forbidden(*args, **kwargs):
    raise AssertionError("Report refresh must remain offline; no inference or pricing")


socket.socket.connect = socket.socket.connect_ex = socket.create_connection = forbidden
typesafe_sdk.AsyncTypeSafeClient = forbidden
litellm.acompletion = litellm.aresponses = litellm.completion_cost = forbidden


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


old_failure = json.loads((BASE / "retry/original-failure.jsonl").read_text())
retry_attempt = json.loads((BASE / "retry/attempt.json").read_text())
assert retry_attempt["result"]["error"] is None
history = [
    {
        "backend": "jev",
        "id": old_failure["id"],
        "logical_execution_order": old_failure["execution_order"],
        "original_completed_at": old_failure["completed_at"],
        "original_error": old_failure["result"]["error"],
        "original_failed_latency_seconds": old_failure["result"]["latency_seconds"],
        "retry_completed_at": retry_attempt["completed_at"],
        "retry_latency_seconds": retry_attempt["result"]["latency_seconds"],
        "authorized_additional_assessments": 1,
        "automatic_retries": 0,
        "original_outcome_path": str(BASE / "retry/original-failure.jsonl"),
    }
]
note = (
    "Jev results include one owner-authorized retry replacing the HTTP 504 outcome "
    "at logical execution slot 6975. Original pass: 10,356 valid verdicts and one "
    "operational error; after the one additional assessment: 10,357 valid verdicts. "
    "Quality/coverage/latency summaries use the replacement result. The original "
    "failure remains preserved in retry/original-failure.jsonl. "
    "The retry timestamp extends the assessment period; that period is not uninterrupted run time."
)


def generate(destination):
    independent = build_report(DATASET, JEV)
    independent["retry_history"] = history
    independent["notes"].append(note)
    independent_dir = destination / "jev"
    independent_dir.mkdir()
    write_json(independent_dir / "report.json", independent)
    (independent_dir / "report.md").write_text(markdown(independent))
    write_reviewed_cohorts(independent, independent_dir)
    local = build_report(DATASET, CLEF)
    local_dir = destination / "clef"
    local_dir.mkdir()
    write_json(local_dir / "report.json", local)
    (local_dir / "report.md").write_text(markdown(local))
    write_reviewed_cohorts(local, local_dir)
    combined = write_comparison(DATASET, [JEV, CLEF], destination / "comparison")
    combined["retry_history"] = history
    combined["notes"].append(note)
    write_json(destination / "comparison/comparison.json", combined)
    (destination / "comparison/comparison.md").write_text(markdown(combined))
    assert (
        independent["complete"]
        and independent["counts"]["errors"] == independent["counts"]["pending"] == 0
    )
    assert independent["counts"]["completed_assessments"] == 10357
    assert (
        combined["complete"]
        and combined["counts"]["errors"] == combined["counts"]["pending"] == 0
    )
    assert combined["counts"]["completed_assessments"] == 20714
    assert (
        combined["backends"]["run1/jev"]["tracks"]
        == independent["backends"]["jev"]["tracks"]
    )
    assert (
        combined["backends"]["run1/jev"]["reviewed_cohorts"]
        == independent["backends"]["jev"]["reviewed_cohorts"]
    )
    for source in combined["sources"]:
        for name, expected in source["sha256"].items():
            assert sha(Path(source["path"]) / name) == expected
    return independent, combined


output = args.output.resolve()
if output.exists() or output.is_relative_to(ROOT):
    raise ValueError("Choose a new output directory outside the extracted evidence")
before = {p: sha(p) for p in ROOT.rglob("*") if p.is_file()}
assert (
    sha(DATASET) == "c92191ff4a2378563e4ac991b07fba66f9cb55f539ac2c5e081deb1aadb7d87f"
)
assert (
    sha(JEV / "verdicts.jsonl")
    == "569df163a891fa7cf4d806a3e79aa8cecb977dfc8f49b5504118e5a804f1d202"
)
assert (
    sha(CLEF / "verdicts.jsonl")
    == "9545f49669f773565acb39ee9f06b6dd83f7010200931203871d76f97882db1c"
)
output.mkdir(parents=True)
independent, combined = generate(output)
assert all(sha(p) == expected for p, expected in before.items())
print(
    json.dumps(
        {
            "completed": combined["counts"]["completed_assessments"],
            "errors": combined["counts"]["errors"],
            "pending": combined["counts"]["pending"],
            "source_bytes_unchanged": True,
            "retry_history_disclosed": True,
            "inference_and_network_disabled": True,
        }
    )
)
