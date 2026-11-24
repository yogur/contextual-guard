"""Confirmed review layers must preserve evidence and make overrides explicit."""

import copy
import importlib.util
from pathlib import Path

import pytest
from test_disagreement_groups import row

spec = importlib.util.spec_from_file_location(
    "finalize_review",
    Path(__file__).resolve().parents[1] / "benchmark/scripts/reproduce_reviews.py",
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
HEADER = "decision,verdict,optional_reasoning\n"


def test_later_overrides_and_unchanged_keep_verbatim_rationales_and_all_evidence():
    rows = [row(1), row(2, "authorized"), row(3)]
    before = copy.deepcopy(rows)
    merged, events, summaries = review.merge_reviews(
        rows,
        [
            (
                "first.txt",
                HEADER + "1,authorized,First reason\n3,insufficient_evidence,",
            ),
            (
                "last.txt",
                HEADER
                + "1,modified:unauthorized,Revised, with comma\n2,unchanged,exists\n3,unchanged,exists",
            ),
        ],
    )
    assert merged[0]["human_review"]["label"] == "unauthorized"
    assert merged[0]["human_review"]["rationale"] == "Revised, with comma"
    assert merged[1]["human_review"] == rows[1]["human_review"]
    assert merged[2]["human_review"]["rationale"] == ""
    assert events[2]["before"]["rationale"] == "First reason"
    assert events[2]["after"] == merged[0]["human_review"]
    assert summaries[1] == {
        "file": "last.txt",
        "entries": 3,
        "modified": 1,
        "unchanged": 2,
    }
    assert review.non_human_hash(rows) == review.non_human_hash(merged)
    assert rows == before


@pytest.mark.parametrize(
    "body,human",
    [
        ("1,authorized,Missing override marker", "unauthorized"),
        ("1,modified:authorized,Missing original review", None),
        ("1,authorized,Still uncertain ??", None),
        ("1,authorized,TODO discuss", None),
        ("1,authorized,First\n1,unauthorized,Duplicate", None),
    ],
)
def test_unsettled_or_implicit_changes_cannot_be_imported(body, human):
    with pytest.raises(ValueError):
        review.merge_reviews([row(1, human)], [("verdicts.txt", HEADER + body)])
