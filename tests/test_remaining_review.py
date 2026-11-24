"""Current verdicts and compact comparisons must preserve review evidence."""

import copy
import importlib.util
import re
from pathlib import Path

import pytest
from test_disagreement_groups import Links, row

spec = importlib.util.spec_from_file_location(
    "remaining_review",
    Path(__file__).resolve().parents[1]
    / "benchmark/scripts/render_remaining_review.py",
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
HEADER = "decision,verdict,optional_reasoning\n"


def test_review_file_overrides_and_unchanged_preserve_source_and_rationale():
    rows = [row(1, "unauthorized"), row(2, "authorized"), row(3)]
    before = copy.deepcopy(rows)
    reviews, supplied = review.read_reviews(
        HEADER
        + "1,modified:authorized,Updated, with an unquoted comma\n"
        + "2,unchanged,exists\n3,insufficient_evidence,Needs clarification\n",
        rows,
    )
    assert reviews[1]["label"] == "authorized"
    assert reviews[1]["rationale"] == "Updated, with an unquoted comma"
    assert reviews[2] == rows[1]["human_review"]
    assert reviews[3]["label"] == "insufficient_evidence"
    assert supplied[1] == {"line": 2, "verdict": "modified:authorized"}
    output = review.renderer.render(
        [rows[0]], decision_numbers=[1], human_reviews=reviews
    )
    assert "Human review: authorized" in output
    assert "Human review: unauthorized" not in output
    assert "Updated, with an unquoted comma" in output
    assert "display only" in output
    assert rows == before


@pytest.mark.parametrize(
    "body",
    [
        "1,authorized,ok\n1,unauthorized,duplicate",
        "0,authorized,out of bounds",
        "2,authorized,out of bounds",
        "1,??,unresolved",
        "1,unchanged,no prior judgment",
        "1,authorized",
    ],
)
def test_invalid_review_file_cannot_silently_remove_pending_cases(body):
    with pytest.raises(ValueError):
        review.read_reviews(HEADER + body, [row(1)])


def test_membership_uses_effective_reviews_and_policy_compatible_references():
    rows = [row(1), row(2), row(3, "authorized")] + [row(n) for n in range(4, 9)]
    rows[3]["model_review"]["output"]["label"] = "authorized"
    rows[4]["model_review"] = {"output": None, "error": {"kind": "Timeout"}}
    rows[5]["request"] = None
    rows[6]["request"]["tool_definitions"] += " with different semantics"
    rows[7]["request"]["policy"] += " with different policy"
    before = copy.deepcopy(rows)
    reviews, _ = review.read_reviews(HEADER + "2,unauthorized,Reviewed now", rows)
    batches = review.build_batches(rows, reviews)
    assert sorted(n for b in batches for n in b["members"]) == [1, 7, 8]
    assert {tuple(b["members"]): b["reference"] for b in batches} == {
        (1,): 2,
        (7,): None,
        (8,): None,
    }
    assert batches == review.build_batches(rows, reviews)
    assert rows == before


def test_comparison_prefers_shared_authority_then_arguments_without_using_labels():
    rows = [row(1, "unauthorized"), row(2, "authorized"), row(3)]
    rows[1]["request"]["candidate"]["arguments"]["target"] = "y"
    rows[2]["request"]["candidate"]["arguments"]["target"] = "y"
    reviews, _ = review.read_reviews(HEADER, rows)
    assert review.build_batches(rows, reviews)[0]["reference"] == 2
    reviews[1]["label"], reviews[2]["label"] = "authorized", "unauthorized"
    assert review.build_batches(rows, reviews)[0]["reference"] == 2
    rows[1]["request"]["messages"][0]["origin"] = "external"
    assert review.build_batches(rows, reviews)[0]["reference"] == 1


def test_compact_html_exposes_omitted_fields_destinations_and_context_changes():
    rows = [row(1, "unauthorized"), row(2), row(3)]
    rows[0]["request"]["candidate"]["arguments"].update(
        location=None, enabled=False, amount=0
    )
    rows[1]["request"]["candidate"]["arguments"].update(enabled=0, amount=0.0)
    attack = "<script>forged destination</script>" * 10
    rows[1]["request"]["candidate"]["arguments"]["target"] = attack
    rows[1]["request"]["messages"][0]["origin"] = "external"
    reviews, _ = review.read_reviews(HEADER, rows)
    batches = review.build_batches(rows, reviews)
    output = review.render_remaining(rows, reviews, batches)
    assert re.findall(r'id="review-(\d+)"', output) == ["2", "3"]
    assert '<td class="changed"><span class="omitted">omitted</span></td>' in output
    assert "<code>null</code>" in output and "<code>false</code>" in output
    assert '<td class="changed"><code>0</code></td>' in output
    assert '<td class="changed"><code>0.0</code></td>' in output
    assert "Context 2: application/user instructions" in output
    assert "Context differences from Decision 1" in output
    assert "&quot;origin&quot;: &quot;external&quot;" in output
    assert "<script>" not in output
    assert review.escape(attack) in output  # Full long value remains available.
    assert Links(output).case_targets == {"decision-1", "decision-2", "decision-3"}
    assert Links(output).case_targets <= Links(review.renderer.render(rows)).ids
    assert output == review.render_remaining(rows, reviews, batches)


def test_unreviewed_comparison_still_appears_exactly_once_as_pending():
    rows = [row(1)]
    output = review.render_remaining(rows, {}, review.build_batches(rows, {}))
    assert re.findall(r'id="review-(\d+)"', output) == ["1"]
    assert 'class="reference"' not in output
    assert "Unreviewed comparison" in output


def test_date_and_error_evidence_stays_paired_with_the_original_call():
    rows = [row(1)]
    request = rows[0]["request"]
    request["candidate"]["arguments"]["next_payment_date"] = "2023-11-14"
    request["messages"] += [
        {
            "id": "m000001",
            "role": "assistant",
            "origin": "agent",
            "content": "",
            "tool_calls": [
                {"id": "c000000", "name": "get_current_day", "arguments": {}},
                {"id": "c000001", **copy.deepcopy(request["candidate"])},
            ],
        },
        {
            "id": "m000002",
            "role": "tool",
            "origin": "external",
            "tool_call_id": "c000000",
            "content": '"2026-10-27"',
            "error": None,
        },
        {
            "id": "m000003",
            "role": "tool",
            "origin": "external",
            "tool_call_id": "c000001",
            "content": "Rejected <script>attack</script>",
            "error": "Invalid date",
        },
    ]
    facts = review.observations(request)
    assert facts["date_lookups"][0]["message"] == "m000002"
    assert facts["prior_errors"] == facts["same_call_results"]
    assert facts["prior_errors"][0]["call"]["id"] == "c000001"
    output = review.render_remaining(rows, {}, review.build_batches(rows, {}))
    assert "Current-day lookup: recorded below" in output
    assert "Date lookup results (1)" in output and "Prior tool errors (1)" in output
    assert "Earlier identical-call results (1)" in output
    assert "&lt;script&gt;attack&lt;/script&gt;" in output
    request["messages"] = request["messages"][:1]
    output = review.render_remaining(rows, {}, review.build_batches(rows, {}))
    assert "Current-day lookup: none in this history" in output
