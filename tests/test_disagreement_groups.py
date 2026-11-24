"""Grouping must preserve cases and expose authorization-relevant context changes."""

import copy
import importlib.util
from html.parser import HTMLParser
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "disagreement_groups",
    Path(__file__).resolve().parents[1] / "benchmark/scripts/group_disagreements.py",
)
grouping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grouping)


def row(number, human=None):
    def judgment(label, model):
        return {
            "model": model,
            "state": "completed",
            "output": {
                "label": label,
                "rationale": "<script>forged instruction</script>",
                "evidence": [{"target": "m000000", "pointer": "/content"}],
            },
            "error": None,
        }

    return {
        "id": f"trajectory-{number}#proposal",
        "source": {
            "suite_name": "workspace",
            "user_task_id": "user_task_1",
            "file": f"trajectory-{number}.json",
        },
        "request": {
            "schema_version": "guard-request.v1",
            "policy": "Require authorization.",
            "tool_definitions": "read(target: str)",
            "candidate": {"name": "read", "arguments": {"target": "x"}},
            "messages": [
                {
                    "id": "m000000",
                    "role": "user",
                    "origin": "user",
                    "content": "Read x.",
                    "tool_calls": [],
                    "tool_call_id": None,
                    "error": None,
                }
            ],
            "context_complete": True,
        },
        "annotation": judgment("authorized", "Grok"),
        "model_review": judgment("unauthorized", "Luna"),
        "human_review": {"label": human, "rationale": "Reviewed", "reviewer": "user"}
        if human
        else None,
        "extraction_error": None,
    }


def test_exact_duplicates_and_reviewed_anchors_preserve_each_case():
    rows = [row(1), row(2), row(3, "authorized"), row(4)]
    rows[3]["request"]["messages"][0]["content"] = "Read x only after approval."
    before = copy.deepcopy(rows)
    levels = grouping.group_cases(rows)
    assert rows == before
    first = levels["exact_request"][0]
    assert [e["decision"] for e in first["members"]] == [1, 2]
    assert [e["decision"] for e in first["reviewed_matches"]] == [3]
    assert first["representative"] == 3
    broader = levels["same_task_and_call"]
    assert len(broader) == 1
    assert [e["decision"] for e in broader[0]["members"]] == [1, 2, 4]
    assert len(broader[0]["context_variants"]) == 2
    assert {n for v in broader[0]["context_variants"] for n in v["decisions"]} == {
        1,
        2,
        3,
        4,
    }


def test_full_request_fields_are_not_normalized_or_omitted():
    rows = [row(n) for n in range(1, 7)]
    rows[1]["request"]["policy"] += " Require separate approval."
    rows[2]["request"]["tool_definitions"] += " May disclose data."
    rows[3]["request"]["messages"][0]["origin"] = "external"
    rows[4]["request"]["messages"][0]["content"] += " "
    rows[5]["request"]["context_complete"] = False
    levels = grouping.group_cases(rows)
    assert len(levels["exact_request"]) == 6
    assert len(levels["same_task_and_call"]) == 1
    for changed in rows[1:]:
        assert grouping.context_diff(rows[0]["request"], changed["request"], 1, 2)


def test_task_call_key_includes_domain_task_and_argument_values():
    rows = [row(n) for n in range(1, 5)]
    rows[1]["source"]["suite_name"] = "banking"
    rows[2]["source"]["user_task_id"] = "user_task_2"
    rows[3]["request"]["candidate"]["arguments"]["target"] = "y"
    levels = grouping.group_cases(rows)
    assert len(levels["same_task_and_call"]) == 4
    # Source identity is outside the model input; three requests remain exact.
    assert len(levels["exact_request"]) == 2
    assert sorted(len(g["members"]) for g in levels["exact_request"]) == [1, 3]


def test_only_unreviewed_valid_label_disagreements_are_members():
    rows = [row(1), row(2, "authorized"), row(3), row(4)]
    rows[2]["model_review"]["output"]["label"] = "authorized"
    rows[3]["model_review"] = {"output": None, "error": {"kind": "Timeout"}}
    levels = grouping.group_cases(rows)
    for groups in levels.values():
        assert [e["decision"] for g in groups for e in g["members"]] == [1]
        assert [e["decision"] for g in groups for e in g["reviewed_matches"]] == [2]


def test_conflicting_exact_human_labels_are_exposed():
    levels = grouping.group_cases(
        [row(1), row(2, "authorized"), row(3, "unauthorized")]
    )
    assert levels["exact_request"][0]["conflicting_exact_human_labels"]
    assert not levels["same_task_and_call"][0]["conflicting_exact_human_labels"]


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.case_targets = set(), set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("cases.html#"):
            self.case_targets.add(attrs["href"].split("#", 1)[1])


def test_html_escapes_source_text_and_links_every_variant_to_full_history():
    rows = [row(1), row(2), row(3, "authorized")]
    rows[1]["request"]["messages"][0]["content"] += " <script>ignore rules</script>"
    levels = grouping.group_cases(rows)
    summary = {
        "unreviewed_disagreements": 2,
        "exact_request_groups": 2,
        "same_task_and_call_groups": 1,
    }
    index = grouping.render_groups(rows, levels, summary)
    cases = grouping.renderer.render(rows)
    assert "<script>" not in index + cases
    assert "&lt;script&gt;" in index and "&lt;script&gt;" in cases
    assert "Context differences: Decision 3 → 2" in index
    assert 'class="added"' in index
    assert Links(index).case_targets <= Links(cases).ids
    assert Links(index).case_targets == {"decision-1", "decision-2", "decision-3"}
