"""Review navigation must match persisted labels and original decision identities."""

import importlib.util
from html.parser import HTMLParser
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "render_dataset_review",
    Path(__file__).resolve().parents[1] / "benchmark/scripts/render_dataset_review.py",
)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def row(primary="authorized", quality="unauthorized", *, error=None):
    def result(label, model):
        return {
            "model": model,
            "output": {
                "label": label,
                "rationale": "<script>alert(1)</script>",
                "evidence": [],
            },
            "error": None,
        }

    return {
        "id": "trace#proposal",
        "source": {"suite_name": "suite", "user_task_id": "task"},
        "request": {
            "candidate": {"name": "read", "arguments": {}},
            "policy": "policy",
            "tool_definitions": "tools",
            "messages": [],
            "context_complete": True,
        },
        "annotation": result(primary, "bedrock/global.primary"),
        "model_review": result(quality, "bedrock/global.quality")
        if quality
        else {"error": error}
        if error
        else None,
        "human_review": None,
    }


class Links(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids, self.targets = set(), []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#decision-"):
            self.targets.append(attrs["href"][1:])


def test_comparison_groups_missing_errors_and_preserves_subset_numbers():
    rows = [
        row(),
        row(quality=None, error={"kind": "ServiceUnavailableError"}),
        row(quality="authorized"),
    ]
    rows[0]["human_review"] = {"label": "unauthorized"}
    output = renderer.render(rows, decision_numbers=[90, 29, 48])
    assert "Disagreements (1)" in output
    assert "Missing judgments (1)" in output
    assert "Agreements (1)" in output
    assert "Original dataset decision numbers are preserved" in output
    for number in (90, 29, 48):
        assert f"Decision {number} — read" in output
        assert f'<li value="{number}">' in output
    parsed = Links(output)
    assert set(parsed.targets) == {"decision-90", "decision-29", "decision-48"}
    assert set(parsed.targets) <= parsed.ids
    assert "decision-1" not in parsed.ids
    assert "bedrock/global.primary" in output and "bedrock/global.quality" in output
    assert "Human review: unauthorized" in output
    assert "ServiceUnavailableError" in output
    assert "<script>" not in output
    assert "&lt;script&gt;" in output
    assert output == renderer.render(rows, decision_numbers=[90, 29, 48])


def test_single_annotator_does_not_imply_a_missing_second_judgment():
    output = renderer.render([row(quality=None)])
    assert "Compare judgments" not in output
    assert "Decision 1 — read" in output
    assert "Human review: pending" in output


@pytest.mark.parametrize("numbers", [[1, 1], [0, 2], [1], [True, 2]])
def test_invalid_decision_numbers_rejected(numbers):
    with pytest.raises(ValueError, match="Decision numbers"):
        renderer.render([row(), row()], decision_numbers=numbers)
