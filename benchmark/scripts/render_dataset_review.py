"""Render a saved dataset as a deterministic, read-only HTML review copy."""

import argparse
import html
import json
from pathlib import Path


def escape(value):
    return html.escape(str(value))


def pretty(value):
    return escape(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def label(row, field):
    result = row.get(field) or {}
    output = result.get("output") if not result.get("error") else None
    return output.get("label") if output else None


def comparison(row):
    primary, review = (label(row, field) for field in ("annotation", "model_review"))
    if not primary or not review:
        return "missing judgment"
    return "agreement" if primary == review else "disagreement"


def render(rows, *, decision_numbers=None, human_reviews=None):
    numbers = list(
        range(1, len(rows) + 1) if decision_numbers is None else decision_numbers
    )
    if (
        len(numbers) != len(rows)
        or any(type(n) is not int or n <= 0 for n in numbers)
        or len(set(numbers)) != len(numbers)
    ):
        raise ValueError(
            "Decision numbers must be unique positive integers, one per row"
        )
    numbered = list(zip(numbers, rows))

    def human_review(number, row):
        return (human_reviews or {}).get(number, row.get("human_review"))

    paired = any(row.get("model_review") is not None for row in rows)
    valid = [row for row in rows if row["request"] is not None]
    reviewed = sum(human_review(n, row) is not None for n, row in numbered)
    sources = [row.get("source") or {} for row in rows]
    tasks = {(s.get("suite_name"), s.get("user_task_id")) for s in sources}
    trajectories = {s.get("file") for s in sources}
    task_headers = all(
        isinstance(s.get(key), str) and s[key]
        for s in sources
        for key in ("suite_name", "user_task_id")
    )
    parts = [
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset review</title>
<style>
:root{color-scheme:light dark}
body{font-family:system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:light-dark(#f3f4f6,#242628);padding:1rem;border-radius:6px}
article{border-top:1px solid light-dark(#ccd0d5,#555);padding:1.5rem 0}
summary{cursor:pointer;padding:.6rem 0}code{overflow-wrap:anywhere}
li{margin:.5rem 0}a{color:light-dark(#1552a4,#98c4ff)}
h2{font-size:1.2rem}h3{font-size:1rem}p{line-height:1.5}
.task-heading{font-size:1.5rem;margin-top:2.5rem}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:.6rem;border-bottom:1px solid light-dark(#ccd0d5,#555)}
.table-wrap{overflow-x:auto}.status{padding:.6rem;border-left:4px solid #78828c}
.disagreement{border-color:#c47910;background:light-dark(#fff3dc,#392c16)}
.missing-judgment{border-color:#b74848;background:light-dark(#ffecec,#3a2020)}
:target{outline:2px solid light-dark(#1552a4,#98c4ff);outline-offset:4px}
</style></head><body><h1>Dataset review</h1>""",
        f"<p>{len(rows)} decisions · {len(tasks)} tasks · "
        f"{len(trajectories)} trajectories · {reviewed} human reviewed.</p>",
        "<p>Primary model labels and saved human reviews are shown separately. "
        "Full pre-call history remains available under each decision. "
        + (
            "Decision numbers follow dataset row order; "
            if decision_numbers is None
            else "Original dataset decision numbers are preserved; "
        )
        + "this copy does not edit the dataset.</p>",
    ]
    if human_reviews is not None:
        parts.append(
            "<p><strong>Human verdicts include the latest review-file overrides for "
            "display only. They have not been imported into the saved dataset.</strong></p>"
        )
    if paired:
        parts.append('<section id="comparison-index"><h2>Compare judgments</h2>')
        for field, title in (
            ("annotation", "Primary"),
            ("model_review", "Model review"),
        ):
            models = sorted(
                {
                    row[field]["model"]
                    for row in rows
                    if (row.get(field) or {}).get("model")
                }
            )
            parts.append(
                f"<p>{title} model: <code>{escape(', '.join(models) or 'not recorded')}</code></p>"
            )
        for status, title in (
            ("disagreement", "Disagreements"),
            ("missing judgment", "Missing judgments"),
            ("agreement", "Agreements"),
        ):
            selected = [(n, r) for n, r in numbered if comparison(r) == status]
            if status == "agreement":
                parts.append(f"<details><summary>{title} ({len(selected)})</summary>")
            else:
                parts.append(f"<h3>{title} ({len(selected)})</h3>")
            parts.append(
                '<div class="table-wrap"><table><thead><tr><th>Decision</th><th>Primary</th><th>Model review</th><th>Human review</th></tr></thead><tbody>'
            )
            for n, row in selected:
                human = (human_review(n, row) or {}).get("label") or "unreviewed"
                parts.append(
                    f'<tr><td><a href="#decision-{n}">Decision {n}</a></td><td>{escape(label(row, "annotation") or "missing")}</td><td>{escape(label(row, "model_review") or "missing")}</td><td>{escape(human)}</td></tr>'
                )
            parts.append("</tbody></table></div>")
            if status == "agreement":
                parts.append("</details>")
        parts.append("</section>")
    shared = {}
    for field, title in (
        ("policy", "Application authorization policy"),
        ("tool_definitions", "Tool definitions"),
    ):
        values = {row["request"][field] for row in valid}
        if len(values) == 1:
            shared[field] = values.pop()
            parts.append(
                f"<details><summary>{title} (shared)</summary>"
                f"<pre>{escape(shared[field])}</pre></details>"
            )
    parts.append("<h2>Decision index</h2><ol>")
    for number, row in numbered:
        source = row.get("source") or {}
        name = (
            row["request"]["candidate"]["name"]
            if row["request"]
            else "extraction error"
        )
        parts.append(
            f'<li value="{number}"><a href="#decision-{number}">{escape(source.get("user_task_id", "unknown task"))}'
            f" · {escape(source.get('attack_type') or 'clean')} · {escape(name)}</a>"
            + (f" — {comparison(row).title()}" if paired else "")
            + "</li>"
        )
    parts.append("</ol>")
    previous_task = None
    for number, row in numbered:
        if task_headers:
            source = row["source"]
            task = (source["suite_name"], source["user_task_id"])
            if task != previous_task:
                parts.append(
                    '<h2 class="task-heading">User task — '
                    f"{escape(task[0])} / {escape(task[1])}</h2>"
                )
                previous_task = task
        request = row["request"]
        name = request["candidate"]["name"] if request else "extraction error"
        parts.append(
            f'<article id="decision-{number}"><h2>Decision {number} — {escape(name)}</h2>'
            f"<p><code>{escape(row['id'])}</code></p>"
        )
        if paired:
            status = comparison(row)
            parts.append(
                f'<p class="status {status.replace(" ", "-")}"><strong>{status.title()}</strong> · Primary: {escape(label(row, "annotation") or "missing")} · Model review: {escape(label(row, "model_review") or "missing")}</p><p><a href="#comparison-index">Back to comparison</a></p>'
            )
        if request is None:
            parts.append(f"<pre>{escape(row['extraction_error'])}</pre>")
        else:
            parts.append(
                f"<h3>Proposed call</h3><pre>{pretty(request['candidate'])}</pre>"
            )
            for field, title in (
                ("policy", "Policy"),
                ("tool_definitions", "Tool definitions"),
            ):
                if field not in shared:
                    parts.append(
                        f"<details><summary>{title}</summary>"
                        f"<pre>{escape(request[field])}</pre></details>"
                    )
        for field, title in (
            ("annotation", "Primary annotation"),
            ("model_review", "Model review"),
        ):
            result = row.get(field)
            if field == "model_review" and result is None and not paired:
                continue
            output = (result or {}).get("output") if label(row, field) else None
            model = (result or {}).get("model")
            if model:
                title += f" ({model})"
            if output:
                parts.append(
                    f"<h3>{title}: {escape(output['label'])}</h3>"
                    f"<p>{escape(output['rationale'])}</p>"
                    "<details><summary>Annotation evidence references</summary>"
                    f"<pre>{pretty(output['evidence'])}</pre></details>"
                )
            else:
                error = (result or {}).get("error")
                parts.append(
                    f"<h3>{title}: {'operational error' if error else 'pending'}</h3>"
                    + (f"<pre>{pretty(error)}</pre>" if error else "")
                )
        review = human_review(number, row)
        parts.append(
            f"<h3>Human review: {escape(review['label'])}</h3><pre>{pretty(review)}</pre>"
            if review
            else "<p>Human review: pending.</p>"
        )
        if request:
            parts.append(
                f"<details><summary>Full pre-call history — {len(request['messages'])} messages"
                "</summary>"
            )
            calls = {}
            for message in request["messages"]:
                for call in message["tool_calls"]:
                    calls[call["id"]] = call["name"]
                title = " · ".join(message[k] for k in ("id", "role", "origin"))
                if message["role"] == "tool":
                    name = calls.get(message["tool_call_id"])
                    if name:
                        title += " · " + name
                parts.append(f"<h3>{escape(title)}</h3><pre>{pretty(message)}</pre>")
            parts.append(
                "</details><p>Full available context retained: "
                f"{escape(request['context_complete'])}.</p>"
            )
        parts.append('<a href="#">Back to index</a></article>')
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Default: review.html beside the dataset"
    )
    args = parser.parse_args()
    output = args.output or args.dataset.with_name("review.html")
    if output.resolve() == args.dataset.resolve():
        parser.error("Output must differ from the dataset")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    output.write_text(render(rows))
    print(output)


if __name__ == "__main__":
    main()
