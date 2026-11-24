"""Render remaining disagreements against current human reviews, entirely offline."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from jev_contracts import LABELS
from jev_dataset.pipeline import dumps, validate_saved

spec = importlib.util.spec_from_file_location(
    "disagreement_groups", Path(__file__).with_name("group_disagreements.py")
)
grouping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grouping)
renderer = grouping.renderer
escape, pretty = renderer.escape, renderer.pretty


def read_reviews(
    text,
    rows,
    *,
    previous=None,
    reviewer="user review file (display only; not imported)",
):
    reviews = (
        dict(previous)
        if previous is not None
        else {
            n: dict(row["human_review"])
            for n, row in enumerate(rows, 1)
            if row.get("human_review")
        }
    )
    supplied = {}
    lines = text.splitlines()
    if not lines or lines[0] != "decision,verdict,optional_reasoning":
        raise ValueError("Expected decision,verdict,optional_reasoning header")
    for line_number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        number, verdict, rationale = line.split(",", 2)
        number = int(number)
        if number in supplied or not 1 <= number <= len(rows):
            raise ValueError(f"Invalid or repeated decision at line {line_number}")
        if verdict == "unchanged":
            if number not in reviews:
                raise ValueError(f"No existing human verdict for decision {number}")
        else:
            label = verdict.removeprefix("modified:")
            if label not in LABELS:
                raise ValueError(f"Unresolved or invalid verdict at line {line_number}")
            reviews[number] = {
                "label": label,
                "rationale": rationale,
                "reviewer": reviewer,
            }
        supplied[number] = {"line": line_number, "verdict": verdict}
    return reviews, supplied


def batch_key(row):
    request = row["request"]
    return (
        row["source"]["suite_name"],
        row["source"]["user_task_id"],
        request["candidate"]["name"],
        grouping.digest(request["policy"]),
        grouping.digest(request["tool_definitions"]),
    )


def authority(request):
    return [m for m in request["messages"] if m["role"] in {"system", "user"}]


def argument_changes(before, after):
    return [
        key
        for key in sorted(before.keys() | after.keys())
        if key not in before
        or key not in after
        or dumps(before[key]) != dumps(after[key])
    ]


def build_batches(rows, reviews):
    pending, references = defaultdict(list), defaultdict(list)
    for number, row in enumerate(rows, 1):
        if row["request"] is None:
            continue
        key = batch_key(row)
        if number in reviews:
            references[key].append(number)
        elif renderer.comparison(row) == "disagreement":
            pending[key].append(number)
    batches = []
    for key, numbers in sorted(pending.items(), key=lambda p: (-len(p[1]), p[0])):

        def distance(number):
            request = rows[number - 1]["request"]
            return (
                sum(
                    authority(request) != authority(rows[n - 1]["request"])
                    for n in numbers
                ),
                sum(
                    len(
                        argument_changes(
                            request["candidate"]["arguments"],
                            rows[n - 1]["request"]["candidate"]["arguments"],
                        )
                    )
                    for n in numbers
                ),
                number,
            )

        reference = min(references[key], key=distance) if references[key] else None
        batches.append(
            {
                "id": f"batch-{len(batches) + 1}",
                "domain": key[0],
                "task": key[1],
                "tool": key[2],
                "policy_sha256": key[3],
                "definitions_sha256": key[4],
                "members": numbers,
                "reference": reference,
            }
        )
    return batches


def observations(request):
    calls, results = {}, []
    for message in request["messages"]:
        calls.update((call["id"], call) for call in message.get("tool_calls", []))
        call = calls.get(message.get("tool_call_id"))
        if message["role"] == "tool" and call:
            results.append(
                {
                    "message": message["id"],
                    "call": call,
                    "content": message.get("content"),
                    "error": message.get("error"),
                }
            )
    return {
        "date_lookups": [r for r in results if r["call"]["name"] == "get_current_day"],
        "prior_errors": [r for r in results if r["error"]],
        "same_call_results": [
            r
            for r in results
            if r["call"]["name"] == request["candidate"]["name"]
            and dumps(r["call"]["arguments"])
            == dumps(request["candidate"]["arguments"])
        ],
    }


def folded(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(raw) <= 140:
        return f"<code>{escape(raw)}</code>"
    return (
        f"<details><summary>{escape(raw[:100])}…</summary>"
        f"<pre>{pretty(value)}</pre></details>"
    )


def case_link(number):
    return f'<a href="cases.html#decision-{number}">Decision {number}</a>'


def render_remaining(rows, reviews, batches):
    count = sum(len(b["members"]) for b in batches)
    parts = [
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Remaining disagreement review</title><style>
:root{color-scheme:light dark}body{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;line-height:1.5}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:light-dark(#f3f4f6,#242628);padding:.7rem}
details{margin:.6rem 0}summary{cursor:pointer}details.batch{border:1px solid #888;padding:1rem;margin:1.3rem 0}
.batch>summary{font-weight:700}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse}
th,td{text-align:left;vertical-align:top;min-width:130px;padding:.6rem;border:1px solid #888}td code{overflow-wrap:anywhere}
.changed{background:light-dark(#fff1cf,#403319)}.reference{background:light-dark(#e9eef5,#24313e)}
a{color:light-dark(#1552a4,#98c4ff)}:target{outline:2px solid #788cb8}.omitted{font-style:italic}
</style></head><body><h1>Remaining disagreement review</h1>""",
        f"<p>{count} remaining decisions · {len(batches)} batches. Each remaining "
        "decision appears once below; reviewed comparisons are separate.</p>",
        "<p>Comparisons use the latest human review file, including overrides, for "
        "display only. The saved dataset is unchanged. Similarity does not assign "
        "a verdict to an unreviewed case.</p>",
        "<p>Amber cells differ from the comparison. Long values expand to show the "
        "complete original text. Context differences include every request field "
        "except the candidate shown in the table. Decision links open the full "
        "policy, tool definitions, history and independent model judgments.</p>",
    ]
    for batch in batches:
        numbers = batch["members"]
        base_number = batch["reference"] or numbers[0]
        base = rows[base_number - 1]["request"]
        shown = [base_number] + [n for n in numbers if n != base_number]
        arguments = [rows[n - 1]["request"]["candidate"]["arguments"] for n in shown]
        fields = sorted({key for args in arguments for key in args})
        shared = {
            k: arguments[0][k]
            for k in fields
            if all(
                k in a and k in arguments[0] and dumps(a[k]) == dumps(arguments[0][k])
                for a in arguments
            )
        }
        fields = [k for k in fields if k not in shared]
        parts.append(
            f'<details class="batch" id="{batch["id"]}"><summary>'
            f"{escape(batch['domain'])} / {escape(batch['task'])} / "
            f"{escape(batch['tool'])} · {len(numbers)} remaining</summary>"
        )
        if batch["reference"]:
            review = reviews[base_number]
            parts.append(
                f"<p><strong>Reviewed comparison: {case_link(base_number)} — "
                f"{escape(review['label'])}</strong></p><blockquote>"
                f"{escape(review.get('rationale', ''))}</blockquote>"
            )
        else:
            parts.append(f"<p>Unreviewed comparison: {case_link(base_number)}.</p>")
        variants = {}
        for n in shown:
            key = dumps(authority(rows[n - 1]["request"]))
            variants.setdefault(key, f"Context {len(variants) + 1}")
        for key, title in variants.items():
            parts.append(
                f"<h3>{title}: application/user instructions</h3><pre>{pretty(json.loads(key))}</pre>"
            )
        parts.append(
            f"<h3>Arguments shared by every row</h3><pre>{pretty(shared)}</pre>"
            f"<details><summary>Comparison history — read once before comparing variants</summary>"
            f"<pre>{pretty([m for m in base['messages'] if m['role'] not in {'system', 'user'}])}</pre></details>"
            '<div class="table-wrap"><table><thead><tr><th>Case</th><th>Instructions</th>'
            + "".join(f"<th>{escape(k)}</th>" for k in fields)
            + "<th>Evidence and context differences</th></tr></thead><tbody>"
        )
        # The reviewed reference is distinct from review targets and never counted twice.
        table_numbers = ([base_number] if batch["reference"] else []) + numbers
        for n in table_numbers:
            request = rows[n - 1]["request"]
            args = request["candidate"]["arguments"]
            is_reference = n == batch["reference"]
            row_class = "reference" if is_reference else "pending"
            row_id = f"{batch['id']}-reference" if is_reference else f"review-{n}"
            parts.append(f'<tr class="{row_class}" id="{row_id}">')
            parts.append(
                f"<td>{case_link(n)}<br>{'Reviewed comparison' if is_reference else 'Unreviewed'}"
                f"<br>Primary: {escape(renderer.label(rows[n - 1], 'annotation') or 'missing')}"
                f"<br>Checker: {escape(renderer.label(rows[n - 1], 'model_review') or 'missing')}"
                f"</td><td>{variants[dumps(authority(request))]}</td>"
            )
            changes = argument_changes(base["candidate"]["arguments"], args)
            for key in fields:
                value = (
                    folded(args[key])
                    if key in args
                    else '<span class="omitted">omitted</span>'
                )
                parts.append(
                    f'<td class="{"changed" if key in changes else ""}">{value}</td>'
                )
            facts = observations(request)
            parts.append("<td>")
            if any("date" in k or k == "start_time" for k in args):
                parts.append(
                    "<p>Current-day lookup: "
                    + (
                        "recorded below"
                        if facts["date_lookups"]
                        else "none in this history"
                    )
                    + ".</p>"
                )
            for key, title in (
                ("date_lookups", "Date lookup results"),
                ("prior_errors", "Prior tool errors"),
                ("same_call_results", "Earlier identical-call results"),
            ):
                if facts[key]:
                    parts.append(
                        f"<details><summary>{title} ({len(facts[key])})</summary><pre>{pretty(facts[key])}</pre></details>"
                    )
            if n != base_number:
                diff = grouping.context_diff(base, request, base_number, n)
                parts.append(
                    f"<details><summary>Context differences from Decision {base_number}</summary><pre>{escape(diff)}</pre></details>"
                    if diff
                    else "<p>Context identical to comparison.</p>"
                )
            parts.append("</td></tr>")
        parts.append("</tbody></table></div></details>")
    return "\n".join(parts + ["</body></html>\n"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "dataset": args.dataset.read_bytes(),
        "reviews": args.reviews.read_bytes(),
    }
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in inputs.items()}
    previous = args.output / "groups.json"
    if previous.exists() and json.loads(previous.read_text())["input_sha256"] != hashes:
        parser.error(
            "Inputs changed; preserve this report and use a fresh output directory"
        )
    rows = [json.loads(line) for line in inputs["dataset"].splitlines()]
    validate_saved(rows)
    reviews, supplied = read_reviews(inputs["reviews"].decode(), rows)
    batches = build_batches(rows, reviews)
    numbers = sorted(
        {
            n
            for b in batches
            for n in b["members"] + ([b["reference"]] if b["reference"] else [])
        }
    )
    summary = {
        "remaining": sum(len(b["members"]) for b in batches),
        "batches": len(batches),
        "cases_with_reviewed_comparison": sum(
            len(b["members"]) for b in batches if b["reference"]
        ),
        "review_file_entries": len(supplied),
        "effective_reviewed_cases": len(reviews),
    }
    manifest = {
        "input_sha256": hashes,
        "dataset": str(args.dataset.resolve()),
        "review_file": str(args.reviews.resolve()),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        or None,
        "script_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(grouping.__file__), Path(renderer.__file__)]
        },
        "selection": "Unreviewed valid-label disagreements; domain/task/tool batches with identical policy and definitions.",
        "comparison": "Prefer identical application/user messages, then fewest differing argument fields summed across members, then original decision number. Labels do not affect selection.",
        "summary": summary,
        "batches": batches,
        "case_decisions": numbers,
        "human_reviews": reviews,
        "supplied_review_lines": supplied,
        "dataset_edits": 0,
        "inference_calls": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "groups.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "index.html").write_text(render_remaining(rows, reviews, batches))
    (args.output / "cases.html").write_text(
        renderer.render(
            [rows[n - 1] for n in numbers],
            decision_numbers=numbers,
            human_reviews=reviews,
        )
    )
    assert (
        args.dataset.read_bytes() == inputs["dataset"]
        and args.reviews.read_bytes() == inputs["reviews"]
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
