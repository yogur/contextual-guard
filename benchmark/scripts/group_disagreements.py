"""Group unreviewed disagreements offline without changing dataset rows or labels."""

import argparse
import difflib
import hashlib
import importlib.util
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from jev_dataset.pipeline import dumps, validate_saved

spec = importlib.util.spec_from_file_location(
    "dataset_review_renderer", Path(__file__).with_name("render_dataset_review.py")
)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
escape, pretty = renderer.escape, renderer.pretty


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def task_call(row):
    return (
        row["source"]["suite_name"],
        row["source"]["user_task_id"],
        dumps(row["request"]["candidate"]),
    )


def member(number, row):
    return {
        "decision": number,
        "id": row["id"],
        "source_file": row["source"].get("file"),
        "primary_label": renderer.label(row, "annotation"),
        "quality_label": renderer.label(row, "model_review"),
        "human_label": (row.get("human_review") or {}).get("label"),
    }


def group_cases(rows):
    exact, broader = defaultdict(list), defaultdict(list)
    reviewed_exact, reviewed_broader = defaultdict(list), defaultdict(list)
    for number, row in enumerate(rows, 1):
        if row["request"] is None:
            continue
        full = dumps(row["request"])
        key = task_call(row)
        entry = member(number, row)
        if entry["human_label"]:
            reviewed_exact[full].append(entry)
            reviewed_broader[key].append(entry)
        elif (
            entry["primary_label"]
            and entry["quality_label"]
            and entry["primary_label"] != entry["quality_label"]
        ):
            exact[full].append(entry)
            broader[key].append(entry)

    def groups(pending, reviewed, prefix):
        ordered = sorted(
            pending,
            key=lambda key: (
                not bool(reviewed[key]),
                -len(pending[key]),
                pending[key][0]["decision"],
            ),
        )
        result = []
        for index, key in enumerate(ordered, 1):
            references = reviewed[key]
            candidates = references or pending[key]
            result.append(
                {
                    "id": f"{prefix}-{index}",
                    "members": pending[key],
                    "reviewed_matches": references,
                    "representative": candidates[0]["decision"],
                    "request_sha256": digest(key) if prefix == "exact" else None,
                    "conflicting_exact_human_labels": prefix == "exact"
                    and len({r["human_label"] for r in references}) > 1,
                }
            )
        return result

    levels = {
        "exact_request": groups(exact, reviewed_exact, "exact"),
        "same_task_and_call": groups(broader, reviewed_broader, "task-call"),
    }
    for group in levels["same_task_and_call"]:
        variants = defaultdict(list)
        for entry in group["members"] + group["reviewed_matches"]:
            variants[dumps(rows[entry["decision"] - 1]["request"])].append(entry)
        group["context_variants"] = []
        for full, entries in variants.items():
            representative = next(
                (e["decision"] for e in entries if e["human_label"]),
                entries[0]["decision"],
            )
            group["context_variants"].append(
                {
                    "request_sha256": digest(full),
                    "representative": representative,
                    "decisions": [e["decision"] for e in entries],
                }
            )
    return levels


def context_diff(before, after, before_number, after_number):
    def context(request):
        return json.dumps(
            {k: v for k, v in request.items() if k != "candidate"},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).splitlines()

    return "\n".join(
        difflib.unified_diff(
            context(before),
            context(after),
            fromfile=f"Decision {before_number}",
            tofile=f"Decision {after_number}",
            lineterm="",
        )
    )


def case_link(number):
    return f'<a href="cases.html#decision-{number}">Decision {number}</a>'


def members_table(group):
    parts = [
        "<table><tr><th>Case</th><th>Trajectory</th><th>Grok</th>"
        "<th>Luna</th><th>Existing human verdict</th></tr>"
    ]
    for entry in group["reviewed_matches"] + group["members"]:
        parts.append(
            f"<tr><td>{case_link(entry['decision'])}</td>"
            + "".join(
                f"<td>{escape(entry[key] or 'Unreviewed')}</td>"
                for key in (
                    "source_file",
                    "primary_label",
                    "quality_label",
                    "human_label",
                )
            )
            + "</tr>"
        )
    return "\n".join(parts + ["</table>"])


def render_groups(rows, levels, summary):
    parts = [
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grouped disagreement review</title><style>
:root{color-scheme:light dark}body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:light-dark(#f3f4f6,#242628);padding:1rem}
details{border:1px solid light-dark(#ccc,#555);padding:.7rem;margin:1rem 0}summary{cursor:pointer;font-weight:600}
table{width:100%;border-collapse:collapse}td,th{text-align:left;vertical-align:top;padding:.5rem;border-bottom:1px solid #888}
td{overflow-wrap:anywhere}a{color:light-dark(#1552a4,#98c4ff)}:target{outline:2px solid #788cb8}
.added{background:light-dark(#e0f2e0,#173b24)}.removed{background:light-dark(#ffe0e0,#462525)}
</style></head><body><h1>Grouped disagreement review</h1>""",
        f"<p>{summary['unreviewed_disagreements']} unreviewed disagreements · "
        f"{summary['exact_request_groups']} exact-request groups · "
        f"{summary['same_task_and_call_groups']} same-task/same-call groups.</p>",
        "<p>Existing human verdicts are references, not newly applied labels. "
        "Groups organize review; every original case and full history remain available.</p>",
        '<p><a href="#level-1">Level 1: identical requests</a> · '
        '<a href="#level-2">Level 2: same task and proposed call</a></p>',
    ]
    for key, title, anchor in (
        ("exact_request", "Level 1: identical requests", "level-1"),
        ("same_task_and_call", "Level 2: same task and proposed call", "level-2"),
    ):
        parts.append(f'<h2 id="{anchor}">{title}</h2>')
        parts.append(
            "<p>All request fields match, including full history, policy and definitions. "
            "Source identifiers and saved model judgments may differ.</p>"
            if key == "exact_request"
            else "<p>Domain, task, tool and arguments match. Contexts may differ and "
            "can require different verdicts. Compare each variant with the representative.</p>"
        )
        parts.append(
            "<p>Groups with existing human references come first, then larger groups.</p>"
        )
        for group in levels[key]:
            number = group["representative"]
            row = rows[number - 1]
            source, request = row["source"], row["request"]
            parts.append(
                f'<details id="{group["id"]}"><summary>{escape(group["id"])} · '
                f"{escape(source['suite_name'])} / {escape(source['user_task_id'])} · "
                f"{escape(request['candidate']['name'])} · {len(group['members'])} unreviewed · "
                f"{len(group['reviewed_matches'])} existing human references</summary>"
            )
            if group["conflicting_exact_human_labels"]:
                parts.append(
                    "<p><strong>Conflicting human verdicts for identical requests; resolve before reusing.</strong></p>"
                )
            authority = [
                {"role": m["role"], "origin": m["origin"], "content": m["content"]}
                for m in request["messages"]
                if m["role"] in {"system", "user"}
            ]
            parts.append(
                f"<p>Representative: {case_link(number)}. Open it for full context, "
                "policy, definitions and model rationales.</p>"
                f"<h3>Representative system/user messages</h3><pre>{pretty(authority)}</pre>"
                f"<h3>Proposed call</h3><pre>{pretty(request['candidate'])}</pre>"
            )
            parts.append(members_table(group))
            for variant in group.get("context_variants", []):
                other = variant["representative"]
                parts.append(
                    f"<p>Context variant: {case_link(other)} · "
                    + ", ".join(
                        case_link(n) for n in variant["decisions"] if n != other
                    )
                    + "</p>"
                )
                diff = context_diff(request, rows[other - 1]["request"], number, other)
                if not diff:
                    parts.append("<p>Same context as the representative.</p>")
                else:
                    lines = [
                        f'<span class="{"added" if line.startswith("+") else "removed" if line.startswith("-") else ""}">{escape(line)}</span>'
                        for line in diff.splitlines()
                    ]
                    parts.append(
                        f"<details><summary>Context differences: Decision {number} → {other}</summary>"
                        "<pre>" + "\n".join(lines) + "</pre></details>"
                    )
            parts.append("</details>")
    return "\n".join(parts + ["</body></html>\n"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.dataset.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    rows = [json.loads(line) for line in source_bytes.splitlines()]
    del source_bytes
    validate_saved(rows)
    levels = group_cases(rows)
    summary = {
        "unreviewed_disagreements": sum(
            len(g["members"]) for g in levels["exact_request"]
        ),
        "exact_request_groups": len(levels["exact_request"]),
        "repeated_exact_request_groups": sum(
            len(g["members"]) > 1 for g in levels["exact_request"]
        ),
        "exact_cases_with_human_reference": sum(
            len(g["members"]) for g in levels["exact_request"] if g["reviewed_matches"]
        ),
        "same_task_and_call_groups": len(levels["same_task_and_call"]),
        "same_task_call_cases_with_human_reference": sum(
            len(g["members"])
            for g in levels["same_task_and_call"]
            if g["reviewed_matches"]
        ),
        "conflicting_exact_human_groups": sum(
            g["conflicting_exact_human_labels"] for g in levels["exact_request"]
        ),
    }
    numbers = sorted(
        {
            e["decision"]
            for groups in levels.values()
            for g in groups
            for e in g["members"] + g["reviewed_matches"]
        }
    )
    manifest = {
        "source_dataset": str(args.dataset.resolve()),
        "source_dataset_sha256": source_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(
            Path(__file__).with_name("render_dataset_review.py").read_bytes()
        ).hexdigest(),
        "code_revision": subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        or None,
        "methods": {
            "exact_request": "Canonical JSON equality of every request field; array order and string content preserved.",
            "same_task_and_call": "Equal domain/task and canonical candidate tool/arguments; compare all other request fields.",
        },
        "summary": summary,
        "levels": levels,
        "case_decisions": numbers,
        "labels_changed": 0,
        "inference_calls": 0,
    }
    previous = args.output / "groups.json"
    if previous.exists():
        if json.loads(previous.read_text())["source_dataset_sha256"] != source_hash:
            parser.error(
                "Source dataset changed; preserve this report and use a new output directory"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "groups.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "index.html").write_text(render_groups(rows, levels, summary))
    (args.output / "cases.html").write_text(
        renderer.render([rows[n - 1] for n in numbers], decision_numbers=numbers)
    )
    assert hashlib.sha256(args.dataset.read_bytes()).hexdigest() == source_hash
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
