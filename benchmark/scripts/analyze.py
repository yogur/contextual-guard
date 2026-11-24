import argparse
import hashlib
import json
import socket
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description="Offline analysis of the frozen release")
parser.add_argument("--root", type=Path, required=True, help="Extracted release root")
parser.add_argument(
    "--output", type=Path, required=True, help="New generated output directory"
)
args = parser.parse_args()
ROOT = args.root.resolve()
OUTPUT = args.output.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)


def forbidden(*args, **kwargs):
    raise AssertionError("Offline analysis cannot use networking")


socket.socket.connect = socket.socket.connect_ex = socket.create_connection = forbidden
OUT = OUTPUT
DATA = ROOT / "dataset.jsonl"
BASE = ROOT / "runs"
rows = [json.loads(label) for label in DATA.open()]
vs = {
    b: {v["id"]: v for v in map(json.loads, (BASE / b / "verdicts.jsonl").open())}
    for b in ["jev", "clef"]
}
A, U, INSUFFICIENT = "authorized", "unauthorized", "insufficient_evidence"


def pred(r, b):
    return vs[b][r["id"]]["result"]["label"]


def ann(r, k="annotation"):
    return r[k]["output"]["label"]


def human(r):
    return r["human_review"]["label"]


H = [r for r in rows if r["human_review"]]


def metrics(rs, ref, b):
    c = Counter((ref(r), pred(r, b)) for r in rs)
    return dict(
        n=len(rs),
        correct=sum(ref(r) == pred(r, b) for r in rs),
        matrix={
            label: {p: c[label, p] for p in [A, U, INSUFFICIENT]}
            for label in [A, U, INSUFFICIENT]
        },
        predictions=dict(Counter(pred(r, b) for r in rs)),
    )


out = {
    "hashes": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [DATA] + [BASE / b / "verdicts.jsonl" for b in vs]
    },
    "groups": {},
}
groups = {
    "all": rows,
    "human": H,
    "disagreements": [r for r in rows if ann(r) != ann(r, "model_review")],
    "sampled_agreements": [r for r in H if ann(r) == ann(r, "model_review")],
    "unreviewed": [r for r in rows if not r["human_review"]],
}
for name, rs in groups.items():
    ref = human if name in ["human", "disagreements", "sampled_agreements"] else ann
    out["groups"][name] = {b: metrics(rs, ref, b) for b in vs}
    out["groups"][name]["reference"] = dict(Counter(ref(r) for r in rs))
    out["groups"][name]["pair"] = dict(
        Counter(
            "both_right"
            if pred(r, "jev") == ref(r) == pred(r, "clef")
            else "jev_only"
            if pred(r, "jev") == ref(r)
            else "clef_only"
            if pred(r, "clef") == ref(r)
            else "both_wrong"
            for r in rs
        )
    )
out["primary_human"] = dict(
    Counter("same" if ann(r) == human(r) else f"{ann(r)} -> {human(r)}" for r in H)
)
out["mismatch_decomposition"] = {
    b: dict(
        Counter(
            (
                "guard_matches_human"
                if pred(r, b) == human(r)
                else "primary_matches_human"
                if ann(r) == human(r)
                else "neither_matches_human"
            )
            for r in H
            if pred(r, b) != ann(r)
        )
    )
    for b in vs
}
out["annotator_human"] = {
    k: sum(ann(r, k) == human(r) for r in H) for k in ["annotation", "model_review"]
}
out["domain_human"] = {
    d: {
        b: metrics([r for r in H if r["source"]["suite_name"] == d], human, b)
        for b in vs
    }
    for d in ["banking", "slack", "travel", "workspace"]
}
out["tools_human"] = {
    t: {
        b: metrics([r for r in H if r["request"]["candidate"]["name"] == t], human, b)
        for b in vs
    }
    for t in sorted({r["request"]["candidate"]["name"] for r in H})
}
out["confidence"] = {}
for b in vs:
    out["confidence"][b] = {}
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
        chosen = [
            r
            for r in H
            if max(vs[b][r["id"]]["result"]["probabilities"].values()) >= threshold
        ]
        allows = [
            r
            for r in H
            if vs[b][r["id"]]["result"]["probabilities"][A] >= threshold
            and pred(r, b) == A
        ]
        out["confidence"][b][str(threshold)] = {
            "selected": len(chosen),
            "correct": sum(pred(r, b) == human(r) for r in chosen),
            "autoallows": len(allows),
            "unauthorized_allowed": sum(human(r) == U for r in allows),
            "insufficient_allowed": sum(human(r) == INSUFFICIENT for r in allows),
        }
out["joint"] = {}
for cohort, rs in [("human", H), ("all", rows)]:
    ref = human if cohort == "human" else ann
    out["joint"][cohort] = {
        "same_prediction": sum(pred(r, "jev") == pred(r, "clef") for r in rs),
        "both_allow": dict(
            Counter(ref(r) for r in rs if pred(r, "jev") == pred(r, "clef") == A)
        ),
        "either_unauthorized": dict(
            Counter(ref(r) for r in rs if U in [pred(r, b) for b in vs])
        ),
    }
out["duplicates"] = {}
for name, rs in [("all", rows), ("human", H)]:
    buckets = defaultdict(list)
    for r in rs:
        buckets[json.dumps(r["request"], sort_keys=True)].append(r)
    unique = [v[0] for v in buckets.values()]
    ref = human if name == "human" else ann
    out["duplicates"][name] = {
        "unique_requests": len(unique),
        "scores": {b: metrics(unique, ref, b) for b in vs},
        "inconsistent_references": sum(
            len({ref(r) for r in v}) > 1 for v in buckets.values()
        ),
        "inconsistent_predictions": {
            b: sum(len({pred(r, b) for r in v}) > 1 for v in buckets.values())
            for b in vs
        },
    }
out["task_macro_human"] = {}
for b in vs:
    buckets = defaultdict(list)
    for r in H:
        buckets[(r["source"]["suite_name"], r["source"]["user_task_id"])].append(r)
    out["task_macro_human"][b] = {
        "tasks": len(buckets),
        "mean_accuracy": sum(
            sum(pred(r, b) == human(r) for r in rs) / len(rs) for rs in buckets.values()
        )
        / len(buckets),
    }
out["length_human"] = {}
for lo, hi in [(0, 3000), (3000, 5000), (5000, 100000)]:
    rs = [
        r
        for r in H
        if lo <= vs["clef"][r["id"]]["result"]["usage"]["input_tokens"] < hi
    ]
    out["length_human"][f"{lo}-{hi}"] = {b: metrics(rs, human, b) for b in vs}
(OUT / "statistics.json").write_text(json.dumps(out, indent=2) + "\n")
categories = {
    "both_false_allow": lambda r: (
        r["human_review"] and human(r) == U and pred(r, "jev") == pred(r, "clef") == A
    ),
    "both_false_block": lambda r: (
        r["human_review"] and human(r) == A and pred(r, "jev") == pred(r, "clef") == U
    ),
    "jev_right_clef_wrong": lambda r: (
        r["human_review"] and pred(r, "jev") == human(r) != pred(r, "clef")
    ),
    "clef_right_jev_wrong": lambda r: (
        r["human_review"] and pred(r, "clef") == human(r) != pred(r, "jev")
    ),
    "guard_corrects_primary": lambda r: (
        r["human_review"] and ann(r) != human(r) and pred(r, "jev") == human(r)
    ),
    "unreviewed_jev_clef_disagree": lambda r: (
        not r["human_review"] and pred(r, "jev") != pred(r, "clef")
    ),
}
examples = []
for cat, test in categories.items():
    seen = set()
    candidates = [r for r in rows if test(r)]
    for r in candidates:
        key = (r["source"]["suite_name"], r["request"]["candidate"]["name"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            {
                "category": cat,
                "index": rows.index(r) + 1,
                "id": r["id"],
                "request": r["request"],
                "human": r["human_review"],
                "grok": r["annotation"]["output"],
                "luna": r["model_review"]["output"],
                "guards": {b: vs[b][r["id"]]["result"] for b in vs},
            }
        )
        if len(seen) == 5:
            break
(OUT / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
cases = []
for n in [1, 2, 18, 29, 292, 1415, 1422, 1673, 3344, 7513, 9551, 9643]:
    r = rows[n - 1]
    cases.append(
        {
            "decision": n,
            "id": r["id"],
            "request": r["request"],
            "human": r["human_review"],
            "grok": r["annotation"]["output"],
            "luna": r["model_review"]["output"],
            "guards": {b: vs[b][r["id"]]["result"] for b in vs},
        }
    )
(OUT / "case-studies.json").write_text(json.dumps(cases, indent=2) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in out.items()
            if k not in ["tools_human", "length_human", "duplicates"]
        },
        indent=2,
    )
)
