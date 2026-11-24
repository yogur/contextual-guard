"""Reconstruct dataset-v1 human labels from the supplied review layers, offline."""

import argparse
import hashlib
import importlib.util
import json
import socket
from collections import Counter, defaultdict
from pathlib import Path

from jev_dataset.pipeline import dumps, save_rows, validate_saved

spec = importlib.util.spec_from_file_location(
    "remaining_review", Path(__file__).with_name("render_remaining_review.py")
)
remaining = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remaining)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    raise AssertionError("Review reproduction must not use networking or inference")


def non_human_hash(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            dumps({k: v for k, v in row.items() if k != "human_review"}).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def merge_reviews(rows, layers):
    """Layers are ordered (source name, text); only explicit overrides replace reviews."""
    reviews = {
        n: dict(row["human_review"])
        for n, row in enumerate(rows, 1)
        if row.get("human_review")
    }
    events, summaries = [], []
    for name, text in layers:
        lines = text.splitlines()
        updated, supplied = remaining.read_reviews(
            text,
            rows,
            previous=reviews,
            reviewer="user (supplied files; discussion-assisted)",
        )
        counts = Counter()
        for n, entry in supplied.items():
            raw = lines[entry["line"] - 1]
            if "??" in raw or "TODO" in raw:
                raise ValueError(f"Unresolved review: {name}:{entry['line']}")
            verdict = entry["verdict"]
            modified = verdict.startswith("modified:")
            if modified != (n in reviews) and verdict != "unchanged":
                raise ValueError(
                    f"Existing reviews need modified:, new reviews must not: {name}:{entry['line']}"
                )
            action = (
                "unchanged"
                if verdict == "unchanged"
                else "modified"
                if modified
                else "new"
            )
            counts[action] += 1
            events.append(
                {
                    "decision": n,
                    "id": rows[n - 1]["id"],
                    "source_file": name,
                    "source_line": entry["line"],
                    "source_text": raw,
                    "action": action,
                    "before": reviews.get(n),
                    "after": updated[n],
                }
            )
        reviews = updated
        summaries.append({"file": name, "entries": len(supplied), **dict(counts)})
    merged = [row | {"human_review": reviews.get(n)} for n, row in enumerate(rows, 1)]
    return merged, events, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error("Choose a new output directory outside the evidence")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = (
        forbidden
    )
    manifest = json.loads((root / "manifest.json").read_text())
    originals = {}
    for item in manifest["files"]:
        assert sha(root / item["path"]) == item["sha256"], item["path"]
        originals[item["path"]] = item.get("original_path", item["path"])
    source = root / "review/initial/dataset.jsonl"
    initial = json.loads((root / "review/initial/human-reviews.json").read_text())
    assert initial["reviewed_dataset_sha256"] == sha(source)
    rows = [json.loads(line) for line in source.read_bytes().splitlines()]
    validate_saved(rows)
    for entry in initial["reviews"]:
        row = rows[entry["decision"] - 1]
        assert row["id"] == entry["id"]
        assert row["human_review"] == entry["human_review"]
    paths = ["review/grouped/reviews.txt", "review/remaining/reviews.txt"]
    # Retain recorded source names inside provenance, independent of public paths.
    merged, events, layers = merge_reviews(
        rows, [(originals[name], (root / name).read_text()) for name in paths]
    )
    validate_saved(merged)
    assert non_human_hash(merged) == non_human_hash(rows)
    output.mkdir(parents=True)
    save_rows(output, merged)
    assert sha(output / "dataset.jsonl") == (
        "c92191ff4a2378563e4ac991b07fba66f9cb55f539ac2c5e081deb1aadb7d87f"
    )
    indices = defaultdict(list)
    for index, event in enumerate(events):
        indices[event["decision"]].append(index)
    records = [
        {
            "decision": n,
            "id": row["id"],
            "request_sha256": hashlib.sha256(
                dumps(row["request"]).encode()
            ).hexdigest(),
            "human_review": row["human_review"],
            "event_indices": indices[n],
        }
        for n, row in enumerate(merged, 1)
        if row.get("human_review")
    ]
    ledger = {
        "reviewed_dataset_sha256": sha(output / "dataset.jsonl"),
        "source_manifest_sha256": sha(root / "manifest.json"),
        "script_sha256": sha(Path(__file__)),
        "reviews": records,
        "events": events,
        "layers": layers,
        "inherited_reviews": initial["reviews"],
    }
    (output / "human-reviews.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(
        json.dumps(
            {
                "human_reviewed": len(records),
                "dataset_sha256": ledger["reviewed_dataset_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
