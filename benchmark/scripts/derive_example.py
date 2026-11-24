"""Derive the release's 151-row prefix without changing requests or verdicts."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.root / "dataset.jsonl"
    base = args.root / "runs"
    if args.output.exists():
        raise ValueError("Example output must be new")
    args.output.mkdir(parents=True)
    lines = source.read_bytes().splitlines(keepends=True)[:151]
    dataset = args.output / "dataset.jsonl"
    dataset.write_bytes(b"".join(lines))
    ids = [json.loads(line)["id"] for line in lines]
    provenance = {
        "description": "Derived prefix; not a new inference run or representative sample",
        "source_dataset_sha256": sha(source),
        "dataset_sha256": sha(dataset),
        "retained_ids": ids,
        "selection": "First 151 rows, through the first human-reviewed authorized case",
        "runs": {},
    }
    for backend in ("jev", "clef"):
        original = base / backend
        target = args.output / backend
        target.mkdir()
        metadata = json.loads((original / "benchmark.json").read_text())
        metadata["dataset_sha256"] = sha(dataset)
        metadata["selection"] = {"ids": ids, "limit": None}
        (target / "benchmark.json").write_text(
            json.dumps(metadata, sort_keys=True) + "\n"
        )
        verdicts = (original / "verdicts.jsonl").read_bytes().splitlines(keepends=True)
        prefix = verdicts[:151]
        assert [json.loads(line)["id"] for line in prefix] == ids
        (target / "verdicts.jsonl").write_bytes(b"".join(prefix))
        provenance["runs"][backend] = {
            "source_benchmark_sha256": sha(original / "benchmark.json"),
            "source_verdicts_sha256": sha(original / "verdicts.jsonl"),
            "derived_benchmark_sha256": sha(target / "benchmark.json"),
            "derived_verdicts_sha256": sha(target / "verdicts.jsonl"),
            "changes": "Dataset hash and selection only; verdict prefix bytes unchanged",
        }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
