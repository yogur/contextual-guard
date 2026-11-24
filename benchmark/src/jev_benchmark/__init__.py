"""Sequential saved-request replay and persistence. Never imports annotation."""

import fcntl
import hashlib
import json
import os
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from jev_contracts import GuardModel, GuardRequest, GuardResult
from jev_guard import inspect_ollama, make_guard, model_metadata, system_one_input_bytes


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Benchmark is already in use") from None
        yield


def load_dataset(path: Path, limit=None):
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate saved decision IDs")
    selected = rows[:limit]
    for row in selected:
        if row["request"] is not None:
            GuardRequest.model_validate(row["request"])
    return rows, selected, hashlib.sha256(data).hexdigest()


def read_entries(directory, *, recover=False):
    path = directory / "verdicts.jsonl"
    if not path.exists():
        return [], False
    data = path.read_bytes()
    incomplete_tail = bool(data and not data.endswith(b"\n"))
    if incomplete_tail:
        end = data.rfind(b"\n") + 1
        if recover:
            with path.open("r+b") as stream:
                stream.truncate(end)
        data = data[:end]
    return [json.loads(line) for line in data.splitlines()], incomplete_tail


def validate_entries(entries, rows, metadata):
    valid = {row["id"]: row for row in rows if row["request"] is not None}
    seen, first = set(), set()
    for order, entry in enumerate(entries, 1):
        key = (entry["backend"], entry["id"])
        if (
            key in seen
            or entry["id"] not in valid
            or entry["backend"] not in metadata["backend_order"]
        ):
            raise ValueError("Verdict log contains duplicate or unknown outcomes")
        backend = entry["backend"]
        if (
            entry["request_sha256"] != digest(valid[entry["id"]]["request"])
            or type(entry["execution_order"]) is not int
            or type(entry["first_call"]) is not bool
            or entry["execution_order"] != order
            or entry["first_call"] != (backend not in first)
        ):
            raise ValueError("Verdict log identity/order mismatch")
        result = GuardResult.model_validate(entry["result"])
        if "completed_at" in entry and not isinstance(entry["completed_at"], str):
            raise ValueError("Invalid assessment timestamp")
        recorded = result.model.model_dump(mode="json")
        recorded["returned"] = None
        if recorded != metadata["models"][backend]:
            raise ValueError("Verdict model/settings mismatch")
        if (
            result.model.provider == "bedrock_mantle"
            and result.probabilities is not None
        ):
            raise ValueError("LLM cannot invent native probabilities")
        if (
            result.model.provider in {"typesafe", "typesafe_ollama"}
            and result.label
            and result.probabilities is None
        ):
            raise ValueError("Jev verdict requires native probabilities")
        seen.add(key)
        first.add(backend)
    return seen


def load_run(dataset, directory):
    """Validate saved identities without network, configuration or source writes."""
    try:
        metadata = json.loads((directory / "benchmark.json").read_text())
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("models"), dict
        ):
            raise ValueError("Malformed benchmark metadata")
        if metadata.get("execution") is not None and not isinstance(
            metadata["execution"], dict
        ):
            raise ValueError("Malformed execution metadata")
        order = metadata["backend_order"]
        if (
            metadata["schema_version"] != "benchmark.v1"
            or not isinstance(order, list)
            or not order
            or not all(isinstance(name, str) and name for name in order)
            or len(set(order)) != len(order)
            or set(order) != set(metadata["models"])
        ):
            raise ValueError("Malformed benchmark metadata")
        for model in metadata["models"].values():
            GuardModel.model_validate(model)
            if model["returned"] is not None:
                raise ValueError("Metadata must describe requested model")
        for key in ("dataset_sha256", "instructions_sha256", "criteria_sha256"):
            value = metadata[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Invalid metadata hash")
        limit = metadata["selection"]["limit"]
        if limit is not None and type(limit) is not int:
            raise ValueError("Invalid selection limit")
        all_rows, rows, dataset_hash = load_dataset(dataset, limit)
        if (
            dataset_hash != metadata["dataset_sha256"]
            or [r["id"] for r in rows] != metadata["selection"]["ids"]
        ):
            raise ValueError(
                "Dataset identity changed; cannot join saved verdicts and labels"
            )
        entries, incomplete_tail = read_entries(directory)
        validate_entries(entries, rows, metadata)
        return metadata, all_rows, rows, entries, incomplete_tail
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed saved benchmark") from exc


def execution_environment():
    cpu = (
        next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor()
    )
    gpus = []
    for device in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        if not device.parent.name[4:].isdigit():
            continue
        gpus.append(
            {
                key: (device / key).read_text().strip()
                if (device / key).exists()
                else None
                for key in ("vendor", "device", "mem_info_vram_total")
            }
        )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "cpu": cpu,
        "ram_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
        "gpu": gpus or None,
        "load_average": list(os.getloadavg()),
        "warmup_calls": 0,
        "first_call_note": "First recorded assessment, not proof of a cold model",
    }


async def run_benchmark(
    dataset: Path,
    directory: Path,
    profiles: dict,
    question: dict,
    *,
    limit=None,
    factory=make_guard,
    emit=lambda text: print(text, flush=True),
):
    if directory.resolve() == dataset.parent.resolve():
        raise ValueError("Benchmark output must be separate from dataset preparation")
    _, rows, dataset_hash = load_dataset(dataset, limit)
    signature = {
        "schema_version": "benchmark.v1",
        "dataset_sha256": dataset_hash,
        "selection": {"limit": limit, "ids": [row["id"] for row in rows]},
        "backend_order": list(profiles),
        "models": {
            name: model_metadata(settings).model_dump(mode="json")
            for name, settings in profiles.items()
        },
        "instructions_sha256": digest(question["instructions"]),
        "criteria_sha256": digest(question["criteria"]),
    }
    with locked(directory):
        path = directory / "benchmark.json"
        if path.exists():
            recorded = json.loads(path.read_text())
            if {k: v for k, v in recorded.items() if k != "execution"} != signature:
                raise ValueError(
                    "Dataset/selection/settings changed; use a new output directory"
                )
        elif (directory / "verdicts.jsonl").exists():
            raise ValueError("Verdicts exist without benchmark metadata")
        else:
            execution = execution_environment()
            execution["local_runtime"] = {
                name: inspect_ollama(settings)
                for name, settings in profiles.items()
                if settings.provider == "typesafe_ollama"
            }
            write_json(path, signature | {"execution": execution})
        entries, _ = read_entries(directory, recover=True)
        done = validate_entries(entries, rows, signature)
        valid = [row for row in rows if row["request"] is not None]
        planned = {
            name: sum((name, row["id"]) not in done for row in valid)
            for name in profiles
        }
        completed = {name: len(valid) - planned[name] for name in profiles}
        errors = dict.fromkeys(profiles, 0)
        for entry in entries:
            errors[entry["backend"]] += entry["result"]["error"] is not None
        for name, settings in profiles.items():
            if settings.provider == "typesafe_ollama" and planned[name]:
                inspect_ollama(settings)
                oversized = sum(
                    system_one_input_bytes(
                        GuardRequest.model_validate(row["request"]), question
                    )
                    > 65536
                    for row in valid
                )
                if oversized:
                    raise ValueError(
                        f"{name}: {oversized}/{len(valid)} inputs exceed local 64 KiB state/schema limit; bulk dispatch blocked"
                    )
        emit(
            dumps(
                {
                    "selected_rows": len(rows),
                    "extraction_skips": len(rows) - len(valid),
                    "planned_calls": planned,
                    "completed_assessments": completed,
                    "errors": errors,
                    "automatic_retries": 0,
                }
            )
        )
        guards = {
            name: factory(settings, question) for name, settings in profiles.items()
        }
        stopped = set()
        called = {entry["backend"] for entry in entries}
        try:
            for row in valid:
                for name, guard in guards.items():
                    if name in stopped or (name, row["id"]) in done:
                        continue
                    result = await guard.assess(
                        GuardRequest.model_validate(row["request"])
                    )
                    entry = {
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "backend": name,
                        "id": row["id"],
                        "request_sha256": digest(row["request"]),
                        "execution_order": len(entries) + 1,
                        "first_call": name not in called,
                        "result": result.model_dump(
                            mode="json", exclude={"estimated_cost_usd", "cost_source"}
                        ),
                    }
                    # Flush every terminal outcome, including errors, before continuing.
                    with (directory / "verdicts.jsonl").open("a") as stream:
                        stream.write(dumps(entry) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    entries.append(entry)
                    called.add(name)
                    done.add((name, row["id"]))
                    completed[name] += 1
                    errors[name] += result.error is not None
                    emit(
                        dumps(
                            {
                                "progress": name,
                                "completed_assessments": completed[name],
                                "total_assessments": len(valid),
                                "errors": errors[name],
                                "pending": len(valid) - completed[name],
                            }
                        )
                    )
                    if result.error and result.error.kind in {
                        "authentication",
                        "configuration",
                        "rate_limit",
                    }:
                        stopped.add(name)
                        emit(
                            dumps(
                                {
                                    "dispatch_stopped": name,
                                    "error": result.error.model_dump(mode="json"),
                                }
                            )
                        )
        finally:
            for guard in guards.values():
                await guard.aclose()
        summary = {
            "completed_assessments": len(entries),
            "errors": sum(entry["result"]["error"] is not None for entry in entries),
            "pending": len(valid) * len(profiles) - len(entries),
            "stopped_backends": sorted(stopped),
        }
        emit(dumps(summary))
        return summary
