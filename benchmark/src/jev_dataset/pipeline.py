"""JSONL persistence and bounded annotation, independent of transport."""

import asyncio
import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from jev_contracts import AnnotationOutput, GuardRequest, validate_evidence

from .adapter import read_json, sha256

IMMUTABLE = ("id", "source", "request", "extraction_error")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def atomic_write(path: Path, text: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_rows(directory: Path, rows):
    atomic_write(directory / "dataset.jsonl", "".join(dumps(r) + "\n" for r in rows))


def load_rows(directory: Path):
    return [
        json.loads(line)
        for line in (directory / "dataset.jsonl").read_text().splitlines()
    ]


@contextmanager
def locked(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Dataset is already in use") from None
        yield


def evidence_hash(rows):
    return sha256(dumps([{k: r[k] for k in IMMUTABLE} for r in rows]).encode())


def preparation_identity(metadata):
    if "selection" not in metadata:
        return metadata
    # Locations and unselected contents are provenance, not selection identity.
    return {
        k: v
        for k, v in metadata.items()
        if k
        not in {
            "run_directory",
            "definitions_file",
            "selection_file",
            "inputs",
            "summary",
        }
    } | {
        "inputs": [
            item
            for item in metadata["inputs"]
            if item["status"] in {"selected", "error"}
        ]
    }


def initialize(directory: Path, rows, metadata):
    """Identical preparation resumes; changed evidence never reuses labels."""
    with locked(directory):
        path = directory / "run.json"
        if path.exists():
            existing = read_json(path)
            if preparation_identity(existing["preparation"]) != json.loads(
                dumps(preparation_identity(metadata))
            ):
                raise ValueError(
                    "Preparation inputs/settings changed; use a new output directory"
                )
            if not (directory / "dataset.jsonl").exists():
                if (directory / "attempts.jsonl").exists():
                    raise ValueError("Dataset missing but attempts exist")
                save_rows(directory, rows)
            saved = load_rows(directory)
            if [{k: r[k] for k in IMMUTABLE} for r in saved] != [
                {k: r[k] for k in IMMUTABLE} for r in rows
            ]:
                raise ValueError(
                    "Saved requests/source changed; use a new output directory"
                )
            if "selection" in metadata:
                existing["preparation"] = metadata
                atomic_write(path, dumps(existing) + "\n")
            return saved
        if (directory / "dataset.jsonl").exists() or (
            directory / "attempts.jsonl"
        ).exists():
            raise ValueError(
                "Output exists without run metadata; use a new output directory"
            )
        # Metadata first: a crash here leaves an explicitly incomplete preparation.
        atomic_write(
            path,
            dumps(
                {
                    "preparation": metadata,
                    "evidence_sha256": evidence_hash(rows),
                    "stages": {},
                    "attempt_totals": {},
                }
            )
            + "\n",
        )
        save_rows(directory, rows)
        return rows


def read_log(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    # A killed append can leave an incomplete tail. Recover only that tail;
    # malformed complete lines remain errors rather than disappearing.
    if data and not data.endswith(b"\n"):
        end = data.rfind(b"\n") + 1
        with path.open("r+b") as stream:
            stream.truncate(end)
        data = data[:end]
    return [json.loads(line) for line in data.splitlines()]


def append_log(path, entry):
    with path.open("a") as stream:
        stream.write(dumps(entry) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def attempts(directory):
    return read_log(directory / "attempts.jsonl")


def append_attempt(directory, entry):
    append_log(directory / "attempts.jsonl", entry)


def validate_saved(rows):
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate saved decision IDs")
    for row in rows:
        if row["request"] is None:
            continue
        request = GuardRequest.model_validate(row["request"])
        for field in ("annotation", "model_review"):
            result = row[field]
            if result is not None and (
                not isinstance(result, dict)
                or result.get("state") not in {"completed", "retry", "paused"}
            ):
                raise ValueError("Malformed saved annotation state")
            if (
                result is not None
                and result.get("output") is None
                and result.get("error") is None
            ):
                raise ValueError(
                    "Saved annotation has neither judgment nor operational error"
                )
            if result and result.get("output") is not None:
                validate_evidence(
                    AnnotationOutput.model_validate(result["output"]), request
                )
                if result.get("error") is not None:
                    raise ValueError("Saved result contains both output and error")


def finished(result):
    return result is not None and result.get("state") == "completed"


def disposition(result):
    if result.get("output") is not None:
        return "completed", False
    error = result.get("error") or {}
    kind, status = error.get("kind"), error.get("status_code")
    if kind in {
        "invalid_response",
        "incomplete_or_refused_response",
        "ContextWindowExceededError",
    }:
        return "completed", False
    if status == 429 or kind == "RateLimitError":
        return "paused", True
    if kind in {
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "UnsupportedParamsError",
        "NoCredentialsError",
    } or status in {400, 401, 403, 404, 422}:
        return "completed", True
    if status in {408, 500, 502, 503, 504} or kind in {
        "Timeout",
        "APITimeoutError",
        "APIConnectionError",
        "ServiceUnavailableError",
        "InternalServerError",
    }:
        return "retry", False
    return "completed", False


def save_annotations(directory, expected, rows, stages):
    # Merge once at a stage boundary/exit, retaining saved human reviews.
    saved = load_rows(directory)
    if [{k: r[k] for k in IMMUTABLE} for r in saved] != expected:
        raise ValueError(
            "Evidence changed during annotation; attempt retained, dataset untouched"
        )
    for current, row in zip(rows, saved):
        for stage in stages:
            row[stage] = current[stage]
    save_rows(directory, saved)


def totals(entries, dispatches=None):
    result = {}
    for stage in ("annotation", "model_review"):
        selected = [e["result"] for e in entries if e["stage"] == stage]
        result[stage] = {
            "attempts": len(selected),
            "errors": sum(r.get("error") is not None for r in selected),
            "estimated_cost_usd": sum(
                r["estimated_cost_usd"]
                for r in selected
                if r.get("estimated_cost_usd") is not None
            ),
            "unpriced_attempts": sum(
                r.get("estimated_cost_usd") is None for r in selected
            ),
        }
        if dispatches is not None:
            sent = [d for d in dispatches if d["stage"] == stage]
            returned = {(e["id"], e["stage"], e["attempt"]) for e in entries}
            unknown = sum(
                (d["id"], d["stage"], d["attempt"]) not in returned for d in sent
            )
            result[stage].update(
                dispatches=len(sent),
                unknown_outcomes=unknown,
                unpriced_dispatches=result[stage]["unpriced_attempts"] + unknown,
            )
    return result


async def run_annotations(
    directory: Path,
    rubric: str,
    profile: dict,
    *,
    review_profile=None,
    review_limit=None,
    max_attempts=3,
    retries=None,
    concurrency=1,
    transport=None,
    emit=lambda text: print(text, flush=True),
):
    """Transport has annotate(request, rubric, **profile)'s one-call contract."""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if retries is not None:
        if type(retries) is not int or retries < 0:
            raise ValueError("retries must be a nonnegative integer")
        max_attempts = retries + 1
    elif not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between one and three")
    if review_limit is not None and (review_limit < 0 or review_profile is None):
        raise ValueError("review_limit requires review and must be nonnegative")
    if transport is None:
        from jev_llm import annotate

        transport = annotate
    with locked(directory):
        metadata = read_json(directory / "run.json")
        rows = load_rows(directory)
        validate_saved(rows)
        if evidence_hash(rows) != metadata["evidence_sha256"]:
            raise ValueError(
                "Saved requests/source changed; use a new output directory"
            )
        expected = [{k: r[k] for k in IMMUTABLE} for r in rows]
        stages = {"annotation": (profile, rows)}
        if review_profile is not None:
            stages["model_review"] = (review_profile, rows[:review_limit])
        for stage, (settings, selected) in stages.items():
            if concurrency > 1 and settings.get("model", "").startswith("ollama_chat/"):
                raise ValueError("Ollama annotation requires concurrency=1")
            signature = {
                "profile": settings,
                "rubric_sha256": sha256(rubric.encode()),
                "max_attempts": max_attempts,
                "ids": [r["id"] for r in selected],
            }
            if retries is not None:
                signature.update(
                    retries=retries, dispatch_accounting="before_transport.v1"
                )
            previous = metadata["stages"].get(stage)
            if previous is not None and previous != signature:
                raise ValueError(
                    f"{stage} settings changed; use a new output directory"
                )
            metadata["stages"][stage] = signature
        entries = attempts(directory)
        dispatches = (
            read_log(directory / "dispatches.jsonl") if retries is not None else None
        )
        dispatch_counts = {}
        if dispatches is not None:
            valid = set()
            stage_ids = {s: set(v["ids"]) for s, v in metadata["stages"].items()}
            for d in dispatches:
                key = (d["id"], d["stage"])
                dispatch_counts[key] = dispatch_counts.get(key, 0) + 1
                if (
                    d["id"] not in stage_ids.get(d["stage"], set())
                    or d["attempt"] != dispatch_counts[key]
                    or d["attempt"] > max_attempts
                ):
                    raise ValueError("Dispatch log does not match decisions/settings")
                valid.add((*key, d["attempt"]))
            returned = {(e["id"], e["stage"], e["attempt"]) for e in entries}
            if len(returned) != len(entries) or not returned <= valid:
                raise ValueError("Attempt log lacks unique matching dispatches")
        latest = {(e["id"], e["stage"]): e for e in entries}
        attempt_counts = {}
        for entry in entries:
            key = (entry["id"], entry["stage"])
            attempt_counts[key] = attempt_counts.get(key, 0) + 1
        by_id = {r["id"]: r for r in rows}
        dirty = False
        # Rebuild unsaved stage progress from the durable attempt log in memory.
        for (decision_id, stage), entry in latest.items():
            saved = by_id.get(decision_id)
            if saved is None or stage not in metadata["stages"]:
                raise ValueError(
                    "Attempt log does not match prepared decisions/settings"
                )
            if not finished(saved[stage]):
                saved[stage] = entry["result"]
                dirty = True
        if dispatches is not None:
            # An interrupted dispatch remains spent even without a response.
            last_dispatch = {(d["id"], d["stage"]): d for d in dispatches}
            for (decision_id, stage), d in last_dispatch.items():
                if (decision_id, stage, d["attempt"]) not in returned:
                    result = {
                        "state": "completed",
                        "output": None,
                        "error": {
                            "kind": "unknown_dispatch_outcome",
                            "attempt": d["attempt"],
                        },
                        "usage": None,
                        "estimated_cost_usd": None,
                    }
                    if by_id[decision_id][stage] != result:
                        by_id[decision_id][stage] = result
                        dirty = True
        validate_saved(rows)
        planned = {}
        for stage, (_, selected) in stages.items():
            selected_ids = {r["id"] for r in selected}
            planned[stage] = sum(
                r["id"] in selected_ids
                and r["request"] is not None
                and not finished(r[stage])
                for r in rows
            )
        if any(planned.values()):
            metadata.setdefault("executions", []).append(
                {
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "concurrency": concurrency,
                }
            )
        atomic_write(directory / "run.json", dumps(metadata) + "\n")
        emit(
            dumps(
                {
                    "planned_annotations": planned["annotation"],
                    "planned_reviews": planned.get("model_review", 0),
                    "maximum_attempts_per_decision_stage": max_attempts,
                    "maximum_retry_allowance": sum(planned.values())
                    * (max_attempts - 1),
                    "concurrency": concurrency,
                }
            )
        )
        stopped = False
        stop_error = None
        try:
            for stage, (settings, selected) in stages.items():
                selected_ids = {r["id"] for r in selected}
                if retries is not None:
                    context_error = next(
                        (
                            r[stage]["error"]
                            for r in rows
                            if r["id"] in selected_ids
                            and ((r[stage] or {}).get("error") or {}).get("kind")
                            == "ContextWindowExceededError"
                        ),
                        None,
                    )
                    if context_error:
                        stopped = True
                        return {
                            "stopped": True,
                            "error": context_error,
                            "attempt_totals": totals(entries, dispatches),
                        }
                pending = iter(
                    r
                    for r in rows
                    if r["id"] in selected_ids
                    and r["request"] is not None
                    and not finished(r[stage])
                )

                async def worker():
                    nonlocal dirty, stopped, stop_error
                    while not stopped:
                        row = next(pending, None)
                        if row is None:
                            return
                        key = (row["id"], stage)
                        prior = latest.get(key)
                        retry_index = (
                            prior["retry_index"] + 1
                            if prior and prior["result"]["state"] == "retry"
                            else 1
                        )
                        if dispatches is not None:
                            retry_index = dispatch_counts.get(key, 0) + 1
                        for index in range(retry_index, max_attempts + 1):
                            if stopped:
                                return
                            if dispatches is not None:
                                dispatch = {
                                    "id": row["id"],
                                    "stage": stage,
                                    "attempt": index,
                                    "dispatched_at_utc": datetime.now(
                                        timezone.utc
                                    ).isoformat(),
                                    "concurrency": concurrency,
                                }
                                # All writes run synchronously on this event-loop
                                # thread; workers never write from transport threads.
                                append_log(directory / "dispatches.jsonl", dispatch)
                                dispatches.append(dispatch)
                                dispatch_counts[key] = index
                            result = await transport(
                                GuardRequest.model_validate(row["request"]),
                                rubric,
                                **settings,
                            )
                            state, stop = disposition(result)
                            if (
                                retries is not None
                                and result.get("output") is None
                                and (result.get("error") or {}).get("kind")
                                in {
                                    "invalid_response",
                                    "incomplete_or_refused_response",
                                }
                            ):
                                state = "retry"
                            if (
                                retries is not None
                                and (result.get("error") or {}).get("kind")
                                == "ContextWindowExceededError"
                            ):
                                stop = True
                            if (
                                state == "retry"
                                or retries is not None
                                and state == "paused"
                            ) and index == max_attempts:
                                state = "completed"
                            result = result | {"state": state}
                            if retries is not None:
                                result["supplying_attempt"] = index
                            attempt = attempt_counts.get(key, 0) + 1
                            entry = {
                                "id": row["id"],
                                "stage": stage,
                                "retry_index": index,
                                "attempt": attempt,
                                "result": result,
                            }
                            append_attempt(directory, entry)
                            entries.append(entry)
                            attempt_counts[key] = attempt
                            latest[key] = entry
                            row[stage] = result
                            dirty = True
                            if stop:
                                emit(dumps({"dispatch_stopped": result["error"]}))
                                stopped = True
                                if stop_error is None:
                                    stop_error = result["error"]
                            if state != "retry" or stopped:
                                break
                            await asyncio.sleep(index)

                workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
                try:
                    await asyncio.gather(*workers)
                finally:
                    # On interruption/error, stop sibling workers before saving.
                    for task in workers:
                        task.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
                if stopped:
                    return {
                        "stopped": True,
                        "error": stop_error,
                        "attempt_totals": totals(entries, dispatches),
                    }
                if dirty:
                    save_annotations(directory, expected, rows, metadata["stages"])
                    dirty = False
                emit(
                    dumps(
                        {
                            "stage_finished": stage,
                            "attempt_totals": totals(entries, dispatches),
                        }
                    )
                )
        finally:
            if dirty:
                save_annotations(directory, expected, rows, metadata["stages"])
            metadata["attempt_totals"] = totals(entries, dispatches)
            atomic_write(directory / "run.json", dumps(metadata) + "\n")
            emit(
                dumps(
                    {"stopped": stopped, "attempt_totals": metadata["attempt_totals"]}
                )
            )
        return {
            "stopped": False,
            "attempt_totals": totals(entries, dispatches),
            "failed_decisions": sum(
                r[stage] is not None and r[stage].get("error") is not None
                for stage in stages
                for r in rows
            ),
        }
