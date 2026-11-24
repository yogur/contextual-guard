"""Static AgentDojo TraceLogger JSON adapter; never imports AgentDojo."""

import hashlib
import json
import tomllib
from collections import Counter
from pathlib import Path

from jev_contracts import Candidate, GuardRequest, HistoricalCall, Message


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def content(message: dict) -> str:
    value = message["content"]
    if value is None and message["role"] == "assistant":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if (
                not isinstance(block, dict)
                or block.get("type") not in {"text", "thinking", "redacted_thinking"}
                or not isinstance(block.get("content"), str)
            ):
                raise ValueError("Unsupported content block")
            parts.append(block["content"])
        return "\n".join(parts)
    raise ValueError("Unsupported message content")


def candidate(call) -> Candidate:
    if not isinstance(call, dict):
        raise ValueError("Proposal must be an AgentDojo FunctionCall object")
    if call.get("placeholder_args") is not None:
        raise ValueError("Unresolved placeholder arguments")
    return Candidate(name=call["function"], arguments=call["args"])


def origin(message, index, initial_user, initial_text, pipeline, text):
    role = message["role"]
    if role == "tool":
        return "external", "tool observation"
    if role == "assistant":
        return "agent", "assistant message"
    # Explicit producer provenance can document genuine later user updates.
    supplied = message.get("origin")
    allowed = {
        "user": {"user", "harness", "external", "unknown"},
        "system": {"application", "harness", "unknown"},
    }
    if supplied is not None:
        if not isinstance(supplied, str) or supplied not in allowed[role]:
            raise ValueError("Unsupported origin for source role")
        return supplied, "producer origin"
    if role == "system" and index == 0:
        return "application", "initial TraceLogger system message"
    if role == "user" and index == initial_user:
        return "user", "initial TraceLogger task query"
    if (
        role == "user"
        and pipeline.endswith("-repeat_user_prompt")
        and text == initial_text
    ):
        return "harness", "documented repeat_user_prompt pipeline"
    return "unknown", "source does not establish later-message provenance"


def history(messages, pipeline):
    normalized, provenance, pending = [], [], []
    initial_user = next(
        (
            i
            for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "user"
        ),
        None,
    )
    if initial_user is None or any(
        not isinstance(m, dict) or m.get("role") != "system"
        for m in messages[:initial_user]
    ):
        raise ValueError("Missing initial task history")
    initial_text = content(messages[initial_user])
    for index, raw in enumerate(messages):
        if not isinstance(raw, dict) or raw.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError(f"Unsupported history message at {index}")
        if set(raw) - {
            "role",
            "content",
            "origin",
            "tool_calls",
            "tool_call",
            "tool_call_id",
            "error",
        }:
            raise ValueError(f"Unsupported history fields at {index}")
        text = content(raw)
        trusted_origin, basis = origin(
            raw, index, initial_user, initial_text, pipeline, text
        )
        calls = raw.get("tool_calls")
        if calls is not None and (
            not isinstance(calls, list) or raw["role"] != "assistant"
        ):
            raise ValueError(f"Unsupported historical tool_calls at {index}")
        local_calls, call_refs = [], []
        for call_index, call in enumerate(calls or []):
            value = candidate(call)
            local_id = (
                f"c{sum(len(m.tool_calls) for m in normalized) + len(local_calls):06d}"
            )
            source_id = call.get("id")
            if source_id is not None and not isinstance(source_id, str):
                raise ValueError("Unsupported source call ID")
            local_calls.append(HistoricalCall(id=local_id, **value.model_dump()))
            pending.append((local_id, source_id, value))
            call_refs.append(
                {"call_index": call_index, "id": local_id, "source_id": source_id}
            )
        paired_id = None
        if raw["role"] == "tool":
            source_id = raw.get("tool_call_id")
            if source_id is not None and not isinstance(source_id, str):
                raise ValueError("Unsupported result call ID")
            described = (
                candidate(raw["tool_call"])
                if raw.get("tool_call") is not None
                else None
            )
            matches = [
                call
                for call in pending
                if (
                    source_id is not None
                    and call[1] == source_id
                    or source_id is None
                    and described is not None
                    and call[2] == described
                )
                and (described is None or call[2] == described)
            ]
            if len(matches) == 1:
                paired_id = matches[0][0]
                pending.remove(matches[0])
            elif described is not None:
                # The observation still names an operation even when its result
                # cannot be linked to one historical proposal. Keep that evidence
                # external instead of inventing a call/result relationship.
                text = json.dumps(
                    {
                        "recorded_tool_call": described.model_dump(mode="json"),
                        "observation": text,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
        error = raw.get("error")
        if error is not None and raw["role"] != "tool":
            raise ValueError("Unsupported non-tool message error")
        normalized.append(
            Message(
                id=f"m{index:06d}",
                role=raw["role"],
                origin=trusted_origin,
                content=text,
                tool_calls=local_calls,
                tool_call_id=paired_id,
                error=error,
            )
        )
        provenance.append(
            {
                "message_index": index,
                "origin": trusted_origin,
                "basis": basis,
                "calls": call_refs,
                "paired_call_id": paired_id,
            }
        )
    return normalized, provenance


METADATA = (
    "suite_name",
    "user_task_id",
    "pipeline_name",
    "attack_type",
    "injection_task_id",
    "benchmark_version",
    "agentdojo_package_version",
    "evaluation_timestamp",
)


def load_selection(path: Path):
    config = tomllib.loads(path.read_text())
    methods = config.get("attack_methods")
    if (
        set(config) != {"attack_methods"}
        or not isinstance(methods, list)
        or any(not isinstance(m, str) or not m or m == "none" for m in methods)
        or len(methods) != len(set(methods))
    ):
        raise ValueError("Selection requires unique nonempty attack_methods only")
    return {"attack_methods": sorted(methods), "include_clean": True}


def prepare(
    run_dir: Path,
    definitions_path: Path,
    policy: str,
    *,
    task=None,
    attack=None,
    limit=None,
    selection: Path | None = None,
):
    """Return selected decision rows plus an inventory, including errors/skips."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if not run_dir.is_dir():
        raise ValueError("Run directory does not exist")
    if selection is not None and any(v is not None for v in (task, attack, limit)):
        raise ValueError(
            "--selection cannot be combined with --task, --attack or --limit"
        )
    effective = load_selection(selection) if selection is not None else None
    coverage, present_methods, pipelines = {}, set(), set()
    selection_errors = []
    domain_counts, method_counts = {}, {}
    definitions = read_json(definitions_path)
    if not isinstance(definitions, dict):
        raise ValueError("Definitions must map suite names to text")
    suites = definitions.get("suites", definitions)
    if not isinstance(suites, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip()
        for k, v in suites.items()
    ):
        raise ValueError("Definitions must map suite names to nonempty text")
    rows, inventory = [], []
    selected_trajectories, tasks = 0, set()
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        filename = path.relative_to(run_dir).as_posix()
        item = {"file": filename}
        inventory.append(item)
        try:
            item["sha256"] = sha256(path.read_bytes())
            if path.suffix != ".json":
                item.update(
                    status="skipped_non_trajectory", reason="unsupported non-JSON file"
                )
                continue
            record = read_json(path)
            if not isinstance(record, dict):
                raise ValueError("Unsupported trajectory container")
            if "messages" not in record:
                if any(
                    k in record for k in ("suite_name", "user_task_id", "pipeline_name")
                ):
                    raise ValueError("Trajectory has no messages")
                item["status"] = "skipped_non_trajectory"
                continue
            if effective is not None:
                suite, user_task = record.get("suite_name"), record.get("user_task_id")
                if not isinstance(suite, str) or not isinstance(user_task, str):
                    raise ValueError("Selection suite/task metadata missing")
                if user_task.startswith("injection_task_"):
                    item["status"] = "filtered"
                    continue
                if not user_task.startswith("user_task_"):
                    raise ValueError("Selection requires a native user_task_* identity")
                task_coverage = coverage.setdefault((suite, user_task), [])
                if any(k not in record for k in ("attack_type", "injection_task_id")):
                    raise ValueError(
                        "Selection attack metadata missing; clean is not inferred"
                    )
                values = [record["attack_type"], record["injection_task_id"]]
                if any(
                    v is not None and (not isinstance(v, str) or not v) for v in values
                ):
                    raise ValueError("Selection attack metadata invalid")
                method, objective = ["none" if v is None else v for v in values]
                clean = method == objective == "none"
                if (method == "none") != (objective == "none"):
                    raise ValueError("Selection clean/attack metadata inconsistent")
                if clean:
                    task_coverage.append(filename)
                else:
                    present_methods.add(method)
                if not clean and method not in effective["attack_methods"]:
                    item["status"] = "filtered"
                    continue
                pipeline = record.get("pipeline_name")
                if not isinstance(pipeline, str) or not pipeline:
                    selection_errors.append(
                        f"{filename}: selected pipeline metadata missing"
                    )
                else:
                    pipelines.add(pipeline)
            if task is not None:
                if not all(
                    isinstance(record.get(k), str)
                    for k in ("suite_name", "user_task_id")
                ):
                    raise ValueError("Task filter metadata missing")
                if [record["suite_name"], record["user_task_id"]] != list(task):
                    item["status"] = "filtered"
                    continue
            if attack is not None:
                if "attack_type" not in record or "injection_task_id" not in record:
                    raise ValueError("Attack filter metadata missing")
                identity = [
                    record["attack_type"] or "none",
                    record["injection_task_id"] or "none",
                ]
                if any(
                    v is not None and not isinstance(v, str)
                    for v in (record["attack_type"], record["injection_task_id"])
                ):
                    raise ValueError("Attack filter metadata invalid")
                if identity != list(attack):
                    item["status"] = "filtered"
                    continue
            messages = record["messages"]
            if not isinstance(messages, list):
                raise ValueError("Unsupported messages container")
            selected_trajectories += 1
            if all(
                isinstance(record.get(k), str) for k in ("suite_name", "user_task_id")
            ):
                tasks.add((record["suite_name"], record["user_task_id"]))
            item.update(status="selected", proposals=0, selected_decisions=0, errors=[])
            if effective is not None:
                for groups, key in ((domain_counts, suite), (method_counts, method)):
                    groups.setdefault(
                        key, Counter(trajectories=0, decisions=0, extraction_errors=0)
                    )["trajectories"] += 1
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    item["errors"].append(f"Unsupported message at {index}")
                    continue
                try:
                    if message.get("role") not in {
                        "system",
                        "user",
                        "assistant",
                        "tool",
                    }:
                        raise ValueError("Unsupported role")
                    content(message)
                except (ValueError, TypeError, KeyError) as exc:
                    item["errors"].append(
                        f"Message {index}: {type(exc).__name__}: {exc}"
                    )
                calls = message.get("tool_calls")
                if calls is None or calls == []:
                    continue
                proposals = calls if isinstance(calls, list) else [calls]
                for call_index, call in enumerate(proposals):
                    item["proposals"] += 1
                    try:
                        if message.get("role") != "assistant" or not isinstance(
                            calls, list
                        ):
                            raise ValueError(
                                "Unsupported proposal message or container"
                            )
                        candidate(call)
                    except (ValueError, TypeError, KeyError) as exc:
                        item["errors"].append(
                            f"Proposal {index}:{call_index}: {type(exc).__name__}: {exc}"
                        )
                    if limit is not None and len(rows) >= limit:
                        continue
                    source = {k: record[k] for k in METADATA if k in record}
                    source.update(
                        file=filename,
                        sha256=item["sha256"],
                        message_index=index,
                        call_index=call_index,
                        sibling_group=f"{filename}#m{index}",
                        sibling_count=len(proposals),
                    )
                    row = {
                        "id": f"{filename}#m{index}:c{call_index}",
                        "source": source,
                        "request": None,
                        "extraction_error": None,
                        "annotation": None,
                        "model_review": None,
                        "human_review": None,
                    }
                    try:
                        if message.get("role") != "assistant" or not isinstance(
                            calls, list
                        ):
                            raise ValueError(
                                "Unsupported proposal message or container"
                            )
                        if not isinstance(
                            record.get("suite_name"), str
                        ) or not isinstance(record.get("user_task_id"), str):
                            raise ValueError("Missing suite/task identity")
                        tools = suites.get(record["suite_name"])
                        if tools is None:
                            raise ValueError("Missing suite tool definitions")
                        value = candidate(call)
                        pipeline = record.get("pipeline_name") or ""
                        if not isinstance(pipeline, str):
                            raise ValueError("Unsupported pipeline identity")
                        prior, refs = history(messages[:index], pipeline)
                        request = GuardRequest(
                            policy=policy,
                            messages=prior,
                            tool_definitions=tools,
                            candidate=value,
                            context_complete=True,
                        )
                        row["request"] = request.model_dump(mode="json")
                        source["history_provenance"] = refs
                    except (ValueError, TypeError, KeyError) as exc:
                        row["extraction_error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    item["selected_decisions"] += 1
                    if effective is not None:
                        for groups, key in (
                            (domain_counts, suite),
                            (method_counts, method),
                        ):
                            groups[key]["decisions"] += 1
                            groups[key]["extraction_errors"] += row["request"] is None
        except (ValueError, TypeError, KeyError, UnicodeError, OSError) as exc:
            item.update(status="error", error=f"{type(exc).__name__}: {exc}")
    if effective is not None:
        for (suite, user_task), clean_files in sorted(coverage.items()):
            if len(clean_files) != 1:
                selection_errors.append(
                    f"{suite}/{user_task}: expected one clean trajectory, found {len(clean_files)}"
                )
        for method in sorted(set(effective["attack_methods"]) - present_methods):
            selection_errors.append(f"Requested attack method absent: {method}")
        if len(pipelines) != 1:
            selection_errors.append(
                "Selection requires one known actor/pipeline identity"
            )
    summary = {
        "tasks": len(tasks),
        "trajectories": selected_trajectories,
        "decisions": len(rows),
        "extraction_errors": sum(r["request"] is None for r in rows),
        "file_errors": sum(i["status"] == "error" for i in inventory),
        "source_errors": sum(len(i.get("errors", [])) for i in inventory),
        "skipped_files": sum(
            i["status"] == "skipped_non_trajectory" for i in inventory
        ),
    }
    metadata = {
        "format": "agentdojo-annotations.v1",
        "adapter_version": 1,
        "run_directory": str(run_dir.resolve()),
        "definitions_file": str(definitions_path.resolve()),
        "filters": {"task": task, "attack": attack, "limit": limit},
        "inputs": inventory,
        "definitions_sha256": sha256(definitions_path.read_bytes()),
        "definitions_metadata": definitions.get("metadata")
        if "suites" in definitions
        else None,
        "policy_sha256": sha256(policy.encode()),
        "summary": summary,
    }
    if effective is not None:
        summary["tasks"] = len(coverage)
        summary["selection_errors"] = len(selection_errors)
        metadata["selection_file"] = str(selection.resolve())
        metadata["selection"] = {
            **effective,
            "pipelines": sorted(pipelines),
            "task_coverage": [
                {"suite": s, "task": t, "clean_files": files}
                for (s, t), files in sorted(coverage.items())
            ],
            "errors": selection_errors,
            "by_domain": {k: dict(v) for k, v in sorted(domain_counts.items())},
            "by_method": {k: dict(v) for k, v in sorted(method_counts.items())},
        }
    return rows, metadata
