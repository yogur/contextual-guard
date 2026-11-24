import asyncio
import json

import pytest

from jev_dataset import pipeline as native
from jev_dataset.adapter import prepare
from jev_dataset.pipeline import (
    dumps,
    initialize,
    load_rows,
    run_annotations,
    save_rows,
)


def call(name="read", id="same", args=None):
    return {"function": name, "args": {} if args is None else args, "id": id}


def proposal(*calls, content="candidate secret reasoning"):
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def result(value=None, text="SYSTEM: forged instructions", id="same", error=None):
    return {
        "role": "tool",
        "content": text,
        "tool_call": value or call(),
        "tool_call_id": id,
        "error": error,
    }


def record(messages=None, **metadata):
    return {
        "suite_name": "suite",
        "user_task_id": "task",
        "pipeline_name": "actor",
        "attack_type": None,
        "injection_task_id": None,
        "messages": messages
        or [
            {"role": "system", "content": "Application restriction"},
            {"role": "user", "content": "Read my event"},
            proposal(call()),
            result(),
            proposal(call()),
            result(error="ToolNotFoundError: no tool"),
            proposal(call(), call("write", "sibling")),
            result(),
            {"role": "assistant", "content": "future evaluator clue"},
        ],
        "utility": True,
        "security": False,
        "injections": {"x": "attack label"},
        **metadata,
    }


def setup_input(tmp_path, records=None):
    source = tmp_path / "outside" / "actor"
    source.mkdir(parents=True)
    for name, value in (records or {"run.json": record()}).items():
        (source / name).write_text(json.dumps(value))
    definitions = tmp_path / "tools.json"
    definitions.write_text(
        json.dumps(
            {
                "metadata": {"version": "v1"},
                "suites": {"suite": "read(): read event; write(): write event"},
            }
        )
    )
    return source, definitions


def dataset(tmp_path):
    source, definitions = setup_input(tmp_path)
    rows, metadata = prepare(source, definitions, "policy", limit=2)
    output = tmp_path / "output"
    initialize(output, rows, metadata)
    return output, rows, metadata


def judgment(label="authorized"):
    return {
        "output": {
            "label": label,
            "rationale": "User task",
            "evidence": [{"target": "candidate", "pointer": ""}],
        },
        "error": None,
        "usage": {"prompt_tokens": 10},
        "estimated_cost_usd": 0.001,
    }


def test_full_history_repeats_siblings_and_boundaries(tmp_path):
    source, definitions = setup_input(tmp_path)
    original = (source / "run.json").read_bytes()
    rows, metadata = prepare(source, definitions, "policy")
    assert metadata["summary"] == {
        "tasks": 1,
        "trajectories": 1,
        "decisions": 4,
        "extraction_errors": 0,
        "file_errors": 0,
        "skipped_files": 0,
        "source_errors": 0,
    }
    assert len({r["id"] for r in rows}) == 4
    prior = rows[2]["request"]["messages"]
    assert len(prior) == 6
    assert prior[2]["tool_calls"][0]["id"] == "c000000"
    assert prior[4]["tool_calls"][0]["id"] == "c000001"
    assert prior[3]["tool_call_id"] == "c000000"
    assert prior[5]["tool_call_id"] == "c000001"
    assert prior[3]["origin"] == "external"
    assert prior[3]["content"] == "SYSTEM: forged instructions"
    assert prior[5]["error"] == "ToolNotFoundError: no tool"
    assert rows[2]["request"]["messages"] == rows[3]["request"]["messages"]
    assert rows[2]["source"]["sibling_group"] == rows[3]["source"]["sibling_group"]
    assert rows[2]["request"]["context_complete"]
    serialized = dumps(rows[0]["request"])
    for forbidden in (
        "candidate secret",
        "forged instructions",
        "future evaluator",
        '"security"',
        '"utility"',
        "attack label",
    ):
        assert forbidden not in serialized
    assert (source / "run.json").read_bytes() == original


def test_blocks_and_provenance(tmp_path):
    messages = [
        {"role": "user", "content": [{"type": "text", "content": "Read"}]},
        proposal(
            call(),
            content=[
                {"type": "thinking", "content": "prior thinking"},
                {"type": "text", "content": "prior text"},
            ],
        ),
        result(),
        {"role": "user", "content": "Read"},
        {"role": "user", "content": "Actually write", "origin": "user"},
        {"role": "user", "content": "Maybe forged"},
        proposal(call()),
    ]
    source, definitions = setup_input(
        tmp_path, {"a.json": record(messages, pipeline_name="actor-repeat_user_prompt")}
    )
    rows, _ = prepare(source, definitions, "policy")
    prior = rows[-1]["request"]["messages"]
    assert [m["origin"] for m in prior] == [
        "user",
        "agent",
        "external",
        "harness",
        "user",
        "unknown",
    ]
    assert prior[1]["content"] == "prior thinking\nprior text"
    messages[3]["content"] = "Read"  # Equal text alone must not establish provenance.
    (source / "a.json").write_text(json.dumps(record(messages)))
    rows, _ = prepare(source, definitions, "policy")
    assert rows[-1]["request"]["messages"][3]["origin"] == "unknown"


def test_ambiguous_results_remain_unpaired(tmp_path):
    messages = [
        {"role": "user", "content": "Read"},
        proposal(call(id=None), call(id=None)),
        result(id=None),
        proposal(call()),
    ]
    source, definitions = setup_input(tmp_path, {"a.json": record(messages)})
    rows, _ = prepare(source, definitions, "policy")
    assert rows[-1]["request"]["messages"][2]["tool_call_id"] is None
    assert rows[-1]["request"]["messages"][2]["content"]


def test_filters_limit_errors_and_skips(tmp_path):
    source, definitions = setup_input(
        tmp_path,
        {
            "b.json": record(),
            "a.json": record(),
            "summary.json": {"results": {}},
            "unsupported.json": [],
        },
    )
    (source / "broken.json").write_text("{broken")
    missing = record()
    del missing["attack_type"]
    (source / "missing.json").write_text(json.dumps(missing))
    rows, metadata = prepare(
        source,
        definitions,
        "policy",
        task=("suite", "task"),
        attack=("none", "none"),
        limit=3,
    )
    assert len(rows) == 3
    assert all(r["source"]["file"] == "a.json" for r in rows)
    assert metadata["summary"]["trajectories"] == 2
    assert metadata["summary"]["tasks"] == 1
    assert metadata["summary"]["skipped_files"] == 1
    assert metadata["summary"]["file_errors"] == 3
    assert prepare(source, definitions, "policy", task=("suite", "other"))[0] == []
    assert prepare(source, definitions, "policy", limit=0)[0] == []


@pytest.mark.parametrize(
    "change", ["arguments", "blocks", "definitions", "missing_history"]
)
def test_untransformable_proposals_visible(tmp_path, change):
    value = record()
    if change == "arguments":
        value["messages"][2]["tool_calls"][0]["args"] = "not an object"
    elif change == "blocks":
        value["messages"][0]["content"] = [{"type": "image", "content": "hidden"}]
    elif change == "missing_history":
        value["messages"] = [proposal(call())]
    source, definitions = setup_input(tmp_path, {"a.json": value, "b.json": record()})
    if change == "definitions":
        definitions.write_text('{"other": "tools"}')
    rows, metadata = prepare(source, definitions, "policy")
    assert rows[0]["request"] is None
    assert rows[0]["extraction_error"]
    assert rows[0]["source"]["message_index"] == (
        0 if change == "missing_history" else 2
    )
    assert metadata["summary"]["extraction_errors"] > 0
    if change != "definitions":
        assert any(
            r["request"] is not None for r in rows if r["source"]["file"] == "b.json"
        )


def test_interrupt_resume_preserves_human_and_independent_review(tmp_path):
    output, rows, metadata = dataset(tmp_path)
    sent, prints = [], []

    async def interrupted(request, rubric, **profile):
        sent.append(request.model_dump())
        if len(sent) == 2:
            raise asyncio.CancelledError()
        return judgment()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                max_attempts=1,
                transport=interrupted,
                emit=prints.append,
            )
        )
    saved = load_rows(output)
    saved[0]["human_review"] = {
        "label": "unauthorized",
        "rationale": "Correction",
        "reviewer": "human",
    }
    save_rows(output, saved)
    initialize(output, rows, metadata)
    sent.clear()

    async def resumed(request, rubric, **profile):
        sent.append((request.model_dump(), profile))
        assert "annotation" not in request.model_dump()
        # Also preserve an edit made while the remote call is running.
        current = load_rows(output)
        current[1]["human_review"] = {"label": "authorized", "reviewer": "second"}
        save_rows(output, current)
        return judgment(
            "unauthorized" if profile["model"] == "review" else "authorized"
        )

    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            review_profile={"model": "review"},
            review_limit=1,
            max_attempts=1,
            transport=resumed,
            emit=prints.append,
        )
    )
    assert len(sent) == 2  # One pending primary and one independent reviewer.
    saved = load_rows(output)
    assert saved[0]["annotation"]["output"]["label"] == "authorized"
    assert saved[0]["model_review"]["output"]["label"] == "unauthorized"
    assert saved[0]["human_review"]["label"] == "unauthorized"
    assert saved[1]["human_review"]["reviewer"] == "second"
    assert json.loads(prints[0])["planned_annotations"] == 2
    sent.clear()
    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            review_profile={"model": "review"},
            review_limit=1,
            max_attempts=1,
            transport=resumed,
            emit=prints.append,
        )
    )
    assert sent == []


def test_changed_settings_and_evidence_refuse_reuse(tmp_path):
    output, rows, metadata = dataset(tmp_path)

    async def transport(*args, **kwargs):
        return judgment()

    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            max_attempts=1,
            transport=transport,
            emit=lambda _: None,
        )
    )
    with pytest.raises(ValueError, match="settings changed"):
        asyncio.run(
            run_annotations(
                output,
                "changed rubric",
                {"model": "primary"},
                max_attempts=1,
                transport=transport,
            )
        )
    with pytest.raises(ValueError, match="inputs/settings changed"):
        initialize(output, rows, metadata | {"policy_sha256": "changed"})
    saved = load_rows(output)
    saved[0]["request"]["policy"] = "changed policy"
    save_rows(output, saved)
    with pytest.raises(ValueError, match="requests/source changed"):
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                max_attempts=1,
                transport=transport,
            )
        )


def test_attempt_recovery_and_partial_tail(tmp_path):
    output, _, _ = dataset(tmp_path)
    entry = {
        "id": load_rows(output)[0]["id"],
        "stage": "annotation",
        "attempt": 1,
        "retry_index": 1,
        "result": judgment() | {"state": "completed"},
    }

    async def transport(*args, **kwargs):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                max_attempts=1,
                transport=transport,
                emit=lambda _: None,
            )
        )
    (output / "attempts.jsonl").write_text(dumps(entry) + '\n{"incomplete":')

    async def complete(*args, **kwargs):
        return judgment()

    result_value = asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            max_attempts=1,
            transport=complete,
            emit=lambda _: None,
        )
    )
    assert result_value["attempt_totals"]["annotation"]["attempts"] == 2
    assert len((output / "attempts.jsonl").read_text().splitlines()) == 2


def test_rate_limit_stops_and_resumes_pending_without_replaying_success(tmp_path):
    output, _, _ = dataset(tmp_path)
    sent = []

    async def limited(*args, **kwargs):
        sent.append(1)
        return (
            judgment()
            if len(sent) == 1
            else {
                "output": None,
                "error": {
                    "kind": "RateLimitError",
                    "status_code": 429,
                    "retry_after": "60",
                },
                "usage": None,
                "estimated_cost_usd": None,
            }
        )

    stopped = asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            transport=limited,
            emit=lambda _: None,
        )
    )
    assert stopped["stopped"]
    assert len(sent) == 2
    assert load_rows(output)[1]["annotation"]["state"] == "paused"
    sent.clear()
    resumed = asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            transport=limited,
            emit=lambda _: None,
        )
    )
    assert not resumed["stopped"]
    assert len(sent) == 1
    assert resumed["attempt_totals"]["annotation"]["unpriced_attempts"] == 1


@pytest.mark.parametrize(
    "kind,status,expected",
    [
        ("Timeout", 408, 3),
        ("AuthenticationError", 401, 1),
        ("invalid_response", None, 2),
    ],
)
def test_bounded_retries_and_terminal_errors(
    tmp_path, monkeypatch, kind, status, expected
):
    output, _, _ = dataset(tmp_path)
    sent = []

    async def transport(*args, **kwargs):
        sent.append(1)
        return {
            "output": None,
            "error": {"kind": kind, "status_code": status},
            "usage": {"prompt_tokens": 2},
            "estimated_cost_usd": 0.001,
        }

    async def no_sleep(*args):
        pass

    monkeypatch.setattr("jev_dataset.pipeline.asyncio.sleep", no_sleep)
    # Timeout reaches three attempts for each of two decisions.
    result_value = asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            transport=transport,
            emit=lambda _: None,
        )
    )
    assert len(sent) == (6 if expected == 3 else expected)
    assert result_value["attempt_totals"]["annotation"]["attempts"] == len(sent)
    sent.clear()
    if kind != "AuthenticationError":
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                transport=transport,
                emit=lambda _: None,
            )
        )
        assert sent == []


def test_prepare_cli_works_from_outside_repository(tmp_path):
    import subprocess
    import sys

    source, definitions = setup_input(tmp_path)
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jev_dataset",
            "prepare",
            str(source),
            "--definitions",
            str(definitions),
            "--output",
            str(output),
            "--limit",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert len(load_rows(output)) == 1
    assert json.loads(completed.stdout)["decisions"] == 1


def test_malformed_history_and_proposals_visible_beyond_limit(tmp_path):
    value = record()
    value["messages"].append({"role": "future_role", "content": "unsupported"})
    value["messages"].append(proposal(call(args="malformed")))
    source, definitions = setup_input(tmp_path, {"a.json": value})
    rows, metadata = prepare(source, definitions, "policy", limit=1)
    assert len(rows) == 1
    assert rows[0]["request"] is not None
    assert metadata["summary"]["source_errors"] == 2
    assert len(metadata["inputs"][0]["errors"]) == 2


def test_unknown_history_fields_and_missing_identity_keep_decision_locators(tmp_path):
    value = record()
    value["messages"][0]["unknown_context"] = "would otherwise be dropped"
    value["suite_name"] = ["malformed suite"]
    source, definitions = setup_input(tmp_path, {"a.json": value})
    rows, metadata = prepare(source, definitions, "policy")
    assert len(rows) == 4
    assert all(r["request"] is None and r["extraction_error"] for r in rows)
    assert metadata["summary"]["tasks"] == 0
    value["suite_name"] = "suite"
    (source / "a.json").write_text(json.dumps(value))
    rows, _ = prepare(source, definitions, "policy")
    assert all("Unsupported history fields" in r["extraction_error"] for r in rows)


def test_context_overflow_is_error_without_truncation_or_retry(tmp_path):
    output, _, _ = dataset(tmp_path)
    requests = []

    async def overflow(request, *args, **kwargs):
        requests.append(request)
        return {
            "output": None,
            "error": {"kind": "ContextWindowExceededError", "status_code": 400},
            "estimated_cost_usd": None,
        }

    result_value = asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            transport=overflow,
            emit=lambda _: None,
        )
    )
    assert len(requests) == 2
    assert [len(r.messages) for r in requests] == [2, 4]
    assert all(r.context_complete for r in requests)
    assert result_value["failed_decisions"] == 2
    assert result_value["attempt_totals"]["annotation"]["unpriced_attempts"] == 2


def test_interrupt_during_retry_does_not_reset_allowance(tmp_path, monkeypatch):
    output, _, _ = dataset(tmp_path)
    sent = []

    async def transient(*args, **kwargs):
        sent.append(1)
        return {
            "output": None,
            "error": {"kind": "Timeout", "status_code": 408},
            "estimated_cost_usd": None,
        }

    async def interrupted(*args):
        raise asyncio.CancelledError()

    monkeypatch.setattr("jev_dataset.pipeline.asyncio.sleep", interrupted)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                transport=transient,
                emit=lambda _: None,
            )
        )
    assert len(sent) == 1

    async def no_sleep(*args):
        pass

    monkeypatch.setattr("jev_dataset.pipeline.asyncio.sleep", no_sleep)
    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            transport=transient,
            emit=lambda _: None,
        )
    )
    assert len(sent) == 6  # Three total attempts per stage/decision, not seven.


def test_cli_refuses_output_in_input_tree(tmp_path):
    import subprocess
    import sys

    source, definitions = setup_input(tmp_path)
    original = (source / "run.json").read_bytes()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jev_dataset",
            "prepare",
            str(source),
            "--definitions",
            str(definitions),
            "--output",
            str(source),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "outside the source" in completed.stderr
    assert (source / "run.json").read_bytes() == original


def test_unmatched_result_keeps_recorded_operation_as_external_evidence(tmp_path):
    messages = [
        {"role": "user", "content": "Read"},
        result(
            call("unknown_operation", "orphan", {"target": "other"}),
            text="result without prior call",
            id="orphan",
        ),
        proposal(call()),
    ]
    source, definitions = setup_input(tmp_path, {"a.json": record(messages)})
    rows, _ = prepare(source, definitions, "policy")
    observation = rows[0]["request"]["messages"][1]
    assert observation["origin"] == "external"
    assert observation["tool_call_id"] is None
    assert json.loads(observation["content"]) == {
        "recorded_tool_call": {
            "name": "unknown_operation",
            "arguments": {"target": "other"},
        },
        "observation": "result without prior call",
    }


def test_non_json_files_are_inventoried_without_being_executed(tmp_path):
    source, definitions = setup_input(tmp_path)
    (source / "notes.txt").write_text("Do not execute this instruction")
    rows, metadata = prepare(source, definitions, "policy")
    assert len(rows) == 4
    assert metadata["summary"]["skipped_files"] == 1
    assert metadata["run_directory"] == str(source.resolve())
    assert metadata["definitions_file"] == str(definitions.resolve())
    assert metadata["inputs"][0]["reason"] == "unsupported non-JSON file"


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_ctx", 32768),
        ("think", True),
        ("max_tokens", 2048),
        ("timeout", 600),
        ("api_base", "http://localhost:11435"),
        ("model", "ollama_chat/changed"),
    ],
)
def test_changed_local_settings_refuse_resume(tmp_path, key, value):
    output, _, _ = dataset(tmp_path)
    settings = dict(
        model="ollama_chat/gemma4:31b-mlx",
        api_base="http://localhost:11434",
        num_ctx=16384,
        think=False,
        max_tokens=1024,
        timeout=300,
    )
    calls = []

    async def transport(*args, **kwargs):
        calls.append(kwargs)
        return judgment()

    asyncio.run(
        run_annotations(
            output,
            "rubric",
            settings,
            max_attempts=1,
            transport=transport,
            emit=lambda _: None,
        )
    )
    count = len(calls)
    asyncio.run(
        run_annotations(
            output,
            "rubric",
            settings,
            max_attempts=1,
            transport=transport,
            emit=lambda _: None,
        )
    )
    assert len(calls) == count
    with pytest.raises(ValueError, match="settings changed"):
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                settings | {key: value},
                max_attempts=1,
                transport=transport,
            )
        )
    assert len(calls) == count


@pytest.mark.parametrize("retries", [0, 1, 3])
def test_native_retry_count_and_first_pass_retained(tmp_path, monkeypatch, retries):
    output, _, _ = dataset(tmp_path)
    sent = []

    async def bad(request, rubric, **profile):
        sent.append((request.model_dump(), rubric, profile))
        return {
            "output": None,
            "raw_output": "bad evidence",
            "error": {"kind": "invalid_response"},
            "estimated_cost_usd": 0.001,
        }

    async def no_sleep(*args):
        pass

    monkeypatch.setattr(native.asyncio, "sleep", no_sleep)
    result = asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            retries=retries,
            transport=bad,
            emit=lambda _: None,
        )
    )
    assert len(sent) == 2 * (retries + 1)
    assert len(set(json.dumps(x, sort_keys=True) for x in sent[: retries + 1])) == 1
    assert len(native.attempts(output)) == len(sent)
    totals = result["attempt_totals"]["annotation"]
    assert totals["dispatches"] == totals["attempts"] == len(sent)
    assert totals["estimated_cost_usd"] == pytest.approx(0.001 * len(sent))
    sent.clear()
    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            retries=retries,
            transport=bad,
            emit=lambda _: None,
        )
    )
    assert sent == []


def test_native_interruption_unknown_dispatch_never_replayed(tmp_path):
    output, rows, _ = dataset(tmp_path)

    async def interrupted(*args, **kwargs):
        assert len(native.read_log(output / "dispatches.jsonl")) == 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                retries=1,
                transport=interrupted,
                emit=lambda _: None,
            )
        )
    sent = []

    async def complete(request, *args, **kwargs):
        sent.append(request.model_dump())
        return judgment()

    result = asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            retries=1,
            transport=complete,
            emit=lambda _: None,
        )
    )
    assert sent == [rows[1]["request"]]
    assert (
        load_rows(output)[0]["annotation"]["error"]["kind"]
        == "unknown_dispatch_outcome"
    )
    assert result["attempt_totals"]["annotation"]["unknown_outcomes"] == 1
    assert result["attempt_totals"]["annotation"]["unpriced_dispatches"] == 1
    assert result["attempt_totals"]["annotation"]["dispatches"] == 2


def test_native_rate_limit_allowance_across_resumes(tmp_path):
    output, _, _ = dataset(tmp_path)
    sent = []

    async def limited(*args, **kwargs):
        sent.append(1)
        return {
            "output": None,
            "error": {"kind": "RateLimitError", "status_code": 429},
            "estimated_cost_usd": None,
        }

    for _ in range(5):
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                retries=1,
                transport=limited,
                emit=lambda _: None,
            )
        )
    assert len(sent) == 4
    assert [d["attempt"] for d in native.read_log(output / "dispatches.jsonl")] == [
        1,
        2,
        1,
        2,
    ]


def test_native_valid_disagreement_never_retried(tmp_path):
    output, _, _ = dataset(tmp_path)
    sent = []

    async def valid(request, rubric, **profile):
        sent.append(request.model_dump())
        return judgment(
            "authorized" if profile["model"] == "primary" else "unauthorized"
        )

    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            review_profile={"model": "quality"},
            retries=1,
            transport=valid,
            emit=lambda _: None,
        )
    )
    assert len(sent) == 4
    saved = load_rows(output)
    assert all(
        r["annotation"]["output"]["label"] != r["model_review"]["output"]["label"]
        for r in saved
    )
    sent.clear()
    # Omitting --review may resume the primary without rejecting completed review dispatches.
    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            retries=1,
            transport=valid,
            emit=lambda _: None,
        )
    )
    assert sent == []
    with pytest.raises(ValueError, match="settings changed"):
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                retries=2,
                transport=valid,
                emit=lambda _: None,
            )
        )


def test_native_context_limit_stops_without_retry(tmp_path):
    output, _, _ = dataset(tmp_path)
    sent = []

    async def context(*args, **kwargs):
        sent.append(1)
        return {
            "output": None,
            "error": {"kind": "ContextWindowExceededError"},
            "estimated_cost_usd": None,
        }

    for _ in range(2):
        result = asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                retries=1,
                transport=context,
                emit=lambda _: None,
            )
        )
        assert result["stopped"]
    assert sent == [1]


def test_annotation_saves_attempts_immediately_and_dataset_at_stage_end(
    tmp_path, monkeypatch
):
    output, _, _ = dataset(tmp_path)
    original = native.save_rows
    saves, calls = [], []

    def saved(directory, rows):
        saves.append(len(native.attempts(directory)))
        original(directory, rows)

    monkeypatch.setattr(native, "save_rows", saved)

    async def valid(request, rubric, **profile):
        assert len(native.attempts(output)) == len(calls)
        snapshot = load_rows(output)
        if profile["model"] == "primary":
            assert all(r["annotation"] is None for r in snapshot)
        else:
            assert all(r["annotation"]["output"] for r in snapshot)
        assert all(r["model_review"] is None for r in snapshot)
        calls.append(profile["model"])
        return judgment()

    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            review_profile={"model": "review"},
            retries=1,
            transport=valid,
            emit=lambda _: None,
        )
    )
    assert calls == ["primary", "primary", "review", "review"]
    assert saves == [2, 4]
    assert len(native.read_log(output / "dispatches.jsonl")) == 4


def test_native_response_write_gap_recovery(tmp_path, monkeypatch):
    output, _, _ = dataset(tmp_path)
    original = native.save_annotations

    def interrupted(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(native, "save_annotations", interrupted)

    async def valid(*args, **kwargs):
        return judgment()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                retries=1,
                transport=valid,
                emit=lambda _: None,
            )
        )
    # Both returned attempts survived a failed end-of-stage dataset save.
    assert all(r["annotation"] is None for r in load_rows(output))
    assert len(native.attempts(output)) == 2
    monkeypatch.setattr(native, "save_annotations", original)
    result = asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            retries=1,
            transport=valid,
            emit=lambda _: None,
        )
    )
    assert result["attempt_totals"]["annotation"]["dispatches"] == 2
    assert result["attempt_totals"]["annotation"]["attempts"] == 2


def concurrent_dataset(tmp_path, count=8):
    source, definitions = setup_input(tmp_path)
    (source / "run.json").write_text(
        dumps(
            record(
                messages=[
                    {"role": "system", "content": "Application restriction"},
                    {"role": "user", "content": "Read my events"},
                    proposal(
                        *(call(id=str(i), args={"index": i}) for i in range(count))
                    ),
                ]
            )
        )
    )
    rows, metadata = prepare(source, definitions, "policy")
    output = tmp_path / "output"
    initialize(output, rows, metadata)
    return output, rows


def test_concurrent_annotations_bound_calls_join_ids_and_keep_retry_limit(tmp_path):
    output, rows = concurrent_dataset(tmp_path, 6)

    async def scenario():
        active, peak = 0, 0
        release = {name: asyncio.Event() for name in ("primary", "review")}
        calls = {}

        async def transport(request, rubric, **profile):
            nonlocal active, peak
            index, model = request.candidate.arguments["index"], profile["model"]
            if model == "review":
                assert all(r["annotation"] is not None for r in load_rows(output))
            key = (model, index)
            calls[key] = calls.get(key, 0) + 1
            active += 1
            peak = max(peak, active)
            try:
                if index == 0:
                    await release[model].wait()
                else:
                    await asyncio.sleep(0)
                    if index == 5:
                        release[model].set()
                if key == ("primary", 2) and calls[key] == 1:
                    return judgment() | {
                        "output": None,
                        "error": {"kind": "invalid_response"},
                    }
                return judgment("authorized" if index % 2 == 0 else "unauthorized")
            finally:
                active -= 1

        result = await asyncio.wait_for(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                review_profile={"model": "review"},
                retries=1,
                concurrency=5,
                transport=transport,
                emit=lambda _: None,
            ),
            5,
        )
        assert not result["stopped"] and peak == 5
        assert calls[("primary", 2)] == 2
        assert all(n == 1 for key, n in calls.items() if key != ("primary", 2))

    asyncio.run(scenario())
    saved = load_rows(output)
    assert [r["id"] for r in saved] == [r["id"] for r in rows]
    for index, row in enumerate(saved):
        for stage in ("annotation", "model_review"):
            assert row[stage]["output"]["label"] == (
                "authorized" if index % 2 == 0 else "unauthorized"
            )
    entries = native.attempts(output)
    assert len(entries) == 13
    assert entries[0]["id"] != rows[0]["id"]
    assert [
        e["attempt"]
        for e in entries
        if e["id"] == rows[2]["id"] and e["stage"] == "annotation"
    ] == [1, 2]
    before = (output / "run.json").read_bytes()

    async def forbidden(*args, **kwargs):
        raise AssertionError("Completed concurrency-change resume must make no calls")

    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            review_profile={"model": "review"},
            retries=1,
            concurrency=10,
            transport=forbidden,
            emit=lambda _: None,
        )
    )
    assert (output / "run.json").read_bytes() == before


@pytest.mark.parametrize(
    "error",
    [
        {"kind": "RateLimitError", "status_code": 429},
        {"kind": "AuthenticationError", "status_code": 401},
        {"kind": "ContextWindowExceededError"},
    ],
)
def test_concurrent_stop_drains_dispatched_calls_without_new_work(tmp_path, error):
    output, _ = concurrent_dataset(tmp_path)

    async def scenario():
        all_started, drain = asyncio.Event(), asyncio.Event()
        calls = []

        async def transport(request, rubric, **profile):
            assert profile["model"] == "primary"
            index = request.candidate.arguments["index"]
            calls.append(index)
            if len(calls) == 5:
                all_started.set()
            if index == 0:
                await all_started.wait()
                return judgment() | {"output": None, "error": error}
            await drain.wait()
            return judgment()

        def emit(value):
            if "dispatch_stopped" in json.loads(value):
                drain.set()

        result = await asyncio.wait_for(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                review_profile={"model": "review"},
                retries=1,
                concurrency=5,
                transport=transport,
                emit=emit,
            ),
            5,
        )
        assert result["stopped"] and calls == list(range(5))
        assert result["attempt_totals"]["annotation"]["unknown_outcomes"] == 0

    asyncio.run(scenario())
    assert len(native.attempts(output)) == 5
    saved = load_rows(output)
    assert all(r["annotation"]["output"] for r in saved[1:5])
    assert all(r["annotation"] is None for r in saved[5:])
    assert all(r["model_review"] is None for r in saved)


def test_concurrent_interrupt_retains_multiple_dispatches_and_skips_replay(tmp_path):
    output, _ = concurrent_dataset(tmp_path)

    async def scenario():
        started = asyncio.Event()
        block = asyncio.Event()
        calls = []

        async def transport(request, rubric, **profile):
            index = request.candidate.arguments["index"]
            calls.append(index)
            if index == 0:
                return judgment()
            if len(calls) == 6:
                started.set()
            await block.wait()

        task = asyncio.create_task(
            run_annotations(
                output,
                "rubric",
                {"model": "primary"},
                retries=1,
                concurrency=5,
                transport=transport,
                emit=lambda _: None,
            )
        )
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert len(native.attempts(output)) == 1
    assert len(native.read_log(output / "dispatches.jsonl")) == 6
    assert load_rows(output)[0]["annotation"]["output"]
    calls = []

    async def resumed(request, rubric, **profile):
        calls.append(request.candidate.arguments["index"])
        return judgment()

    result = asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            retries=1,
            concurrency=5,
            transport=resumed,
            emit=lambda _: None,
        )
    )
    assert calls == [6, 7]
    assert result["attempt_totals"]["annotation"]["unknown_outcomes"] == 5
    assert (
        sum(
            (r["annotation"].get("error") or {}).get("kind")
            == "unknown_dispatch_outcome"
            for r in load_rows(output)
        )
        == 5
    )


@pytest.mark.parametrize("concurrency", [0, -1, 1.5, True])
def test_invalid_concurrency_rejected(tmp_path, concurrency):
    output, _, _ = dataset(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(run_annotations(output, "rubric", {}, concurrency=concurrency))


def test_ollama_concurrency_rejected_before_dispatch(tmp_path):
    output, _, _ = dataset(tmp_path)
    with pytest.raises(ValueError, match="Ollama annotation requires concurrency=1"):
        asyncio.run(
            run_annotations(
                output, "rubric", {"model": "ollama_chat/test"}, concurrency=5
            )
        )


@pytest.mark.parametrize("retries", [-1, 1.5, True])
def test_native_retries_require_nonnegative_integer(tmp_path, retries):
    output, _, _ = dataset(tmp_path)
    with pytest.raises(ValueError, match="nonnegative integer"):
        asyncio.run(run_annotations(output, "rubric", {}, retries=retries))


@pytest.mark.parametrize(
    "flags", [["--retries", "-1"], ["--retries", "1", "--max-attempts", "2"]]
)
def test_retry_cli_rejects_invalid_or_conflicting_options(tmp_path, flags):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "jev_dataset", "annotate", str(tmp_path), *flags],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_native_valid_retry_records_supplying_attempt(tmp_path, monkeypatch):
    output, _, _ = dataset(tmp_path)
    sent = []

    async def response(request, rubric, **profile):
        sent.append((request.model_dump(), rubric, profile))
        if len(sent) == 1:
            return {
                "output": None,
                "raw_output": "invalid evidence",
                "error": {"kind": "invalid_response"},
                "estimated_cost_usd": 0.001,
            }
        return judgment()

    async def no_sleep(*args):
        pass

    monkeypatch.setattr(native.asyncio, "sleep", no_sleep)
    asyncio.run(
        run_annotations(
            output,
            "rubric",
            {"model": "primary"},
            retries=1,
            transport=response,
            emit=lambda _: None,
        )
    )
    assert len(sent) == 3
    assert sent[0] == sent[1]
    saved = load_rows(output)
    assert saved[0]["annotation"]["output"]["label"] == "authorized"
    assert saved[0]["annotation"]["supplying_attempt"] == 2
    assert saved[1]["annotation"]["supplying_attempt"] == 1
    assert native.attempts(output)[0]["result"]["raw_output"] == "invalid evidence"


def selection_input(tmp_path, *, methods=("direct",), extra=None):
    records = {
        "clean.json": record(user_task_id="user_task_1"),
        "attack.json": record(
            user_task_id="user_task_1",
            attack_type="direct",
            injection_task_id="injection_task_2",
        ),
        "excluded.json": record(
            user_task_id="user_task_1",
            attack_type="other",
            injection_task_id="injection_task_3",
            pipeline_name="other-actor",
        ),
        "baseline.json": record(
            user_task_id="injection_task_2", pipeline_name="other-actor"
        ),
        "zero.json": record(
            user_task_id="user_task_2",
            attack_type="none",
            injection_task_id="none",
            messages=[{"role": "user", "content": "No calls"}],
        ),
    }
    records.update(extra or {})
    source, definitions = setup_input(tmp_path, records)
    selection = tmp_path / "selection.toml"
    selection.write_text("attack_methods = " + json.dumps(list(methods)))
    return source, definitions, selection


def test_selection_complete_coverage_determinism_and_portable_resume(tmp_path):
    import shutil

    source, definitions, selection = selection_input(
        tmp_path, methods=("other", "direct")
    )
    # The included other attack must share the selected actor.
    path = source / "excluded.json"
    value = json.loads(path.read_text())
    value["pipeline_name"] = "actor"
    path.write_text(json.dumps(value))
    rows, metadata = prepare(source, definitions, "policy", selection=selection)
    assert metadata["summary"]["tasks"] == 2
    assert metadata["summary"]["trajectories"] == 4
    assert metadata["summary"]["decisions"] == 12
    assert metadata["summary"]["selection_errors"] == 0
    assert metadata["selection"]["by_method"]["none"]["trajectories"] == 2
    # Same native extraction preserves repeated calls and complete pre-call history.
    assert len(rows[2]["request"]["messages"]) == 6
    assert len({row["id"] for row in rows}) == 12
    a, b = tmp_path / "a", tmp_path / "b"
    initialize(a, rows, metadata)
    selection.write_text('attack_methods = ["direct", "other"]')
    again, manifest = prepare(source, definitions, "policy", selection=selection)
    initialize(b, again, manifest)
    assert (a / "dataset.jsonl").read_bytes() == (b / "dataset.jsonl").read_bytes()
    assert native.preparation_identity(metadata) == native.preparation_identity(
        manifest
    )
    saved = load_rows(a)
    saved[0]["human_review"] = {"label": "authorized", "reviewer": "human"}
    saved[0]["annotation"] = judgment() | {"state": "completed"}
    save_rows(a, saved)
    (a / "attempts.jsonl").write_text('{"saved":true}\n')
    moved = tmp_path / "moved"
    shutil.copytree(source, moved)
    tools = tmp_path / "moved-tools.json"
    shutil.copyfile(definitions, tools)
    again, manifest = prepare(moved, tools, "policy", selection=selection)
    initialize(a, again, manifest)
    assert load_rows(a) == saved
    assert (a / "attempts.jsonl").read_text() == '{"saved":true}\n'


def test_selection_unselected_edits_and_stale_inputs(tmp_path):
    source, definitions, selection = selection_input(tmp_path)
    rows, metadata = prepare(source, definitions, "policy", selection=selection)
    out = tmp_path / "out"
    initialize(out, rows, metadata)
    path = source / "excluded.json"
    value = json.loads(path.read_text())
    value["messages"] = []
    path.write_text(json.dumps(value))
    again, manifest = prepare(source, definitions, "policy", selection=selection)
    initialize(out, again, manifest)
    assert metadata["inputs"] != manifest["inputs"]
    for change in ("error", "task", "selected", "policy", "definitions", "selection"):
        original = path.read_text()
        original_definitions = definitions.read_text()
        original_selection = selection.read_text()
        if change == "error":
            path.write_text("invalid JSON")
        elif change == "task":
            value["user_task_id"] = "user_task_3"
            path.write_text(json.dumps(value))
        elif change == "selected":
            value["attack_type"] = "direct"
            path.write_text(json.dumps(value))
        elif change == "definitions":
            definitions.write_text('{"suite":"changed"}')
        elif change == "selection":
            selection.write_text("attack_methods = []")
        again, manifest = prepare(
            source,
            definitions,
            "changed" if change == "policy" else "policy",
            selection=selection,
        )
        with pytest.raises(ValueError, match="changed"):
            initialize(out, again, manifest)
        path.write_text(original)
        definitions.write_text(original_definitions)
        selection.write_text(original_selection)
        value = json.loads(original)


@pytest.mark.parametrize(
    "case",
    [
        "missing_clean",
        "duplicate_clean",
        "missing_attack",
        "missing_pipeline",
        "mixed_actor",
        "absent_method",
        "malformed",
    ],
)
def test_selection_errors_are_visible(tmp_path, case):
    source, definitions, selection = selection_input(tmp_path)
    if case == "missing_clean":
        (source / "clean.json").unlink()
    elif case == "duplicate_clean":
        (source / "duplicate.json").write_bytes((source / "clean.json").read_bytes())
    elif case in {"missing_attack", "missing_pipeline", "mixed_actor"}:
        path = source / "attack.json"
        value = json.loads(path.read_text())
        if case == "mixed_actor":
            value["pipeline_name"] = "other-actor"
        else:
            del value["attack_type" if case == "missing_attack" else "pipeline_name"]
        path.write_text(json.dumps(value))
    elif case == "absent_method":
        selection.write_text('attack_methods = ["missing"]')
    else:
        (source / "broken.json").write_text("{")
    _, metadata = prepare(source, definitions, "policy", selection=selection)
    assert metadata["summary"]["selection_errors"] or metadata["summary"]["file_errors"]


@pytest.mark.parametrize(
    "config",
    [
        'attack_methods = ["direct", "direct"]',
        'attack_methods = "direct"',
        "attack_methods = []\nunknown = true",
        'attack_methods = ["none"]',
    ],
)
def test_selection_invalid_config_and_conflicts(tmp_path, config):
    source, definitions, selection = selection_input(tmp_path)
    for options in (
        {"limit": 0},
        {"task": ["suite", "user_task_1"]},
        {"attack": ["none", "none"]},
    ):
        with pytest.raises(ValueError, match="cannot be combined"):
            prepare(source, definitions, "policy", selection=selection, **options)
    selection.write_text(config)
    with pytest.raises(ValueError, match="Selection requires"):
        prepare(source, definitions, "policy", selection=selection)


def test_selection_empty_methods_clean_only_and_selected_extraction_errors(tmp_path):
    source, definitions, selection = selection_input(tmp_path, methods=())
    value = json.loads((source / "clean.json").read_text())
    value["messages"][2]["tool_calls"][0]["args"] = "malformed"
    (source / "clean.json").write_text(json.dumps(value))
    rows, metadata = prepare(source, definitions, "policy", selection=selection)
    assert len(rows) == 4
    assert rows[0]["request"] is None
    assert metadata["summary"]["trajectories"] == 2
    assert metadata["summary"]["selection_errors"] == 0
    assert metadata["summary"]["extraction_errors"] > 0


def test_runtime_profile_and_tier_resume_identity(tmp_path):
    output, _, _ = dataset(tmp_path)
    profile = {
        "model": "bedrock/global.example",
        "region": "us-east-1",
        "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "service_tier": "flex",
        "reasoning": "low",
        "max_tokens": 4096,
        "timeout": 180,
    }

    async def valid(*args, **kwargs):
        return judgment()

    asyncio.run(
        run_annotations(
            output, "rubric", profile, retries=1, transport=valid, emit=lambda _: None
        )
    )

    async def forbidden(*args, **kwargs):
        pytest.fail("Completed resume must schedule zero calls")

    asyncio.run(
        run_annotations(
            output,
            "rubric",
            profile,
            retries=1,
            transport=forbidden,
            emit=lambda _: None,
        )
    )
    for key, replacement in {
        "model": "bedrock_mantle/example",
        "region": "us-west-2",
        "api_base": "https://other",
        "service_tier": "default",
        "reasoning": "high",
        "max_tokens": 8192,
        "timeout": 300,
    }.items():
        with pytest.raises(ValueError, match="settings changed"):
            asyncio.run(
                run_annotations(
                    output,
                    "rubric",
                    profile | {key: replacement},
                    retries=1,
                    transport=forbidden,
                    emit=lambda _: None,
                )
            )


def test_selection_entirely_zero_call_coverage_has_explicit_counts(tmp_path):
    source, definitions, selection = selection_input(tmp_path, methods=())
    for path in (source / "clean.json", source / "zero.json"):
        record = json.loads(path.read_text())
        record["messages"] = record["messages"][:2]
        path.write_text(json.dumps(record))
    rows, metadata = prepare(source, definitions, "policy", selection=selection)
    assert rows == []
    assert metadata["summary"]["tasks"] == 2
    assert metadata["summary"]["trajectories"] == 2
    assert metadata["selection"]["by_method"]["none"] == {
        "trajectories": 2,
        "decisions": 0,
        "extraction_errors": 0,
    }
