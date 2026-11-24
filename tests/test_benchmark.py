import asyncio
import json

import pytest
from test_core import request
from test_guard import PROFILES, QUESTION, result

from jev_benchmark import digest, dumps, read_entries, run_benchmark
from jev_benchmark.report import latencies, preparation_costs, score, write_report


def row(index, label="authorized", human=None, extraction=False):
    return {
        "id": f"trace.json#m{index}:c0",
        "source": {
            "suite_name": "workspace",
            "user_task_id": "task-1",
            "file": "trace.json",
            "sibling_group": f"trace.json#m{index}",
            "sibling_count": 1,
        },
        "request": None if extraction else request().model_dump(mode="json"),
        "extraction_error": "bad call" if extraction else None,
        "annotation": None
        if label is None
        else {
            "state": "completed",
            "error": None,
            "model": "bedrock_mantle/openai.gpt-6-luna",
            "output": {
                "label": label,
                "rationale": "PRIVATE ANNOTATION",
                "evidence": [{"target": "candidate", "pointer": ""}],
            },
        },
        "human_review": {"label": human, "rationale": "PRIVATE HUMAN"}
        if human
        else None,
        "model_review": {
            "state": "completed",
            "error": None,
            "output": {
                "label": "unauthorized",
                "rationale": "PRIVATE REVIEW",
                "evidence": [{"target": "candidate", "pointer": ""}],
            },
        },
    }


def save(path, rows):
    path.write_text("".join(dumps(r) + "\n" for r in rows))


def fake_factory(calls, *, failures=None, interrupt_at=None):
    failures = failures or {}

    class Fake:
        def __init__(self, settings, question):
            self.name = "jev" if settings.provider == "typesafe" else "bedrock"

        async def assess(self, value):
            calls.append((self.name, value))
            if interrupt_at is not None and len(calls) == interrupt_at:
                raise KeyboardInterrupt
            failure = failures.get(
                (self.name, len([c for c in calls if c[0] == self.name]))
            )
            kwargs = {}
            if self.name == "jev" and not failure:
                kwargs = {
                    "probabilities": {
                        "authorized": 0.8,
                        "unauthorized": 0.1,
                        "insufficient_evidence": 0.1,
                    },
                    "confidence": 0.6,
                }
            return result(error=failure, backend=self.name, **kwargs)

        async def aclose(self):
            pass

    return Fake


def run(dataset, output, calls, **kwargs):
    factory = kwargs.pop("factory", fake_factory(calls))
    return asyncio.run(
        run_benchmark(
            dataset,
            output,
            PROFILES,
            QUESTION,
            factory=factory,
            emit=lambda text: None,
            **kwargs,
        )
    )


def test_sequential_order_isolation_skip_and_resume(tmp_path):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "benchmark"
    rows = [row(0, extraction=True), row(1), row(2, label=None)]
    save(dataset, rows)
    before = dataset.read_bytes()
    calls, planned = [], []
    summary = asyncio.run(
        run_benchmark(
            dataset,
            output,
            PROFILES,
            QUESTION,
            factory=fake_factory(calls),
            emit=planned.append,
        )
    )
    assert [name for name, value in calls] == ["jev", "bedrock", "jev", "bedrock"]
    assert all(value == request() for name, value in calls)
    assert all(
        "PRIVATE" not in value.model_dump_json() and "source" not in value.model_dump()
        for name, value in calls
    )
    assert json.loads(planned[0])["planned_calls"] == {"jev": 2, "bedrock": 2}
    assert summary["completed_assessments"] == 4 and summary["pending"] == 0
    assert dataset.read_bytes() == before
    entries, _ = read_entries(output)
    assert [e["first_call"] for e in entries] == [True, True, False, False]
    assert [e["execution_order"] for e in entries] == [1, 2, 3, 4]
    run(dataset, output, calls)
    assert len(calls) == 4
    run(dataset, tmp_path / "fresh", calls)
    assert len(calls) == 8
    limited = []
    run(dataset, tmp_path / "limited", limited, limit=1)
    assert limited == []


@pytest.mark.parametrize("failure", ["authentication", "configuration", "rate_limit"])
def test_backend_stops_other_continues_saved_errors_skip(tmp_path, failure):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "benchmark"
    save(dataset, [row(i) for i in range(3)])
    calls = []
    summary = run(
        dataset,
        output,
        calls,
        factory=fake_factory(calls, failures={("jev", 1): failure}),
    )
    assert [name for name, _ in calls] == ["jev", "bedrock", "bedrock", "bedrock"]
    assert summary["pending"] == 2 and summary["errors"] == 1
    report = write_report(dataset, output)
    assert report["complete"] is False
    metrics = report["backends"]["jev"]["tracks"]["primary_label_agreement"]
    assert metrics["N"] == 3 and metrics["errors_by_reference"]["authorized"] == 1
    assert metrics["pending_by_reference"]["authorized"] == 2
    resumed = []
    run(dataset, output, resumed)
    assert [name for name, _ in resumed] == ["jev", "jev"]
    entries, _ = read_entries(output)
    assert len(entries) == 6 and sum(e["first_call"] for e in entries) == 2


def test_interruption_and_partial_tail_resume(tmp_path):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "benchmark"
    save(dataset, [row(0), row(1)])
    calls = []
    with pytest.raises(KeyboardInterrupt):
        run(dataset, output, calls, factory=fake_factory(calls, interrupt_at=2))
    with (output / "verdicts.jsonl").open("a") as stream:
        stream.write('{"incomplete":')
    report = write_report(dataset, output)
    assert report["incomplete_verdict_log_tail"] and report["counts"]["pending"] == 3
    assert (output / "verdicts.jsonl").read_text().endswith('{"incomplete":')
    resumed = []
    run(dataset, output, resumed)
    assert [name for name, _ in resumed] == ["bedrock", "jev", "bedrock"]
    assert not read_entries(output)[1]


def test_resume_rejects_input_prompt_selection_configuration_changes(tmp_path):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "benchmark"
    rows = [row(0), row(1)]
    save(dataset, rows)
    calls = []
    run(dataset, output, calls)
    changed = PROFILES | {"jev": PROFILES["jev"].model_copy(update={"timeout": 5.0})}
    for profiles, question, limit in (
        (changed, QUESTION, None),
        (dict(reversed(list(PROFILES.items()))), QUESTION, None),
        (PROFILES, QUESTION | {"instructions": "changed"}, None),
        (PROFILES, QUESTION, 1),
    ):
        with pytest.raises(ValueError, match="changed"):
            asyncio.run(
                run_benchmark(
                    dataset,
                    output,
                    profiles,
                    question,
                    limit=limit,
                    factory=fake_factory(calls),
                )
            )
    rows[0]["human_review"] = {"label": "unauthorized"}
    save(dataset, rows)
    with pytest.raises(ValueError, match="changed"):
        run(dataset, output, calls)
    with pytest.raises(ValueError, match="identity changed"):
        write_report(dataset, output)
    assert len(calls) == 4


def test_denominators_errors_pending_abstention_and_human_override():
    rows = [
        row(0, "unauthorized"),
        row(1, "unauthorized"),
        row(2, "unauthorized"),
        row(3, "authorized", human="unauthorized"),
        row(4, "authorized"),
        row(5, "insufficient_evidence"),
        row(6, None, human="unauthorized"),
    ]
    outcomes = {
        rows[0]["id"]: result("unauthorized").model_dump(),
        rows[1]["id"]: result(error="timeout").model_dump(),
        rows[3]["id"]: result("unauthorized").model_dump(),
        rows[4]["id"]: result("insufficient_evidence").model_dump(),
        rows[5]["id"]: result("insufficient_evidence").model_dump(),
        rows[6]["id"]: result("unauthorized").model_dump(),
    }
    primary = score(rows, outcomes, "primary_label_agreement")
    assert primary["N"] == 6 and primary["three_way"] == {
        "numerator": 2,
        "denominator": 6,
        "value": 2 / 6,
    }
    assert primary["unauthorized_detection"]["value"] == 1 / 3
    assert primary["authorized_false_block_rate"]["value"] == 1 / 2
    assert primary["valid_verdict_coverage"]["numerator"] == 4
    assert primary["determinate_verdict_coverage"]["numerator"] == 2
    assert primary["insufficient_evidence_predictions"]["numerator"] == 2
    assert (
        primary["errors_by_reference"]["unauthorized"]
        == primary["pending_by_reference"]["unauthorized"]
        == 1
    )
    human = score(rows, outcomes, "human_reviewed_accuracy")
    assert human["N"] == 2 and human["three_way"]["value"] == 1.0
    assert human["authorized_false_block_rate"]["value"] is None
    empty = score([row(0, None)], {}, "primary_label_agreement")
    assert empty["N"] == 0 and empty["three_way"]["value"] is None


def test_report_is_offline_tracks_unknown_costs_slices_and_first_latency(
    tmp_path, monkeypatch
):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "benchmark"
    save(dataset, [row(0, human="unauthorized"), row(1, None), row(2, extraction=True)])
    run(dataset, output, [])

    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting cannot infer")

    monkeypatch.setattr("litellm.acompletion", forbidden)
    monkeypatch.setattr("typesafe_sdk.AsyncTypeSafeClient.system_one", forbidden)
    report = write_report(dataset, output)
    assert report == json.loads((output / "report.json").read_text())
    assert report["counts"]["unlabeled_primary"] == 1
    assert report["counts"]["tasks"] == report["counts"]["trajectories"] == 1
    backend = report["backends"]["bedrock"]
    assert backend["same_model_as_primary_annotation"]
    assert backend["tracks"]["primary_label_agreement"]["three_way"]["value"] == 1.0
    assert backend["tracks"]["human_reviewed_accuracy"]["three_way"]["value"] == 0.0
    assert report["model_review_diagnostic"]["disagreements_with_primary"] == 1
    assert backend["latency"]["successful"]["sample_count"] == 2
    assert backend["latency"]["subsequent_successful"]["sample_count"] == 1
    assert "cost" not in backend
    assert not report["preparation_costs"]["available"]
    assert "N/A" in (output / "report.md").read_text()
    assert set(report["slices"]) == {"domain", "tool"}
    assert latencies(list(range(1, 22)))["p95_seconds"] == 20


@pytest.mark.parametrize(
    "usage,write_total,read_total,reported",
    [
        (
            [
                {
                    "prompt_tokens_details": {
                        "cache_creation_tokens": 10,
                        "cache_write_tokens": 10,
                        "cached_tokens": 0,
                    }
                },
                {
                    "prompt_tokens_details": {
                        "cache_write_tokens": 5,
                        "cached_tokens": 2,
                    }
                },
            ],
            15,
            2,
            2,
        ),
        (
            [
                {
                    "prompt_tokens_details": {
                        "cache_creation_tokens": 0,
                        "cache_write_tokens": 0,
                        "cached_tokens": 0,
                    }
                },
                None,
            ],
            0,
            0,
            1,
        ),
        ([{"input_tokens": 30}, None], 0, 0, 0),
    ],
)
def test_offline_reports_disclose_cache_usage_without_double_counting(
    tmp_path, monkeypatch, usage, write_total, read_total, reported
):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "benchmark"
    save(dataset, [row(0), row(1)])
    run(dataset, output, [])
    entries, _ = read_entries(output)
    for name in PROFILES:
        own = [entry for entry in entries if entry["backend"] == name]
        for entry, value in zip(own, usage):
            entry["result"]["usage"] = value
    (output / "verdicts.jsonl").write_text(
        "".join(dumps(entry) + "\n" for entry in entries)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting cannot infer")

    monkeypatch.setattr("litellm.acompletion", forbidden)
    monkeypatch.setattr("typesafe_sdk.AsyncTypeSafeClient.system_one", forbidden)
    report = write_report(dataset, output)
    assert report == json.loads((output / "report.json").read_text())
    text = (output / "report.md").read_text()
    assert "Creation/write aliases count once" in text
    for name in PROFILES:
        cache = report["backends"][name]["prompt_cache"]
        for key, known in (("write_tokens", write_total), ("read_tokens", read_total)):
            assert cache[key] == {
                "known_total_tokens": known,
                "total_tokens": known if reported == 2 else None,
                "reported_assessments": reported,
                "unreported_assessments": 2 - reported,
            }
        assert (
            "Prompt cache (tokens): `" + json.dumps(cache, sort_keys=True) + "`" in text
        )


def test_preparation_spend_needs_matching_companions_and_all_attempts(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    rows = [row(0), row(1)]
    rows[0]["annotation"]["estimated_cost_usd"] = 9.0
    save(dataset, rows)
    assert not preparation_costs(dataset, rows, rows[:1])["available"]
    signature = digest(
        [
            {key: r[key] for key in ("id", "source", "request", "extraction_error")}
            for r in rows
        ]
    )
    (tmp_path / "run.json").write_text(
        dumps(
            {
                "evidence_sha256": signature,
                "attempt_totals": {
                    "annotation": {"attempts": 3},
                    "model_review": {"attempts": 1},
                },
                "stages": {
                    "annotation": {"ids": [r["id"] for r in rows]},
                    "model_review": {"ids": [rows[0]["id"]]},
                },
            }
        )
    )
    entries = [
        {
            "id": rows[0]["id"],
            "stage": "annotation",
            "result": {"estimated_cost_usd": 0.01},
        },
        {
            "id": rows[0]["id"],
            "stage": "annotation",
            "result": {"estimated_cost_usd": 0.02},
        },
        {
            "id": rows[1]["id"],
            "stage": "annotation",
            "result": {"estimated_cost_usd": 0.03},
        },
        {
            "id": rows[0]["id"],
            "stage": "model_review",
            "result": {"estimated_cost_usd": None},
        },
    ]
    (tmp_path / "attempts.jsonl").write_text("".join(dumps(e) + "\n" for e in entries))
    value = preparation_costs(dataset, rows, rows[:1])
    assert value["full_dataset"]["annotation"]["estimated_cost_usd"] == 0.06
    assert value["selected_subset"]["annotation"]["estimated_cost_usd"] == 0.03
    assert value["full_dataset"]["model_review"]["unpriced_attempts"] == 1
    (tmp_path / "attempts.jsonl").write_text(
        "".join(dumps(e) + "\n" for e in entries[1:])
    )
    assert not preparation_costs(dataset, rows, rows[:1])["available"]
    (tmp_path / "attempts.jsonl").write_text("".join(dumps(e) + "\n" for e in entries))
    rows[0]["request"]["policy"] = "edited evidence"
    assert not preparation_costs(dataset, rows, rows[:1])["available"]


def test_progress_is_emitted_after_save_and_counts_resumed_outcomes(tmp_path):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "benchmark"
    save(dataset, [row(0, extraction=True), row(1), row(2), row(3)])
    messages = []

    def emit(text):
        message = json.loads(text)
        messages.append(message)
        if "progress" in message:
            # A reported result must already be present in the flushed log.
            assert (output / "verdicts.jsonl").read_bytes().endswith(b"\n")
            entries, incomplete = read_entries(output)
            assert not incomplete and entries[-1]["backend"] == message["progress"]
            own = [e for e in entries if e["backend"] == message["progress"]]
            assert message["completed_assessments"] == len(own)
            assert message["errors"] == sum(
                e["result"]["error"] is not None for e in own
            )
            assert message["total_assessments"] == 3
            assert message["pending"] == 3 - len(own)

    def execute(factory):
        return asyncio.run(
            run_benchmark(
                dataset, output, PROFILES, QUESTION, factory=factory, emit=emit
            )
        )

    with pytest.raises(KeyboardInterrupt):
        execute(fake_factory([], failures={("jev", 2): "timeout"}, interrupt_at=5))
    assert [m["completed_assessments"] for m in messages if "progress" in m] == [
        1,
        1,
        2,
        2,
    ]
    messages.clear()
    calls = []
    summary = execute(fake_factory(calls))
    assert len(calls) == 2
    assert messages[0]["planned_calls"] == {"jev": 1, "bedrock": 1}
    assert messages[0]["completed_assessments"] == {"jev": 2, "bedrock": 2}
    assert messages[0]["errors"] == {"jev": 1, "bedrock": 0}
    assert [m["completed_assessments"] for m in messages if "progress" in m] == [3, 3]
    assert summary["pending"] == 0 and summary["errors"] == 1
    before = {
        p: p.read_bytes()
        for p in (output / "benchmark.json", output / "verdicts.jsonl")
    }
    messages.clear()
    execute(fake_factory(calls))
    assert len(calls) == 2 and not any("progress" in message for message in messages)
    assert messages[0]["completed_assessments"] == {"jev": 3, "bedrock": 3}
    assert all(p.read_bytes() == data for p, data in before.items())
