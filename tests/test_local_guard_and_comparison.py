import asyncio
import json
from pathlib import Path

import httpx2
import pytest
from test_benchmark import fake_factory, row, run, save
from test_core import request
from test_guard import PROFILES, QUESTION

from jev_benchmark import dumps, read_entries, run_benchmark
from jev_benchmark.report import write_comparison, write_report
from jev_guard import JevGuard, load_settings, operational_error, system_one_input_bytes

ROOT = Path(__file__).resolve().parents[1]
CLEF = load_settings(ROOT / "src/jev_guard/resources/guards.toml", ["clef"])["clef"]


def test_sdk_wire_payload_and_client_auth_are_isolated(monkeypatch):
    from typesafe_sdk import AsyncTypeSafeClient

    monkeypatch.setenv("TYPESAFE_API_KEY", "hosted-secret")
    monkeypatch.setattr("jev_guard.inspect_ollama", lambda settings: {})
    seen = []

    def respond(req):
        seen.append((req.headers["authorization"], json.loads(req.content)))
        return httpx2.Response(
            200,
            json={
                "model": json.loads(req.content)["model"],
                "answers": {
                    "authorization": {
                        "type": "choice",
                        "choice": "authorized",
                        "probabilities": {
                            "authorized": 0.8,
                            "unauthorized": 0.1,
                            "insufficient_evidence": 0.1,
                        },
                        "confidence": 0.6,
                    }
                },
                "usage": {"input_tokens": 123, "output_tokens": 1},
            },
        )

    def client(**kwargs):
        return AsyncTypeSafeClient(**kwargs, transport=httpx2.MockTransport(respond))

    monkeypatch.setattr("typesafe_sdk.AsyncTypeSafeClient", client)

    async def assess():
        values = []
        for settings in (PROFILES["jev"], CLEF):
            guard = JevGuard(settings, QUESTION)
            values.append(await guard.assess(request()))
            await guard.aclose()
        return values

    values = asyncio.run(assess())
    assert all(value.label == "authorized" for value in values)
    assert [auth for auth, _ in seen] == ["Bearer hosted-secret", "Bearer ollama"]
    hosted, local = [body for _, body in seen]
    assert hosted["state"] == local["state"] == request().model_dump(mode="json")
    assert hosted["questions"] == local["questions"]
    assert list(local["questions"]["authorization"]["criteria"]) == list(
        QUESTION["criteria"]
    )
    assert local["keep_alive"] == "5m" and "keep_alive" not in hosted
    assert all(value.estimated_cost_usd is None for value in values)
    assert all("secret" not in value.model_dump_json() for value in values)
    assert system_one_input_bytes(request(), QUESTION) == len(
        json.dumps(
            {"state": local["state"], "questions": local["questions"]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_digest", "changed"),
        ("runtime_version", "other"),
        ("num_ctx", 8192),
        ("keep_alive", "1m"),
    ],
)
def test_local_resume_pins_and_zero_call_completed_resume(
    tmp_path, monkeypatch, field, value
):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "run"
    save(dataset, [row(0)])
    monkeypatch.setattr("jev_benchmark.inspect_ollama", lambda settings: {})
    monkeypatch.setattr("jev_guard.inspect_ollama", lambda settings: {})
    calls = []

    class Guard:
        def __init__(self, settings, question):
            self.guard = JevGuard(settings, question)

        async def assess(self, request):
            from jev_contracts import GuardError, GuardResult

            calls.append(request)
            return GuardResult(
                error=GuardError(kind="timeout", diagnostic="Saved timeout"),
                latency_seconds=1,
                model=self.guard.model,
            )

        async def aclose(self):
            pass

    def execute(settings=CLEF):
        return asyncio.run(
            run_benchmark(
                dataset,
                output,
                {"clef": settings},
                QUESTION,
                factory=Guard,
                emit=lambda text: None,
            )
        )

    execute()
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    monkeypatch.setattr(
        "jev_benchmark.inspect_ollama",
        lambda settings: pytest.fail("Completed resume must not inspect transport"),
    )
    execute()
    assert len(calls) == 1 and original == {
        p.name: p.read_bytes() for p in output.iterdir()
    }
    with pytest.raises(ValueError, match="changed"):
        execute(CLEF.model_copy(update={field: value}))


def test_local_oversize_blocks_bulk_before_inference(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    rows = [row(0), row(1)]
    rows[1]["request"]["tool_definitions"] = "X" * 65536
    save(dataset, rows)
    monkeypatch.setattr("jev_benchmark.inspect_ollama", lambda settings: {})
    with pytest.raises(ValueError, match="1/2 inputs exceed"):
        asyncio.run(
            run_benchmark(
                dataset,
                tmp_path / "run",
                {"clef": CLEF},
                QUESTION,
                factory=lambda *args: pytest.fail("Must not dispatch"),
            )
        )


def test_documented_limit_errors_are_safe():
    class Failure(Exception):
        status_code = 400
        body = {"error": "input exceeds context window SECRET"}

    error = operational_error(Failure())
    assert error.kind == "context_overflow" and "SECRET" not in error.model_dump_json()


def test_compare_partial_repeated_profiles_and_historical_unknown_offline(
    tmp_path, monkeypatch
):
    dataset, a, b = tmp_path / "dataset.jsonl", tmp_path / "a", tmp_path / "b"
    rows = [row(0, human="authorized"), row(1, human="unauthorized")]
    rows[1]["model_review"]["output"]["label"] = "authorized"
    save(dataset, rows)
    calls = []
    run(dataset, a, calls)
    calls_b = []
    run(
        dataset,
        b,
        calls_b,
        factory=fake_factory(calls_b, failures={("jev", 1): "authentication"}),
    )
    meta = json.loads((a / "benchmark.json").read_text())
    meta.pop("execution")
    meta["selection"]["limit"] = 2  # Effective selection matches None.
    (a / "benchmark.json").write_text(dumps(meta))
    before = {p: p.read_bytes() for directory in (a, b) for p in directory.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("Offline comparison cannot use transport")

    monkeypatch.setattr("litellm.acompletion", forbidden)
    monkeypatch.setattr("typesafe_sdk.AsyncTypeSafeClient", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    report = write_comparison(dataset, [a, b], tmp_path / "comparison")
    assert not report["complete"] and report["counts"]["pending"] == 1
    assert report["backend_order"] == [
        "run1/jev",
        "run1/bedrock",
        "run2/jev",
        "run2/bedrock",
    ]
    assert report["backends"]["run1/jev"]["execution"] is None
    cohorts = report["backends"]["run2/jev"]["reviewed_cohorts"]
    assert (
        cohorts["annotator_disagreements"]["N"]
        == cohorts["sampled_agreements"]["N"]
        == 1
    )
    assert cohorts["sampled_agreements"]["pending_by_reference"]["unauthorized"] == 1
    assert all(p.read_bytes() == data for p, data in before.items())
    assert report == json.loads((tmp_path / "comparison/comparison.json").read_text())
    assert len(report["sources"]) == 2
    assert write_report(dataset, a)["complete"]


@pytest.mark.parametrize(
    "corruption",
    ["dataset", "selection", "question", "duplicate", "model", "request", "malformed"],
)
def test_compare_rejects_incompatible_or_corrupt_runs(tmp_path, corruption):
    dataset, a, b = tmp_path / "dataset.jsonl", tmp_path / "a", tmp_path / "b"
    save(dataset, [row(0), row(1)])
    run(dataset, a, [])
    run(dataset, b, [], limit=1 if corruption == "selection" else None)
    meta_path = b / "benchmark.json"
    meta = json.loads(meta_path.read_text())
    if corruption == "dataset":
        meta["dataset_sha256"] = "0" * 64
    elif corruption == "question":
        meta["criteria_sha256"] = "0" * 64
    elif corruption == "malformed":
        meta["backend_order"].append("jev")
    meta_path.write_text(dumps(meta))
    entries, _ = read_entries(b)
    if corruption == "duplicate":
        entries.append(entries[0])
    elif corruption == "model":
        entries[0]["result"]["model"]["requested"] = "other"
    elif corruption == "request":
        entries[0]["request_sha256"] = "0" * 64
    (b / "verdicts.jsonl").write_text("".join(dumps(entry) + "\n" for entry in entries))
    with pytest.raises(ValueError):
        write_comparison(dataset, [a, b], tmp_path / "comparison")
    with pytest.raises(ValueError, match="distinct"):
        write_comparison(dataset, [a, a], tmp_path / "duplicate")


@pytest.mark.parametrize(
    "change", ["digest", "runtime", "context", "loaded_context", "cloud"]
)
def test_inventory_rejects_changed_local_runtime_without_auth(monkeypatch, change):
    import io

    from jev_guard import inspect_ollama

    model = {"name": CLEF.model, "digest": CLEF.model_digest}
    data = {
        "version": {"version": CLEF.runtime_version},
        "tags": {"models": [model]},
        "show": {"parameters": f"num_ctx {CLEF.num_ctx}", "capabilities": ["decision"]},
        "ps": {"models": []},
    }
    if change == "digest":
        model["digest"] = "other"
    elif change == "runtime":
        data["version"]["version"] = "other"
    elif change == "context":
        data["show"]["parameters"] = f"num_ctx {CLEF.num_ctx // 2}"
    elif change == "loaded_context":
        data["ps"]["models"] = [
            {"digest": CLEF.model_digest, "context_length": CLEF.num_ctx // 2}
        ]
    else:
        model["remote_model"] = "clef-flash:cloud"

    def open_request(req, **kwargs):
        assert not req.has_header("Authorization")
        return io.BytesIO(json.dumps(data[req.full_url.rsplit("/", 1)[1]]).encode())

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    with pytest.raises(ValueError):
        inspect_ollama(CLEF)


def test_exported_annotation_costs_linked_once_without_inventing_attempts(tmp_path):
    import hashlib

    from jev_benchmark.report import preparation_costs

    dataset = tmp_path / "dataset.jsonl"
    rows = [row(0)]
    save(dataset, rows)
    evidence = {
        "annotation-report.json": {
            "models": {
                "annotation": {
                    "sdk_cost": {"estimated_cost_usd": None, "unpriced_attempts": 2},
                    "dispatches": 2,
                }
            }
        },
        "cost-supplement.json": {"separately_calculated_flex_estimated_usd": 1.2},
    }
    for name, value in evidence.items():
        (tmp_path / name).write_text(dumps(value))
    hashes = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ["dataset.jsonl", *evidence]
    }
    (tmp_path / "verification.json").write_text(dumps({"artifact_sha256": hashes}))
    costs = preparation_costs(dataset, rows, rows)
    assert not costs["available"] and costs["full_dataset"] is None
    assert set(costs["linked_evidence"]) == set(evidence)
    assert not (tmp_path / "attempts.jsonl").exists()
    (tmp_path / "cost-supplement.json").write_text("{}")
    assert (
        "cost-supplement.json"
        not in preparation_costs(dataset, rows, rows)["linked_evidence"]
    )


def test_new_verdicts_omit_cost_fields_and_historical_estimates_remain_readable(
    tmp_path,
):
    dataset, output = tmp_path / "dataset.jsonl", tmp_path / "new"
    save(dataset, [row(0)])
    calls = []
    base = fake_factory(calls)

    class CostedGuard(base):
        async def assess(self, value):
            return (await super().assess(value)).model_copy(
                update={
                    "estimated_cost_usd": 123.0,
                    "cost_source": "Old adapter estimate",
                }
            )

    run(dataset, output, calls, factory=CostedGuard)
    entries, _ = read_entries(output)
    for entry in entries:
        assert "estimated_cost_usd" not in entry["result"]
        assert "cost_source" not in entry["result"]
    report = write_report(dataset, output)
    assert all("cost" not in backend for backend in report["backends"].values())
    # Preserve support for recorded historical values without making projections.
    entries[0]["result"].update(estimated_cost_usd=0.01, cost_source="Historical SDK")
    (output / "verdicts.jsonl").write_text(
        "".join(dumps(entry) + "\n" for entry in entries)
    )
    historical = write_report(dataset, output)
    assert (
        historical["backends"][entries[0]["backend"]]["cost"]["estimated_cost_usd"]
        == 0.01
    )
    assert "projected_selected_calls_usd" not in json.dumps(historical)
