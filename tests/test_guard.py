import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_core import request

from jev_contracts import LABELS, GuardError, GuardResult
from jev_guard import (
    BedrockGuard,
    JevGuard,
    load_question,
    load_settings,
    model_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTION = load_question(ROOT / "src/jev_guard/resources/guard-v1.json")
PROFILES = load_settings(
    ROOT / "src/jev_guard/resources/guards.toml", ["jev", "bedrock"]
)


def result(label="authorized", error=None, backend="bedrock", **kwargs):
    return GuardResult(
        label=None if error else label,
        error=GuardError(kind=error, diagnostic="Safe failure") if error else None,
        latency_seconds=1.0,
        model=model_metadata(PROFILES[backend]),
        **kwargs,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"error": {"kind": "timeout", "diagnostic": "timeout"}},
        {"label": None},
        {"latency_seconds": float("nan")},
        {"estimated_cost_usd": -1.0},
        {"probabilities": dict.fromkeys(LABELS, 0.2), "confidence": 0.1},
        {
            "probabilities": {
                "authorized": float("nan"),
                "unauthorized": 0.0,
                "insufficient_evidence": 0.0,
            },
            "confidence": 1.0,
        },
        {"confidence": 0.8},
        {"probabilities": {"authorized": 1.0}, "confidence": 1.0},
    ],
)
def test_result_rejects_invalid_outcomes(changes):
    with pytest.raises(ValidationError):
        GuardResult.model_validate(result().model_dump() | changes)


def test_guards_receive_identical_complete_evidence_and_semantics(monkeypatch):
    seen, constructors = {}, []
    probability = {
        "authorized": 0.4,
        "unauthorized": 0.35,
        "insufficient_evidence": 0.25,
    }

    class Client:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            assert kwargs["retry"].max_retries == 0
            assert kwargs["retry"].timeout is None
            assert kwargs["timeout"] == PROFILES["jev"].timeout

        async def system_one(self, **kwargs):
            seen["jev"] = kwargs
            answer = SimpleNamespace(
                choice="authorized", probabilities=probability, confidence=0.01
            )
            return SimpleNamespace(
                model="returned-jev",
                usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 20}),
                answers={"authorization": answer},
                choices={"authorization": answer},
            )

        async def aclose(self):
            pass

    async def completion(**kwargs):
        seen["bedrock"] = kwargs
        return SimpleNamespace(
            model_dump=lambda **kwargs: {
                "model": "returned-luna",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"label":"authorized"}'},
                    }
                ],
            }
        )

    monkeypatch.setattr("typesafe_sdk.AsyncTypeSafeClient", Client)
    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr(
        "litellm.completion_cost",
        lambda **kwargs: pytest.fail("Guards must not estimate cost"),
    )
    jev, bedrock = (
        JevGuard(PROFILES["jev"], QUESTION),
        BedrockGuard(PROFILES["bedrock"], QUESTION),
    )

    async def run():
        a, b = await jev.assess(request()), await bedrock.assess(request())
        await jev.assess(request())
        await jev.aclose()
        return a, b

    a, b = asyncio.run(run())
    assert len(constructors) == 1
    assert (
        seen["jev"]["state"]
        == json.loads(seen["bedrock"]["messages"][1]["content"])
        == request().model_dump(mode="json")
    )
    question = seen["jev"]["questions"]["authorization"]
    assert question.instructions == QUESTION["instructions"]
    assert dict(question.criteria) == QUESTION["criteria"]
    assert json.loads(seen["bedrock"]["messages"][0]["content"]) == QUESTION
    assert len(seen["jev"]["questions"]) == 1
    assert seen["bedrock"]["reasoning_effort"] == "low"
    assert seen["bedrock"]["max_retries"] == seen["bedrock"]["num_retries"] == 0
    schema = seen["bedrock"]["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"label"}
    assert schema["additionalProperties"] is False
    assert (
        a.probabilities == probability
        and a.confidence == 0.01
        and a.label == "authorized"
    )
    assert a.model.returned == "returned-jev"
    assert a.estimated_cost_usd is None
    assert b.probabilities is None and b.confidence is None
    assert b.model.returned == "returned-luna" and b.estimated_cost_usd is None


@pytest.mark.parametrize(
    "content,finish,refusal,kind",
    [
        (
            '{"label":"unauthorized","rationale":"extra"}',
            "stop",
            None,
            "invalid_response",
        ),
        ('{"label":"unknown"}', "stop", None, "invalid_response"),
        ('{"label":"authorized"}', "length", None, "incomplete_or_refused_response"),
        ('{"label":"authorized"}', "stop", "refusal", "incomplete_or_refused_response"),
    ],
)
def test_invalid_llm_output_retains_usage_without_estimating_cost(
    monkeypatch, content, finish, refusal, kind
):
    async def completion(**kwargs):
        return SimpleNamespace(
            model_dump=lambda **kwargs: {
                "usage": {"prompt_tokens": 10},
                "choices": [
                    {
                        "finish_reason": finish,
                        "message": {"content": content, "refusal": refusal},
                    }
                ],
            }
        )

    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr(
        "litellm.completion_cost",
        lambda **kwargs: pytest.fail("Guards must not estimate cost"),
    )
    value = asyncio.run(BedrockGuard(PROFILES["bedrock"], QUESTION).assess(request()))
    assert value.label is None and value.error.kind == kind
    assert value.usage == {"prompt_tokens": 10} and value.estimated_cost_usd is None


def test_timing_includes_construction_and_output_validation(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("jev_guard.time.perf_counter", lambda: clock[0])
    original_dump = type(request()).model_dump_json
    original_validate = GuardResult.model_validate

    def dump(self, **kwargs):
        clock[0] += 2.0
        return original_dump(self, **kwargs)

    def validate(value):
        clock[0] += 3.0
        return original_validate(value)

    async def completion(**kwargs):
        clock[0] += 5.0
        return SimpleNamespace(
            model_dump=lambda **kwargs: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"label":"authorized"}'},
                    }
                ]
            }
        )

    monkeypatch.setattr(type(request()), "model_dump_json", dump)
    monkeypatch.setattr(GuardResult, "model_validate", validate)
    monkeypatch.setattr("litellm.acompletion", completion)
    monkeypatch.setattr("litellm.completion_cost", lambda **kwargs: None)
    value = asyncio.run(BedrockGuard(PROFILES["bedrock"], QUESTION).assess(request()))
    assert value.latency_seconds == 10.0


@pytest.mark.parametrize(
    "status,kind",
    [
        (429, "rate_limit"),
        (401, "authentication"),
        (400, "configuration"),
        (500, "provider_error"),
    ],
)
def test_provider_failure_is_one_call_and_safe(monkeypatch, status, kind):
    calls = []

    class Failure(Exception):
        status_code = status
        usage = {"prompt_tokens": 20, "secret": "SECRET"}
        response = SimpleNamespace(
            headers={"retry-after": "30", "authorization": "SECRET"}
        )

    async def completion(**kwargs):
        calls.append(kwargs)
        raise Failure("SECRET body")

    monkeypatch.setattr("litellm.acompletion", completion)
    value = asyncio.run(BedrockGuard(PROFILES["bedrock"], QUESTION).assess(request()))
    assert len(calls) == 1 and value.error.kind == kind and value.label is None
    assert "SECRET" not in value.model_dump_json()
    assert value.error.reset_headers == {"retry-after": "30"}
    assert value.usage == {"prompt_tokens": 20}


@pytest.mark.parametrize(
    "native",
    [
        {
            "choice": "other",
            "probabilities": dict.fromkeys(LABELS, 1 / 3),
            "confidence": 0.0,
        },
        {
            "choice": "authorized",
            "probabilities": {"authorized": 1.0},
            "confidence": 1.0,
        },
        {
            "choice": "authorized",
            "probabilities": dict.fromkeys(LABELS, 1 / 3),
            "confidence": float("nan"),
        },
    ],
)
def test_jev_invalid_native_output_is_an_error_with_usage(monkeypatch, native):
    class Client:
        def __init__(self, **kwargs):
            pass

        async def system_one(self, **kwargs):
            answer = SimpleNamespace(**native)
            return SimpleNamespace(
                model="jev-1.13.0",
                usage=SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 20}),
                answers={"authorization": answer},
                choices={"authorization": answer},
            )

    monkeypatch.setattr("typesafe_sdk.AsyncTypeSafeClient", Client)
    value = asyncio.run(JevGuard(PROFILES["jev"], QUESTION).assess(request()))
    assert value.label is None and value.error.kind == "invalid_response"
    assert value.usage == {"input_tokens": 20} and value.probabilities is None


def test_jev_sdk_validation_error_retains_only_usage(monkeypatch):
    import httpx2
    from typesafe_sdk import TypeSafeAPIResponseValidationError

    class Client:
        def __init__(self, **kwargs):
            pass

        async def system_one(self, **kwargs):
            raise TypeSafeAPIResponseValidationError(
                200,
                {"usage": {"input_tokens": 100, "SECRET": "secret"}, "body": "SECRET"},
                httpx2.Headers(),
                "answers.authorization",
            )

    monkeypatch.setattr("typesafe_sdk.AsyncTypeSafeClient", Client)
    value = asyncio.run(JevGuard(PROFILES["jev"], QUESTION).assess(request()))
    assert value.error.kind == "invalid_response" and value.usage == {
        "input_tokens": 100
    }
    assert "SECRET" not in value.model_dump_json()
