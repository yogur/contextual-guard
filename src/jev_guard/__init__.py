"""One-call guards; no dataset or annotation dependencies."""

import importlib.metadata
import json
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from jev_contracts import (
    LABELS,
    GuardError,
    GuardModel,
    GuardOutput,
    GuardRequest,
    GuardResult,
    Nonempty,
    StrictModel,
)


class GuardSettings(StrictModel):
    provider: Literal["typesafe", "typesafe_ollama", "bedrock_mantle"]
    model: Nonempty
    api_base: Nonempty
    timeout: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    region: str | None = None
    reasoning: str | None = None
    aws_profile: str | None = None
    model_digest: Nonempty | None = None
    runtime_version: Nonempty | None = None
    num_ctx: Annotated[int, Field(gt=0)] | None = None
    keep_alive: Nonempty | None = None

    @model_validator(mode="after")
    def local_settings(self):
        local = (self.model_digest, self.runtime_version, self.num_ctx, self.keep_alive)
        if self.provider == "typesafe_ollama":
            if not all(local) or any((self.region, self.reasoning, self.aws_profile)):
                raise ValueError(
                    "Ollama requires pinned digest/runtime/context/keep_alive"
                )
            if ":cloud" in self.model:
                raise ValueError("Local guard cannot use an Ollama cloud model")
        elif any(value is not None for value in local):
            raise ValueError("Local runtime settings require typesafe_ollama")
        return self


def load_settings(path: Path, names: list[str]):
    if not names or len(set(names)) != len(names):
        raise ValueError("Select distinct backend profiles in execution order")
    profiles = tomllib.loads(path.read_text())
    return {name: GuardSettings.model_validate(profiles[name]) for name in names}


def load_question(path: Path):
    question = json.loads(path.read_text())
    if (
        set(question) != {"instructions", "criteria"}
        or not isinstance(question["instructions"], str)
        or not question["instructions"]
        or set(question["criteria"]) != set(LABELS)
        or not all(isinstance(v, str) and v for v in question["criteria"].values())
    ):
        raise ValueError("Expected shared instructions and three label definitions")
    return question


def model_metadata(settings: GuardSettings):
    sdk = "litellm" if settings.provider == "bedrock_mantle" else "typesafe-sdk"
    effective = settings.model_dump(mode="json")
    # Keep historical hosted/Bedrock metadata and resume identities readable.
    if settings.provider != "typesafe_ollama":
        for key in ("model_digest", "runtime_version", "num_ctx", "keep_alive"):
            effective.pop(key)
    return GuardModel(
        requested=settings.model,
        provider=settings.provider,
        endpoint=settings.api_base,
        settings=effective | {"max_retries": 0},
        sdk_version=importlib.metadata.version(sdk),
    )


def operational_error(exc, *, configuration=False):
    """Never persist exception bodies, which can contain auth or HTTP internals."""
    status = getattr(exc, "status_code", getattr(exc, "status", None))
    name = type(exc).__name__
    body = getattr(exc, "body", None)
    # Classify documented limit errors without saving provider bodies or secrets.
    limit_error = status == 400 and any(
        marker in str(body).lower()
        for marker in (
            "context length",
            "context window",
            "exceeds context",
            "input too long",
        )
    )
    if name == "TypeSafeAPIResponseValidationError":
        kind = "invalid_response"
    elif status == 429 or name == "RateLimitError":
        kind = "rate_limit"
    elif status in {401, 403} or name in {
        "AuthenticationError",
        "PermissionDeniedError",
        "NoCredentialsError",
    }:
        kind = "authentication"
    elif limit_error or status == 413 or name in {"ContextWindowExceededError"}:
        kind = "context_overflow"
    elif (
        configuration
        or status in {400, 404, 422}
        or name
        in {
            "NotFoundError",
            "UnsupportedParamsError",
        }
    ):
        kind = "configuration"
    elif "Timeout" in name:
        kind = "timeout"
    else:
        kind = "provider_error"
    headers = getattr(exc, "headers", None)
    if headers is None:
        headers = getattr(getattr(exc, "response", None), "headers", {})
    reset = {
        key: str(headers[key])[:200]
        for key in (
            "retry-after",
            "x-ratelimit-reset",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        )
        if headers and key in headers
    }
    return GuardError(
        kind=kind, diagnostic=name, status_code=status, reset_headers=reset
    )


def finish(started, values):
    # Validate the entire result before ending the measured assessment boundary.
    result = GuardResult.model_validate(values | {"latency_seconds": 0.0})
    return result.model_copy(update={"latency_seconds": time.perf_counter() - started})


def error_usage(exc):
    usage = getattr(exc, "usage", None)
    if usage is None and isinstance(getattr(exc, "body", None), dict):
        usage = exc.body.get("usage")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    if not isinstance(usage, dict):
        return None
    return {
        key: value
        for key, value in usage.items()
        if key
        in {
            "input_tokens",
            "output_tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }
        and type(value) is int
        and value >= 0
    } or None


def system_one_payload(request: GuardRequest, question: dict):
    from typesafe_sdk import Choice

    return {
        "state": request.model_dump(mode="json"),
        "questions": {"authorization": Choice(**question)},
    }


def system_one_input_bytes(request: GuardRequest, question: dict):
    """SDK-normalized UTF-8 state/schema, including the JSON envelope."""
    from pydantic_core import to_json
    from typesafe_sdk._core.questions import normalize_questions

    payload = system_one_payload(request, question)
    payload["questions"] = normalize_questions(payload["questions"])
    return len(to_json(payload))


def inspect_ollama(settings: GuardSettings):
    """Read-only inventory; no hosted credentials, loading or inference."""

    def read(route, body=None):
        req = urllib.request.Request(
            settings.api_base.rstrip("/") + "/api/" + route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=settings.timeout) as response:
            return json.load(response)

    version = read("version")["version"]
    model = next(
        (m for m in read("tags")["models"] if m["name"] == settings.model), None
    )
    if model is None or model["digest"] != settings.model_digest:
        raise ValueError("Ollama model tag/digest changed or missing")
    show = read("show", {"model": settings.model})
    parameters = dict(line.split() for line in show.get("parameters", "").splitlines())
    if version != settings.runtime_version or parameters.get("num_ctx") != str(
        settings.num_ctx
    ):
        raise ValueError("Ollama runtime/context changed")
    if "decision" not in show.get("capabilities", []) or model.get("remote_model"):
        raise ValueError("Expected a local decision model")
    residency = read("ps")
    if any(
        loaded.get("digest") == settings.model_digest
        and loaded.get("context_length") != settings.num_ctx
        for loaded in residency.get("models", [])
    ):
        raise ValueError("Loaded Ollama context differs from pinned num_ctx")
    return {
        "version": version,
        "model": model,
        "parameters": parameters,
        "residency_before": residency,
    }


class JevGuard:
    def __init__(self, settings: GuardSettings, question: dict):
        if settings.provider not in {"typesafe", "typesafe_ollama"}:
            raise ValueError("JevGuard requires typesafe settings")
        if any((settings.region, settings.reasoning, settings.aws_profile)):
            raise ValueError("Hosted Jev does not support Bedrock settings")
        self.settings, self.question = settings, question
        self.model = model_metadata(settings)
        self.client = None

    async def assess(self, request: GuardRequest) -> GuardResult:
        started = time.perf_counter()
        values = {"model": self.model}
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        try:
            if self.client is None:
                if self.settings.provider == "typesafe_ollama":
                    inspect_ollama(self.settings)
                self.client = AsyncTypeSafeClient(
                    **(
                        {"api_key": "ollama"}
                        if self.settings.provider == "typesafe_ollama"
                        else {}
                    ),
                    model=self.settings.model,
                    base_url=self.settings.api_base,
                    timeout=self.settings.timeout,
                    retry=RetryPolicy(max_retries=0, timeout=None),
                )
        except Exception as exc:
            return finish(
                started, values | {"error": operational_error(exc, configuration=True)}
            )
        try:
            if (
                self.settings.provider == "typesafe_ollama"
                and system_one_input_bytes(request, self.question) > 65536
            ):
                return finish(
                    started,
                    values
                    | {
                        "error": GuardError(
                            kind="context_overflow",
                            diagnostic="State/schema exceeds local 64 KiB limit",
                        )
                    },
                )
            response = await self.client.system_one(
                **system_one_payload(request, self.question),
                **(
                    {"extra_body": {"keep_alive": self.settings.keep_alive}}
                    if self.settings.provider == "typesafe_ollama"
                    else {}
                ),
            )
        except Exception as exc:
            return finish(
                started,
                values | {"error": operational_error(exc), "usage": error_usage(exc)},
            )
        try:
            values["model"] = GuardModel.model_validate(
                self.model.model_dump(mode="json") | {"returned": response.model}
            )
            values["usage"] = response.usage.model_dump(mode="json")
            if set(response.answers) != {"authorization"}:
                raise ValueError("Unexpected answers")
            answer = response.choices["authorization"]
            return finish(
                started,
                values
                | {
                    "label": answer.choice,
                    "probabilities": answer.probabilities,
                    "confidence": answer.confidence,
                },
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            return finish(
                started,
                values
                | {
                    "error": GuardError(
                        kind="invalid_response",
                        diagnostic="Invalid Jev Choice response",
                    )
                },
            )

    async def aclose(self):
        if self.client is not None:
            await self.client.aclose()


class BedrockGuard:
    def __init__(self, settings: GuardSettings, question: dict):
        if settings.provider != "bedrock_mantle":
            raise ValueError("BedrockGuard requires bedrock_mantle settings")
        self.settings, self.question = settings, question
        self.model = model_metadata(settings)

    async def assess(self, request: GuardRequest) -> GuardResult:
        started = time.perf_counter()
        import litellm

        values = {"model": self.model}
        route = f"{self.settings.provider}/{self.settings.model}"
        parameters = {
            "model": route,
            "messages": [
                {
                    "role": "system",
                    "content": json.dumps(self.question, ensure_ascii=False),
                },
                {"role": "user", "content": request.model_dump_json()},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "guard_verdict",
                    "strict": True,
                    "schema": GuardOutput.model_json_schema(),
                },
            },
            "api_base": self.settings.api_base,
            "timeout": self.settings.timeout,
            "max_retries": 0,
            "num_retries": 0,
            "drop_params": False,
        }
        if self.settings.region is not None:
            parameters["aws_region_name"] = self.settings.region
        if self.settings.reasoning is not None:
            parameters["reasoning_effort"] = self.settings.reasoning
            parameters["allowed_openai_params"] = ["reasoning_effort"]
        if self.settings.aws_profile is not None:
            parameters["aws_profile_name"] = self.settings.aws_profile
        try:
            response = await litellm.acompletion(**parameters)
        except Exception as exc:
            return finish(
                started,
                values | {"error": operational_error(exc), "usage": error_usage(exc)},
            )
        try:
            raw = response.model_dump(mode="json")
            values["model"] = GuardModel.model_validate(
                self.model.model_dump(mode="json") | {"returned": raw.get("model")}
            )
            values["usage"] = raw.get("usage")
            choice = raw["choices"][0]
            message = choice["message"]
            if choice.get("finish_reason") != "stop" or message.get("refusal"):
                return finish(
                    started,
                    values
                    | {
                        "error": GuardError(
                            kind="incomplete_or_refused_response",
                            diagnostic="Response did not finish with a verdict",
                        )
                    },
                )
            label = GuardOutput.model_validate_json(message["content"]).label
            return finish(started, values | {"label": label})
        except (ValueError, TypeError, KeyError, IndexError):
            return finish(
                started,
                values
                | {
                    "error": GuardError(
                        kind="invalid_response",
                        diagnostic="Invalid structured label response",
                    )
                },
            )

    async def aclose(self):
        # LiteLLM owns and pools its asynchronous HTTP transport.
        pass


def make_guard(settings: GuardSettings, question: dict):
    return (BedrockGuard if settings.provider == "bedrock_mantle" else JevGuard)(
        settings, question
    )
