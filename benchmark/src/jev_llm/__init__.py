"""Thin LiteLLM annotation client for Bedrock Runtime/Mantle and Ollama chat."""

import importlib.metadata
import json
import math
import time
import tomllib
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import litellm
from litellm.integrations.custom_logger import CustomLogger

from jev_contracts import AnnotationOutput, GuardRequest, validate_evidence


def load_profile(path: str, name: str) -> dict:
    """Read model defaults from TOML; no credentials or model IDs live in code."""
    profile = tomllib.loads(Path(path).read_text())[name]
    provider = profile["provider"]
    common = {"provider", "model", "api_base", "timeout", "max_tokens"}
    specific = {
        "bedrock": {
            "region",
            "reasoning",
            "aws_profile",
            "service_tier",
        },
        "bedrock_mantle": {"region", "reasoning", "aws_profile"},
        "ollama_chat": {"num_ctx", "think"},
    }
    if provider not in specific or set(profile) - common - specific[provider]:
        raise ValueError(f"Unsupported provider/settings in {name}")
    settings = {k: v for k, v in profile.items() if k != "provider"}
    settings["model"] = provider + "/" + profile["model"]
    settings.setdefault("timeout", 120)
    timeout = settings["timeout"]
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    for key in ("num_ctx", "max_tokens"):
        if key in settings and (type(settings[key]) is not int or settings[key] <= 0):
            raise ValueError(f"{key} must be a positive integer")
    if provider == "ollama_chat":
        if not {"api_base", "num_ctx", "think", "max_tokens"} <= settings.keys():
            raise ValueError(
                "Ollama requires endpoint, context, thinking and token limit"
            )
        if type(settings["think"]) is not bool:
            raise ValueError("think must be boolean")
        if settings["max_tokens"] >= settings["num_ctx"]:
            raise ValueError("max_tokens must leave room for the prompt")
    elif provider == "bedrock":
        settings.setdefault("region", "us-east-1")
        settings.setdefault(
            "api_base", "https://bedrock-runtime.{region}.amazonaws.com/openai/v1"
        )
        if settings.get("service_tier") not in {"default", "flex", "priority"}:
            raise ValueError(
                "Runtime requires an explicit default, flex or priority tier"
            )
        if settings.get("max_tokens", 0) < 16:
            raise ValueError("Runtime requires max_tokens >= 16 (SDK minimum)")
        if not isinstance(settings.get("reasoning"), str) or not settings["reasoning"]:
            raise ValueError("Runtime requires explicit reasoning effort")
        settings["api_base"] = settings["api_base"].format(region=settings["region"])
    else:
        settings["api_base"] = settings["api_base"].format(region=settings["region"])
    return settings


def response_format():
    # Bedrock supports a JSON-schema subset. Pydantic enforces these bounds locally.
    def portable(value):
        if isinstance(value, dict):
            return {
                k: portable(v)
                for k, v in value.items()
                if k not in {"minLength", "maxLength", "maxItems"}
            }
        if isinstance(value, list):
            return [portable(v) for v in value]
        return value

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "annotation",
            "strict": True,
            "schema": portable(AnnotationOutput.model_json_schema()),
        },
    }


async def annotate(
    request: GuardRequest,
    rubric: str,
    *,
    model: str,
    region: str | None = None,
    reasoning: str | None = None,
    service_tier: str | None = None,
    api_base: str | None = None,
    aws_profile: str | None = None,
    num_ctx: int | None = None,
    think: bool | None = None,
    max_tokens: int | None = None,
    timeout: float = 120,
) -> dict:
    """Make one call, retain usage even for invalid output, and never execute tools.

    The pipeline owns persistence and retries. No fallback or SDK-internal probing.
    Model names use LiteLLM routes. Endpoint and authentication remain SDK-owned.
    """
    parameters = {
        "model": model,
        "messages": [
            {"role": "system", "content": rubric},
            {"role": "user", "content": request.model_dump_json()},
        ],
        "response_format": response_format(),
        "timeout": timeout,
        "max_retries": 0,
        "num_retries": 0,
        "drop_params": False,
    }
    runtime = model.startswith("bedrock/")
    native, wire = {}, {}
    runtime_wire = None
    if runtime:
        if service_tier not in {"default", "flex", "priority"}:
            raise ValueError("Runtime requires an explicit service tier")
        if type(max_tokens) is not int or max_tokens < 16:
            raise ValueError("Runtime requires max_tokens >= 16 (SDK minimum)")
        # The SDK's bundled catalog predates these model IDs. Declare only
        # documented API capabilities to select native Runtime Responses.
        # Preserve existing SDK prices; add no rates.
        litellm.register_model(
            {
                model: {
                    "litellm_provider": "bedrock",
                    "mode": "responses",
                    "supported_endpoints": ["/v1/responses"],
                    "supports_reasoning": True,
                    "supports_response_schema": True,
                }
            },
            persist_across_reloads=False,
        )
        if not isinstance(reasoning, str) or not reasoning:
            raise ValueError("Runtime requires explicit reasoning effort")
        parameters["service_tier"] = service_tier
    elif service_tier is not None:
        raise ValueError("Tier/native reasoning settings belong only to Runtime")
    if region is not None:
        parameters["aws_region_name"] = region
    if reasoning is not None:
        parameters["reasoning_effort"] = reasoning
        if not runtime:
            parameters["allowed_openai_params"] = ["reasoning_effort"]
    if api_base is not None:
        parameters["api_base"] = api_base
    if aws_profile is not None:
        parameters["aws_profile_name"] = aws_profile
    local = model.startswith("ollama_chat/")
    if runtime:
        call_id = str(uuid4())
        parameters["litellm_call_id"] = call_id

        class RuntimeWire(CustomLogger):
            def log_pre_api_call(self, model, messages, kwargs):
                if kwargs.get("litellm_call_id") != call_id:
                    return
                body = (kwargs.get("additional_args") or {}).get("complete_input_dict")
                if isinstance(body, str):
                    body = json.loads(body)
                if isinstance(body, dict):
                    wire.update(
                        {
                            k: deepcopy(body[k])
                            for k in (
                                "model",
                                "service_tier",
                                "max_output_tokens",
                                "reasoning",
                                "text",
                            )
                            if k in body
                        }
                    )

        runtime_wire = RuntimeWire()
        parameters["callbacks"] = [runtime_wire]
    if local:
        if region is not None or reasoning is not None or aws_profile is not None:
            raise ValueError("AWS/reasoning settings belong only to Bedrock")
        if (
            type(num_ctx) is not int
            or type(max_tokens) is not int
            or not 0 < max_tokens < num_ctx
            or type(think) is not bool
        ):
            raise ValueError(
                "Ollama requires finite context/generation and boolean think"
            )

        def capture_native(details):
            # LiteLLM's normalized usage can fall back to a different tokenizer.
            # Read only the native response, never auth or HTTP internals.
            if details.get("log_event_type") == "post_api_call":
                value = details.get("original_response")
                if isinstance(value, str):
                    value = json.loads(value)
                if isinstance(value, dict):
                    native.update(value)

        parameters.update(
            num_ctx=num_ctx,
            # extra_body merges at the native API's top level, not into options.
            extra_body={"think": think, "truncate": False, "shift": False},
            logger_fn=capture_native,
        )
    elif num_ctx is not None or think is not None:
        raise ValueError("Context/thinking settings belong only to Ollama")
    if max_tokens is not None:
        parameters["max_tokens"] = max_tokens
    result = {
        "model": model,
        "region": region,
        "reasoning": reasoning,
        "api_base": api_base,
        "effective_settings": {
            "model": model,
            "region": region,
            "reasoning": reasoning,
            **(
                {
                    "service_tier": service_tier,
                    "api": "Responses",
                }
                if runtime
                else {}
            ),
            "api_base": api_base,
            "aws_profile": aws_profile,
            "num_ctx": num_ctx,
            "think": think,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_retries": 0,
            "num_retries": 0,
            **({"truncate": False, "shift": False} if local else {}),
        },
        "litellm_version": importlib.metadata.version("litellm"),
        "output": None,
        "raw_output": None,
        "error": None,
        "usage": None,
        "estimated_cost_usd": None,
        "cost_source": (
            "local Ollama: no provider API charge; hardware/energy cost unmeasured"
            if local
            else "litellm.completion_cost; estimate, not billed spend"
        ),
        **({"provider_api_charge_usd": 0} if local else {}),
        **(
            {
                "requested_service_tier": service_tier,
                "returned_service_tier": None,
                "returned_model": None,
                "wire_settings": wire,
                "price_provenance": None,
            }
            if runtime
            else {}
        ),
    }
    started = time.perf_counter()
    try:
        if runtime:
            parameters["input"] = parameters.pop("messages")
            parameters["text"] = {
                "format": {
                    "type": "json_schema",
                    **parameters.pop("response_format")["json_schema"],
                }
            }
            parameters["max_output_tokens"] = parameters.pop("max_tokens")
            parameters["reasoning"] = {"effort": parameters.pop("reasoning_effort")}
            response = await litellm.aresponses(**parameters)
        else:
            response = await litellm.acompletion(**parameters)
    except Exception as exc:
        # Exception bodies can contain HTTP internals or credentials.
        result["error"] = {
            "kind": type(exc).__name__,
            "status_code": getattr(exc, "status_code", None),
        }
        # Retain only known usage and reset fields, never response bodies or auth.
        usage = getattr(exc, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump(mode="json")
        if isinstance(usage, dict):
            result["usage"] = {
                key: usage[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if type(usage.get(key)) is int and usage[key] >= 0
            } or None
        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        reset = {
            key: str(headers[key])[:200]
            for key in (
                "retry-after",
                "x-ratelimit-reset",
                "x-ratelimit-reset-requests",
                "x-ratelimit-reset-tokens",
            )
            if key in headers
        }
        if reset:
            result["error"]["reset_headers"] = reset
        result["latency_seconds"] = time.perf_counter() - started
        return result
    finally:
        if runtime_wire is not None:
            # LiteLLM promotes per-call callbacks into global lists. Capture only
            # this call, detach its recorded body, and retire its callback.
            result["wire_settings"] = deepcopy(wire)
            litellm.logging_callback_manager.remove_callback_from_all_lists(
                runtime_wire
            )
            if runtime_wire in litellm.input_callback:
                litellm.input_callback.remove(runtime_wire)
    result["latency_seconds"] = time.perf_counter() - started
    raw = response.model_dump(mode="json")
    result["usage"] = raw.get("usage")
    if runtime:
        result["returned_model"] = raw.get("model")
        result["returned_service_tier"] = raw.get("service_tier")
        result["native_usage"] = raw.get("usage")
        usage = raw.get("usage") or {}
        result["usage"] = {
            **usage,
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
        } or None
    if local:
        result["native_metrics"] = {
            key: native[key]
            for key in (
                "model",
                "done",
                "done_reason",
                "prompt_eval_count",
                "prompt_eval_cached_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            )
            if key in native
        }
    else:
        try:
            cost_model = model
            if runtime:
                info = litellm.get_model_info(
                    model=cost_model, custom_llm_provider="bedrock"
                )
                result["price_provenance"] = {
                    k: v for k, v in info.items() if "cost" in k
                }
                if service_tier != "default" and any(
                    info.get(f"{direction}_cost_per_token_{service_tier}") is None
                    for direction in ("input", "output")
                ):
                    raise ValueError("SDK has no explicit requested-tier prices")
            cost = litellm.completion_cost(
                completion_response=response,
                model=cost_model,
                **({"service_tier": service_tier} if runtime else {}),
            )
            if isinstance(cost, (float, int)) and math.isfinite(cost) and cost >= 0:
                result["estimated_cost_usd"] = cost
        except Exception:
            pass  # Unknown price stays null; never fabricate a zero-dollar estimate.
    choice = next(iter(raw.get("choices") or []), {})
    message = choice.get("message") or {}
    result["raw_output"] = message.get("content")
    if runtime:
        content = [
            part
            for item in raw.get("output") or []
            if item.get("type") == "message"
            for part in item.get("content") or []
        ]
        result["raw_output"] = "".join(
            part.get("text", "")
            for part in content
            if part.get("type") == "output_text"
        )
        complete = raw.get("status") == "completed" and not any(
            part.get("type") == "refusal" for part in content
        )
    else:
        complete = choice.get("finish_reason") == "stop" and not message.get("refusal")
    prompt_tokens = native.get("prompt_eval_count")
    generated = native.get("eval_count")
    if local and (
        type(prompt_tokens) is not int
        or prompt_tokens <= 0
        or type(generated) is not int
        or generated < 0
        or native.get("done") is not True
    ):
        result["error"] = {"kind": "context_accounting_unavailable"}
    elif local and prompt_tokens + max_tokens > num_ctx:
        result["error"] = {"kind": "ContextWindowExceededError"}
    elif local and generated > max_tokens:
        result["error"] = {"kind": "generation_limit_exceeded"}
    elif not complete:
        result["error"] = {"kind": "incomplete_or_refused_response"}
    else:
        try:
            output = AnnotationOutput.model_validate(json.loads(result["raw_output"]))
            validate_evidence(output, request)
            result["output"] = output.model_dump(mode="json")
        except (ValueError, TypeError):
            result["error"] = {"kind": "invalid_response"}
    return result
