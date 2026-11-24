"""Provider-independent request contract. No dataset or inference dependencies."""

import math
import re
from typing import Annotated, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def valid_json(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        for item in value:
            valid_json(item)
        return value
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            valid_json(item)
        return value
    raise ValueError("Expected finite ordinary JSON values and string object keys")


Nonempty = Annotated[str, Field(min_length=1)]
Label = Literal["authorized", "unauthorized", "insufficient_evidence"]
LABELS = ("authorized", "unauthorized", "insufficient_evidence")
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Candidate(StrictModel):
    name: Nonempty
    arguments: dict[str, JsonValue]

    @field_validator("arguments", mode="before")
    @classmethod
    def arguments_are_json(cls, value):
        if type(value) is not dict:
            raise ValueError("Arguments must be a JSON object")
        return valid_json(value)


class HistoricalCall(Candidate):
    id: Nonempty


class Message(StrictModel):
    id: str
    role: Literal["system", "user", "assistant", "tool"]
    origin: Literal["application", "user", "agent", "external", "harness", "unknown"]
    content: str
    tool_calls: list[HistoricalCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    error: str | None = None


class GuardRequest(StrictModel):
    schema_version: Literal["guard-request.v1"] = "guard-request.v1"
    policy: Nonempty
    messages: list[Message]
    tool_definitions: Nonempty
    candidate: Candidate
    context_complete: bool

    @model_validator(mode="after")
    def local_ids(self):
        calls = []
        seen = set()
        for index, message in enumerate(self.messages):
            if message.id != f"m{index:06d}":
                raise ValueError("Message IDs must be sequential request-local IDs")
            for call in message.tool_calls:
                if call.id != f"c{len(calls):06d}":
                    raise ValueError("Call IDs must be sequential request-local IDs")
                calls.append(call.id)
                seen.add(call.id)
            if message.tool_call_id is not None and message.tool_call_id not in seen:
                raise ValueError("Tool result must reference a prior proposal")
        return self


class EvidenceRef(StrictModel):
    target: Nonempty
    pointer: str


class AnnotationOutput(StrictModel):
    label: Label
    rationale: Annotated[str, Field(min_length=1, max_length=1200)]
    evidence: Annotated[list[EvidenceRef], Field(min_length=1, max_length=8)]


class GuardOutput(StrictModel):
    label: Label


class GuardError(StrictModel):
    kind: Nonempty
    diagnostic: Nonempty
    status_code: int | None = None
    reset_headers: dict[str, str] = Field(default_factory=dict)


class GuardModel(StrictModel):
    requested: Nonempty
    returned: str | None = None
    provider: Nonempty
    endpoint: Nonempty
    settings: dict[str, JsonValue]
    sdk_version: Nonempty


class GuardResult(StrictModel):
    label: Label | None = None
    error: GuardError | None = None
    latency_seconds: Nonnegative
    model: GuardModel
    usage: dict[str, JsonValue] | None = None
    estimated_cost_usd: Nonnegative | None = None
    cost_source: str | None = None
    probabilities: dict[Label, Probability] | None = None
    confidence: Probability | None = None

    @model_validator(mode="after")
    def outcome(self):
        if (self.label is None) == (self.error is None):
            raise ValueError("Result requires either label or operational error")
        if self.probabilities is not None:
            if set(self.probabilities) != set(LABELS) or not math.isclose(
                sum(self.probabilities.values()), 1, abs_tol=0.001
            ):
                raise ValueError("Expected the three-label probability distribution")
        if (self.probabilities is None) != (self.confidence is None):
            raise ValueError("Native probabilities and confidence must appear together")
        if self.error is not None and self.probabilities is not None:
            raise ValueError("Operational error cannot have a policy distribution")
        if self.estimated_cost_usd is not None and not self.cost_source:
            raise ValueError("Cost estimate requires provenance")
        if self.usage is not None:
            valid_json(self.usage)
        return self


class Guard(Protocol):
    async def assess(self, request: GuardRequest) -> GuardResult: ...


def validate_evidence(output: AnnotationOutput, request: GuardRequest):
    targets = {
        "policy": request.policy,
        "tool_definitions": request.tool_definitions,
        "candidate": request.candidate.model_dump(mode="json"),
    }
    targets.update({m.id: m.model_dump(mode="json") for m in request.messages})
    for ref in output.evidence:
        if ref.target not in targets:
            raise ValueError(f"Unknown evidence target: {ref.target}")
        value = targets[ref.target]
        if not ref.pointer:
            continue
        if not ref.pointer.startswith("/") or re.search(r"~(?![01])", ref.pointer):
            raise ValueError("Invalid RFC 6901 pointer")
        for token in ref.pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict) and token in value:
                value = value[token]
            elif (
                isinstance(value, list)
                and re.fullmatch(r"0|[1-9][0-9]*", token)
                and int(token) < len(value)
            ):
                value = value[int(token)]
            else:
                raise ValueError(f"Evidence pointer does not resolve: {ref.pointer}")
    return output
