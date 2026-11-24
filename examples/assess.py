"""Construct a request offline; --backend makes one real assessment when requested."""

import argparse
import asyncio
from importlib.resources import files

from jev_contracts import Candidate, GuardRequest, Message
from jev_guard import load_question, load_settings, make_guard


def request():
    resources = files("jev_guard") / "resources"
    return GuardRequest(
        policy=(resources / "authorization-policy-v1.txt").read_text(),
        messages=[
            Message(
                id="m000000",
                role="user",
                origin="user",
                content="Read meeting 42's details.",
            )
        ],
        tool_definitions="read_event(id: int): read the event with this ID; no writes.",
        candidate=Candidate(name="read_event", arguments={"id": 42}),
        context_complete=True,
    )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["jev", "clef", "bedrock"])
    args = parser.parse_args()
    value = request()
    if not args.backend:
        print(value.model_dump_json(indent=2))
        return
    resources = files("jev_guard") / "resources"
    settings = load_settings(resources / "guards.toml", [args.backend])[args.backend]
    guard = make_guard(settings, load_question(resources / "guard-v1.json"))
    try:
        result = await guard.assess(value)
        print(result.model_dump_json(indent=2, exclude_none=True))
        # The host decides whether to execute, block or ask for clarification.
    finally:
        await guard.aclose()


if __name__ == "__main__":
    asyncio.run(main())
