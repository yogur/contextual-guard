# Contextual Guard

Contextual authorization for agent actions, powered by system one (decision) models.

This repository contains two parts:

- **The guard:** an experimental Python library that assesses whether a proposed
  agent action is authorized in context, with a built-in authorization prompt and
  support for hosted Jev and local Clef-Flash.
- **The [benchmark](benchmark/README.md):** the dataset, tooling and results from
  evaluating those models on AgentDojo traces, including annotations, human review
  and scripts to reproduce the published reports.

Give the guard your application's policy, visible conversation history, tool definitions
and a proposed action. It returns `authorized`, `unauthorized` or
`insufficient_evidence`. Your application decides whether to execute, block or ask
for clarification. Evaluate the guard on your own traces before relying on its judgments.

## Use the guard

Python 3.11 or later (below 3.15). Install from PyPI once published:

```sh
pip install contextual-guard
```

For local installation before publication, run `uv build`, then
`pip install dist/contextual_guard-0.1.0-py3-none-any.whl`.
The Python imports are `jev_guard` and `jev_contracts`.

```python
from importlib.resources import files
from jev_guard import load_question, load_settings, make_guard

resources = files("jev_guard") / "resources"
guard = make_guard(
    load_settings(resources / "guards.toml", ["jev"])["jev"],
    load_question(resources / "guard-v1.json"),
)
# Inside your async application, with a complete GuardRequest:
try:
    result = await guard.assess(request)
    # Handle result.error, then inspect result.label before executing the action.
finally:
    await guard.aclose()
```

[Guard usage](docs/USAGE.md) explains the request contract, provider setup and prompt
configuration. [examples/assess.py](examples/assess.py) constructs a complete request;
running it without arguments prints that request without making a model call.
Guard use requires neither AgentDojo nor the benchmark dataset.

## What the evaluation found

Both Jev and Clef completed 10,357 static replay decisions. Jev performed better
on cases where the two annotators agreed. On 691 human-adjudicated disagreements,
Jev explicitly allowed 130/171 unauthorized actions; Clef allowed 112/171, while
interrupting more authorized actions. Neither tested configuration is established
as a reliable sole authorization gate.

The [visual interpretation](benchmark/docs/results/benchmark-results-interpretation.html)
separates annotator agreement from human-adjudicated disagreement and unsafe allows
from escalation. See [results](benchmark/docs/RESULTS.md) and
[methodology](benchmark/docs/METHODOLOGY.md) for denominators and limitations.
This was static replay, not a measurement of live attack prevention.

## Repository

- `src/`: installable guard and request/result contracts.
- `examples/`: a minimal integration example.
- `benchmark/`: original and prepared data, labels, saved results, helper scripts and documentation.
- `docs/`: guard usage instructions.
- `tests/`: guard and tooling tests.

From a clone, `uv sync --locked` installs the guard, benchmark tooling and development
dependencies. Only the guard is published to PyPI. Full data is distributed as optional
GitHub Release downloads; a small example is included in Git.
[Benchmark instructions](benchmark/README.md) cover offline reproduction and new runs.

## Attribution

The benchmark derives from
[AgentDojo](https://github.com/ethz-spylab/agentdojo), by Edoardo Debenedetti,
Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer and Florian Tramèr
([paper](https://arxiv.org/abs/2406.13352)); mainly task/tool text and run data from a single AgentDojo run.


The recorded actor from the AgentDojo run is GPT-4o; annotations are from Grok and Luna; guard results are
from Jev and Clef-Flash. Human review was owner-supplied.
