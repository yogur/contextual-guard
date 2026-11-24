# Dataset and evaluation

The evaluation compares two system one (decision) models—hosted Jev and local
Clef-Flash—on 10,357 proposed actions
from AgentDojo runs. Each request retains the full available pre-call history.
Grok and Luna annotated independently; human review resolved all 691 disagreements
and checked 50 selected agreements.

Start with the [visual interpretation](docs/results/benchmark-results-interpretation.html),
[results](docs/RESULTS.md), [methodology](docs/METHODOLOGY.md), or
[detailed case analysis](docs/RESULTS_ANALYSIS.md).

Full datasets are distributed through [GitHub Releases](https://github.com/yogur/contextual-guard/releases)
because of their size. The repository includes a small example and archive
checksums. See [download instructions](datasets/README.md#full-data) for release `v0.1.0`.

## Layout

- `config/selection.toml`: trajectory selection used for the published corpus.
- `src/`: AgentDojo preparation, annotation and benchmark modules.
- `scripts/`: review helpers and offline reproduction scripts.
- `datasets/example/`: 151 real rows and matching saved Jev/Clef verdicts.
- `datasets/agentdojo-input/`: one original trajectory and tool definitions for a small extraction example.
- `datasets/v1/`: checksums and manifests for the three full-data release downloads.
- `docs/`: methodology, tooling instructions and results.

The [dataset guide](datasets/README.md) describes the full source, prepared and
labelled data. The [tooling guide](docs/TOOLING.md) covers every retained script,
including which commands make model calls.

## Try the saved example offline

From the repository root:

```sh
uv sync --locked
work=$(mktemp -d)
cp -R benchmark/datasets/example "$work/example"
uv run python -m jev_benchmark report "$work/example/dataset.jsonl" --output "$work/example/jev"
uv run python -m jev_benchmark report "$work/example/dataset.jsonl" --output "$work/example/clef"
uv run python -m jev_benchmark compare "$work/example/dataset.jsonl" \
  --run "$work/example/jev" --run "$work/example/clef" --output "$work/comparison"
```

No credentials or inference are needed. The example demonstrates the workflow;
it is a prefix, not a representative sample.

## Extract the published corpus from AgentDojo runs

The [selection config](config/selection.toml) includes clean user-task trajectories
and recorded attacks using four named methods. Use it with the source download:

```sh
uv run python -m jev_dataset prepare benchmark/datasets/v1/source/runs \
  --definitions benchmark/datasets/v1/source/definitions.json \
  --selection benchmark/config/selection.toml \
  --output data/prepared
```

For this source run, that selects 2,613 trajectories and yields 10,357 requests.
It keeps full trajectories; it does not sample or cap the number of requests.
Other source runs can yield different counts. Without `--selection`, preparation
processes all supported supplied trajectories. This command is offline; annotation
is a separate, optional workflow in the [tooling guide](docs/TOOLING.md).

## Reproduce the full reports

Download and extract the [dataset archive](datasets/README.md#full-data) first.
Then, from the repository root:

```sh
uv run python benchmark/scripts/reproduce_reports.py \
  --root benchmark/datasets/v1/dataset --output data/reports
uv run python benchmark/scripts/analyze.py \
  --root benchmark/datasets/v1/dataset --output data/analysis
uv run python benchmark/scripts/build_interpretation.py \
  --root benchmark/datasets/v1/dataset --output data/interpretation
```

These commands use saved judgments and make no inference calls. The report recipe
preserves the disclosure of Jev's single HTTP 504 retry. The analysis recipes are
specific to dataset v1; they are not general benchmark APIs.

## Run a new benchmark

This command **makes model calls** and may take hours. Run it yourself with the
provider configured as described in [guard usage](../docs/USAGE.md):

```sh
uv run python -m jev_benchmark run \
  benchmark/datasets/v1/dataset/dataset.jsonl \
  --backend jev --output data/my-jev
```

Use `--backend clef` for your configured local model, and a separate output directory.
Repeat an identical command to resume; compare saved runs with
`python -m jev_benchmark compare`. Annotation is a separate step and is never
triggered by a benchmark command.
