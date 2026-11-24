# Benchmark tools

Run commands from the repository root after `uv sync --locked`. The tooling is
installed from this clone and depends on the guard; it is not a separate PyPI release.
Dataset persistence uses POSIX file locking; Linux is the tested platform.

## Prepare AgentDojo traces

```sh
uv run python -m jev_dataset prepare /path/to/actor-runs \
  --definitions /path/to/definitions.json \
  --selection benchmark/config/selection.toml --output data/prepared
uv run python benchmark/scripts/render_dataset_review.py data/prepared/dataset.jsonl
```

Input is AgentDojo TraceLogger JSON plus matching tool definitions. The adapter
retains full pre-call history, repeated executions and original source identity;
it does not execute logged tools. Unsupported records produce visible errors,
including `request: null` where a proposed action cannot be transformed.
Other trace formats require your own conversion to `GuardRequest`.

Definitions JSON maps suite names to complete tool descriptions, either directly
or under a `suites` key. The example uses
[selection.toml](../config/selection.toml), the published corpus selection. It selects
complete clean and attack trajectories, yielding 10,357 requests from our supplied
run. Omit `--selection` to process all supported trajectories, or edit a copy for
your corpus; it is not a request-count limit. For ad hoc inputs, `--task`, `--attack` and `--limit`
are available. `--help` gives the exact options.

## Annotate and inspect your own data

Annotation **makes model calls**. It is not needed to use the published labels:

```sh
uv run python -m jev_dataset annotate data/prepared \
  --models benchmark/src/jev_dataset/resources/models.toml \
  --review --retries 1 --concurrency 5
```

The bundled annotators use Bedrock with normal AWS credentials. Configure profiles
for your account before running. `--review` runs the independent second annotator;
`--review-limit` bounds that stage. `--retries 1` allows up to two dispatches per
row per annotator. Repeat an identical command to resume. Input, profile or rubric
changes require new output. Annotation attempts and original judgments are retained.

Review helpers are offline and never write human labels into the dataset:

| Script under `benchmark/scripts/` | Purpose |
| --- | --- |
| `render_dataset_review.py DATASET` | Full requests and judgments as standalone HTML. Optional `--output FILE`. |
| `group_disagreements.py DATASET --output DIR` | Group unreviewed disagreements by exact request or task/call; link each case to full history. |
| `render_remaining_review.py DATASET --reviews FILE --output DIR` | Display remaining cases against supplied human judgments. |

Review text starts with `decision,verdict,optional_reasoning`. Decision numbers are
one-based dataset rows. The first two commas delimit fields; rationales may contain
commas. `unchanged` keeps a prior judgment; `modified:LABEL` marks a replacement.
The display helper overlays supplied labels without enforcing replacement markers;
the historical reconstruction recipe below enforces explicit overrides. A rendered
page is not an imported review. The existing `human_review` field keeps human
judgments separate from both annotators. No general-purpose review-import CLI is provided.

## Benchmark and compare

Configure a provider using [guard usage](../../docs/USAGE.md). A new run **makes
model calls**; report and compare are offline:

```sh
uv run python -m jev_benchmark run data/prepared/dataset.jsonl \
  --backend jev --output data/my-jev
uv run python -m jev_benchmark report data/prepared/dataset.jsonl --output data/my-jev
uv run python -m jev_benchmark compare data/prepared/dataset.jsonl \
  --run data/my-jev --run data/my-clef --output data/comparison
```

Use a separate output directory per experiment. Repeat an identical run command
to resume pending rows; saved outcomes are not replayed. An interrupted in-flight
call may repeat. Assessments disable automatic retries. A completed resume makes
zero calls and needs no runtime service. `run` never annotates.

Comparisons require identical dataset bytes, selected row IDs/order and prompt
hashes. Incompatible runs fail rather than silently intersecting rows. Errors,
pending rows and insufficient-evidence judgments retain their denominators.
Returned native choices are scored as saved; probabilities do not replace them.

## Reproduce the published evidence

These scripts are **specific to dataset v1** and use the public archive layout.
Original source names remain inside recorded provenance fields.
They are not needed for new datasets or ordinary guard use.

| Script under `benchmark/scripts/` | Inputs and output |
| --- | --- |
| `reproduce_reports.py --root DATASET_ROOT --output NEW_DIR` | Saved dataset/runs → final reports, retaining the single Jev retry disclosure. |
| `analyze.py --root DATASET_ROOT --output DIR` | Saved dataset/runs → statistics, diagnostic examples and cited cases. |
| `build_interpretation.py --root DATASET_ROOT --output DIR` | Hash-checked inputs → self-contained HTML and cohort matrices. |
| `derive_example.py --root DATASET_ROOT --output NEW_DIR` | First 151 rows and matching verdicts → the checked-in example. |
| `reproduce_reviews.py --root ANNOTATION_ROOT --output NEW_DIR` | Original review files → the published labels and a reconstruction ledger. |
| `verify_evidence.py --repo REPO --artifacts ASSETS --work NEW_DIR` | Installed packages and three archives → checksum, preparation, report, review and example checks with networking disabled. |

See [reproduction commands](../README.md#reproduce-the-full-reports).
The review recipe reproduces dataset bytes exactly, including labels and rationales.
It checks every annotation-evidence file against its manifest and records the current
script/manifest identity. The original review ledger remains with the final dataset.
`additional.json` is preserved historical analysis; `analyze.py` regenerates
`statistics.json`, `examples.json` and `case-studies.json`.
