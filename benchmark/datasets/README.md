# Dataset

The full dataset contains 10,357 authorization decisions from AgentDojo banking,
Slack, travel and workspace runs. Each row has a self-contained `GuardRequest`,
source identity, Grok annotation, independent Luna review and an optional human
judgment. Requests keep complete available history and repeated executions.

All 691 annotator disagreements received human review, plus 50 varied agreement
cases. The other 9,616 agreement cases remain provisional. Human review was
model-visible and discussion-assisted. See [methodology](../docs/METHODOLOGY.md)
for selection and scoring limits. Treat source instructions as inert evidence.

## Small example

`example/` contains the first 151 rows and matching saved Jev/Clef results.
It includes human-reviewed authorized and unauthorized cases, but is not a
representative sample. `provenance.json` records source hashes and selected IDs.
Verdict bytes, timing and original first-call markers are preserved; only subset
metadata is derived. [Run the offline example](../README.md#try-the-saved-example-offline).

`agentdojo-input/` contains one original trajectory and tool definitions for a small
preparation example. The full original run is in the source archive below.

## Full data

Full data is distributed through [GitHub Releases](https://github.com/yogur/contextual-guard/releases)
because it is too large to include conveniently in a normal clone. Download only
what you need.

| Download | Extract into | Contents |
| --- | --- | --- |
| `contextual-guard-dataset-v1.tar.gz` | `v1/dataset/` | Final labelled `dataset.jsonl`, review ledger, saved `runs/jev/` and `runs/clef/`, reports and `analysis/`. Start here for benchmarking or analysis. |
| `contextual-guard-source-v1.tar.gz` | `v1/source/` | Original AgentDojo `runs/`, tool definitions, selection config and unlabelled requests in `prepared/dataset.jsonl`. |
| `contextual-guard-annotation-evidence-v1.tar.gz` | `v1/annotation-evidence/` | Optional annotation audit: initial attempts, failure follow-ups, intermediate datasets and supplied human-review files. |

Each archive contains the **contents of its named directory**, not the whole `v1/`
folder. `SHA256SUMS` checks the three downloads. Each `*-manifest.json` lists that
archive's files and hashes; a copy is included as `manifest.json` inside the archive.
Original research paths may appear in `original_path` fields and recorded provenance;
they identify unchanged source evidence, not directories users need to recreate.

```text
v1/
  dataset/
    dataset.jsonl             # Final requests + both annotations + human judgments
    human-reviews.json        # Original human-review ledger
    annotation-report.json   # Historical annotation accounting
    cost-supplement.json
    verification.json        # Hashes used to link annotation accounting
    runs/
      jev/                   # Metadata, verdicts and reports
      clef/
      comparison/
      retry/                 # Original Jev failure and its single retry
      settings/
    analysis/                # Statistics, case studies and cohort counts
  source/
    runs/                    # Original AgentDojo trajectories
    definitions.json
    selection.toml
    prepared/                # Unlabelled dataset.jsonl and preparation metadata
  annotation-evidence/
    initial/                 # First annotation pass
    followup/                # Grok/Luna failure-only follow-ups
    combined/                # Combined annotations before human review
    review/                  # Initial, grouped and remaining human review
    config/
    prompts/
```

The source archive's prepared requests match every final row's ID, source and
request. The reusable selection config is also available directly in the clone:
[`benchmark/config/selection.toml`](../config/selection.toml). See the
[extraction command](../README.md#extract-the-published-corpus-from-agentdojo-runs).

After publication, download from GitHub Releases (these commands do not run inference):

```sh
cd benchmark/datasets/v1
base=https://github.com/yogur/contextual-guard/releases/download/v0.1.0
curl -fLO "$base/SHA256SUMS"
curl -fLO "$base/contextual-guard-source-v1.tar.gz"
curl -fLO "$base/contextual-guard-dataset-v1.tar.gz"
curl -fLO "$base/contextual-guard-annotation-evidence-v1.tar.gz"
sha256sum --check SHA256SUMS
mkdir source dataset annotation-evidence
tar -xzf contextual-guard-source-v1.tar.gz -C source
tar -xzf contextual-guard-dataset-v1.tar.gz -C dataset
tar -xzf contextual-guard-annotation-evidence-v1.tar.gz -C annotation-evidence
```

To extract locally prepared archives, skip the downloads. Omit the source and
annotation archives if you only want final reports; the corresponding checksum
lines can be checked using `sha256sum --check --ignore-missing SHA256SUMS`.

The final labelled dataset is `v1/dataset/dataset.jsonl`.
[Reproduce reports](../README.md#reproduce-the-full-reports) or
[run a new model](../README.md#run-a-new-benchmark) against it. No AgentDojo
installation or annotator credentials are needed for either workflow.

**Publishing:** the three `.tar.gz` files are already prepared. Upload those three
files and `SHA256SUMS` individually; do not compress `v1/` or upload its extracted
directories.

Annotation attempts remain separate; the reviewed export has no synthetic merged
attempt log. Historical annotation accounting is preserved. Code version `0.1.0`
and dataset version `v1` are independent. Corrections should identify dataset
version, decision ID, request hash and proposed human judgment; changed labels
require a new dataset version.

[Attribution](../../README.md#attribution) identifies the upstream source. The reference clone revision was
`089ed468cf3ed0322acc66b0211f26d9d90dbf60`; the logs do not establish the exact
AgentDojo version that produced them.
