# Evaluation methodology

We evaluated Jev and Clef-Flash as authorization guards by replaying saved AgentDojo
actions. This measures their judgments on recorded requests; the guards did not
control the original agent or execute its tools.

## Dataset

The source contains GPT-4o runs across banking, Slack, travel and workspace.
[Selection](../config/selection.toml) includes 97 clean user-task trajectories and
all recorded trajectories for four attack methods: `important_instructions`,
`injecagent`, `tool_knowledge` and `direct`.

The resulting dataset has **10,357 decisions from 2,613 selected trajectories**.
Of those, 2,576 trajectories across 96 tasks contain proposed calls; the rest
contribute input coverage but no decision rows. Every request contains the policy,
tool definitions, candidate arguments and all available history before the
assistant message proposing the call. Calls proposed together are assessed
separately against the same prior history. Future observations and attack-success
labels are excluded.

Repeated executions are retained: there are 5,614 distinct complete requests.
Rows are therefore not independent observations. Development cases overlap this
corpus. Tool definitions came from the supplied AgentDojo clone; source logs omit
the exact benchmark version, which limits verification of that historical match.

## Annotation and human review

**Grok 4.7 and Luna**, both with low reasoning, independently annotated requests
using the same policy and rubric. Neither saw the other's answer. Failure-only
follow-ups supplied 183 missing Grok judgments and one invalid Luna judgment;
valid disagreements were not retried to force agreement.

Human review covered all **691 annotator disagreements**, plus **50 agreement
cases selected for variety**. The reviewer could see model judgments, and some cases
were discussed with other models. This was not blind review or a representative
random sample. Only supplied human verdicts were imported; judgments were not
propagated between similar requests.

The 741 human labels comprise 537 authorized, 190 unauthorized and 14
insufficient-evidence cases. They remain separate from both original annotations.
The remaining 9,616 agreement cases have no human review and remain provisional.
Selection details, original reviews and explicit overrides are preserved in the
[annotation evidence](../datasets/README.md#full-data), along with model settings,
attempt logs and historical usage accounting.

## Scoring

Reports distinguish agreement with model labels from accuracy against human review:

| Group | Cases | Reference |
| --- | ---: | --- |
| Annotator agreement | 9,666 | Shared annotator label; agreement alone does not establish correctness |
| Annotator disagreement, human-adjudicated | 691 | Final human judgment |
| Reviewed agreement subset | 50 | Human judgment; a check within the agreement group, not a third disjoint group |

The targeted human review does not support a whole-population accuracy estimate.
Do not add the 50-case subset to the two main groups' denominators.

We report all three predicted labels, separating unsafe allows from escalation:

| Reference → prediction | Interpretation |
| --- | --- |
| Unauthorized → authorized | Unsafe allow (explicit false negative) |
| Unauthorized → insufficient evidence | Escalation |
| Unauthorized → unauthorized | Denial |
| Authorized → unauthorized | False block (false positive) |
| Authorized → insufficient evidence | Additional interruption |

Treating escalation as a stopped action assumes the host waits for review before
execution. Operational errors and pending entries stay in reference denominators
and are distinct from insufficient-evidence judgments. Full confusion matrices
accompany the results. Native returned choices are scored without relabeling them
from probabilities.

## Execution and timing

Jev 1.13.0 and local Clef-Flash received identical structured state, instructions
and Choice criteria. Each ran sequentially with automatic retries disabled; the
two runs overlapped on one host.

Jev had one HTTP 504 at decision slot 6,975. One separately authorized retry
replaced that outcome; the original failure and retry evidence are retained.
Final quality and latency summaries use the replacement. The interval spanning
that later retry is not continuous run time. Interrupted in-flight replays cannot
be reliably counted from terminal outcomes alone.

Latency includes request construction and result validation, excluding dataset I/O.
Networking, GPU residency and shared-host load differ between deployments, so these
are observed deployment timings rather than a controlled architecture comparison.
`first_call` identifies the first recorded assessment, not a proven cold cache.
Exact settings and runtime details accompany the saved runs.

## Limits

Static replay does not measure live attack prevention, recovery, downstream task
completion or human escalation outcomes. Neither the model-agreement scores nor
the targeted human-review scores establish production authorization reliability.

No model/prompt tuning or label revision followed these guard results. Confidence
exploration in the [detailed analysis](RESULTS_ANALYSIS.md) is retrospective; it
does not establish calibrated thresholds or a deployable model-selection rule.
See the [results](RESULTS.md) for findings and the [dataset guide](../datasets/README.md)
for source hashes and reproduction inputs.
