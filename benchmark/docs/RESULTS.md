# Frozen Jev / Clef-Flash results

Final results dated 2026-10-31. Both deployments have 10,357 valid final verdicts,
zero pending entries and zero remaining operational errors. Jev's original run had
one HTTP 504; one owner-authorized retry replaced only logical slot 6,975. Clef had
no saved failures. The release includes the original failure and retry disclosure.

| Measurement | Hosted Jev 1.13.0 | Local Clef-Flash Q8_0 |
| --- | ---: | ---: |
| Primary-label agreement | 9,734/10,357 (94.0%) | 7,439/10,357 (71.8%) |
| Human-reviewed accuracy | 423/741 (57.1%) | 353/741 (47.6%) |
| Reviewed unauthorized denied | 53/190 (27.9%) | 52/190 (27.4%) |
| Reviewed authorized falsely denied | 139/537 (25.9%) | 182/537 (33.9%) |
| Reviewed insufficient-evidence predictions | 33/741 (4.5%) | 66/741 (8.9%) |
| Successful median / p95 latency | 0.349 / 0.508 s | 0.803 / 1.320 s |

These tracks have different references. The 741 human reviews are targeted and
model-visible; primary agreement is not independent human accuracy. See
[methodology](METHODOLOGY.md) for source, review and runtime limitations.

| Cohort / measurement | Jev | Clef |
| --- | ---: | ---: |
| 9,666 annotator agreements: exact consensus match | 9,325/9,666 (96.5%) | 7,060/9,666 (73.0%) |
| 691 disagreements: exact human match | 377/691 (54.6%) | 329/691 (47.6%) |
| Disagreement unauthorized explicitly allowed | 130/171 (76.0%) | 112/171 (65.5%) |
| Disagreement unauthorized escalated | 5/171 (2.9%) | 10/171 (5.8%) |
| Disagreement unauthorized denied | 36/171 (21.1%) | 49/171 (28.7%) |
| Disagreement authorized denied or escalated | 166/507 (32.7%) | 227/507 (44.8%) |
| 50 sampled agreements: exact human match | 46/50 (92.0%) | 24/50 (48.0%) |

Clef permits 18 fewer unauthorized cases in the contested cohort while interrupting
61 more authorized ones. Both still explicitly allow most unauthorized references
there. This tradeoff does not reverse Jev's advantage on the consensus cohort and
does not establish a way to route live cases by cohort. Neither tested deployment
supports a claim of production-grade authorization reliability or proven attack prevention.

The [self-contained HTML](results/benchmark-results-interpretation.html) preserves
full matrices and unsafe-allow versus escalation accounting. [Detailed analysis](RESULTS_ANALYSIS.md)
includes cases, duplicates and retrospective confidence exploration. Supporting
counts and case studies live in `benchmark/docs/results/`.

Frozen SHA-256 identities:

```text
dataset   c92191ff4a2378563e4ac991b07fba66f9cb55f539ac2c5e081deb1aadb7d87f
Jev       569df163a891fa7cf4d806a3e79aa8cecb977dfc8f49b5504118e5a804f1d202
Clef      9545f49669f773565acb39ee9f06b6dd83f7010200931203871d76f97882db1c
```

[Offline reproduction](../README.md#reproduce-the-full-reports) uses installed wheels and extracted artifacts
with networking/inference disabled. Original reports retain historical absolute
source paths. Regenerated reports change those paths and corresponding prose,
not scores, datasets, models, prompts, verdicts or timing observations.
