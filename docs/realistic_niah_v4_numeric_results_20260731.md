# Realistic NIAH V4 numeric non-thinking results

This note records the completed descriptive V4 run
`run_20260731_v4_numeric_presentation_v3`. It separates measured results from
causal claims: head ablation, head-output patching, residual patching, and
geometric steering have tested implementations, but the full registered causal
sweeps were not part of this run.

## Registered design

- Models: `Qwen3-8B` and `Gemma4-E4B`.
- Context length: 10,000 canonical passage tokens.
- Counts: decimal `1` through `10`.
- Panels: v4.1 fixed; v4.2 position-relaxed; v4.3 city-score-order-relaxed;
  v4.4 city-score-content-relaxed.
- Seeds: 30 paired seeds per panel and count. Seeds 1234--1253 are discovery;
  seeds 1254--1263 are confirmation.
- Behavior label: the parsed complete deterministic greedy continuation after
  the prompt-final `Total:` query. This includes the full multi-token sequence
  for `10`; no first-token candidate probability is used.

## Completion audit

| Artifact per model | Qwen3-8B | Gemma4-E4B |
| --- | ---: | ---: |
| Behavior rows | 1,200 | 1,200 |
| Unique stimulus IDs | 1,200 | 1,200 |
| Representation capture shards | 120 | 120 |
| Raw answer-query attention tensors | 1,200 | 1,200 |
| Non-empty attention-analysis tables | 12 | 12 |
| Non-empty attention figures | 23 | 23 |
| Non-empty labeled representation outputs | 7 | 7 |
| Raw attention-capture bytes | 28,362,547,387 | 1,783,768,868 |

All 54 absolute artifact references found in the five manifests for each model
resolved to existing, non-empty files. The complete remote run is about 33 GB;
Qwen occupies about 30 GB and Gemma about 2.6 GB. The apparent size difference
comes primarily from the registered query-row storage shape, not from a
difference in the number of examples.

## Behavior

Every generated answer was format-valid. The failures below are therefore
numeric errors rather than parser or formatting failures.

| Model | v4.1 confirmation | v4.2 confirmation | v4.3 confirmation | v4.4 confirmation | All 1,200 rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 35% | 45% | 48% | 44% | 45.92% |
| Gemma4-E4B | 43% | 37% | 38% | 37% | 37.58% |

The dominant behavioral transition is count-dependent, not panel-dependent.
On the 400 confirmation examples per model, accuracy by true count is:

| Count | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 92.5% | 100% | 82.5% | 95% | 35% | 17.5% | 0% | 7.5% | 0% | 0% |
| Gemma4-E4B | 100% | 100% | 100% | 45% | 25% | 10% | 2.5% | 2.5% | 0% | 2.5% |

At counts 9 and 10, Qwen undercounts every confirmation example. Gemma
undercounts every count-9 example and 97.5% of count-10 examples. Thus overall
accuracy alone hides a sharp capacity/failure boundary around counts 4--6.

## Prompt-reading representations

Layers were selected only from v4.1 discovery data by maximum grouped-seed
ridge cross-validation R2. The selected layers were Qwen L1 (`span_end`) and L0
(`span_mean`), and Gemma L22 (`span_end`) and L0 (`span_mean`). The same fitted
protocol was then evaluated on confirmation seeds and progressively relaxed
panels.

### Confirmation metrics at the registered primary layers

| Model / pooling | Metric | v4.1 | v4.2 | v4.3 | v4.4 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen span-end L1 | R2 | 0.989 | 0.926 | 0.874 | 0.866 |
| Qwen span-end L1 | MAE | 0.216 | 0.608 | 0.792 | 0.822 |
| Qwen span-mean L0 | R2 | 0.999 | 0.999 | 0.691 | 0.596 |
| Qwen span-mean L0 | MAE | 0.046 | 0.065 | 1.264 | 1.489 |
| Gemma span-end L22 | R2 | 0.954 | 0.914 | 0.890 | 0.916 |
| Gemma span-end L22 | MAE | 0.450 | 0.671 | 0.756 | 0.661 |
| Gemma span-mean L0 | R2 | 0.997 | 0.997 | -0.168 | -0.177 |
| Gemma span-mean L0 | MAE | 0.118 | 0.130 | 2.576 | 2.611 |

Paired confirmation-seed bootstraps show that span-end probe MAE first worsens
at v4.2 in both models. Span-end curve residual relative to signal first has a
strictly positive paired 95% interval at v4.2 for Qwen and v4.3 for Gemma.
Span-mean count decoding remains nearly perfect through v4.2 but collapses at
v4.3, especially for Gemma.

The defensible interpretation is therefore narrower than "a clean scalar
counter." A count-decodable signal persists at the needle end through v4.4,
but it becomes less accurate and seed-sensitive as controls are relaxed. The
span-mean result is strongly dependent on the fixed city-score organization and
does not survive the v4.3 relaxation. PCA alone should not be treated as causal
or as proof of an abstract counter.

## Answer-query attention

Discovery-only rankings are stable under 500 seed bootstraps, but `span_end`
and `span_mean` expose different mechanisms.

- Qwen `span_end`: L29H3 is rank 1 in all four panels with top-8 selection
  frequency 1.0. Its coverage is only about 0.22 and its effective number is
  about 1.04, so it concentrates on roughly one needle rather than aggregating
  uniformly across all needles.
- Qwen `span_mean`: L27H18 is rank 1 in v4.1, v4.3, and v4.4; L28H19 is rank 1
  in v4.2. Coverage is about 0.79--0.86 and the effective number is about
  4.8--5.2, which is much closer to broad aggregation.
- Gemma `span_end`: the top head is L5H2 in v4.1 and L29H6 in v4.2--v4.4.
  Coverage is about 0.85--0.89 and the effective number is about 5.0--5.3.
- Gemma `span_mean`: L35H2 is rank 1 in every panel. Coverage is about
  0.76--0.80 and the effective number is about 4.6--4.8.

Most count-adjusted wrong-minus-correct bootstrap intervals include zero:
only 36 of 256 Qwen comparisons and 24 of 256 Gemma comparisons exclude zero
before any multiplicity correction. Aggregate needle mass alone therefore does
not reliably separate correct from wrong answers.

## Omission and nested-increment diagnostics

These diagnostics are descriptive correlations on confirmation examples, not
interventions.

- For Qwen undercounts, `span_end` attention to the omitted tail relative to
  the retained prefix averages 0.28--0.37 across panels; the low-attention
  bottom-k set overlaps the omitted tail by 0.38--0.58. The analogous
  `span_mean` tail ratio is 0.79--0.84.
- For Gemma undercounts, the `span_end` tail ratio is 0.68--0.93 and bottom-k
  overlap is 0.29--0.46. The `span_mean` tail ratio is usually above 1, with
  only 0.11--0.16 bottom-k overlap.
- When a nested N to N+1 pair registers the increment, attention assigned to
  the newly added needle is higher than when it fails to increment: Qwen
  `span_end` 0.474 versus 0.379 and `span_mean` 1.099 versus 0.808; Gemma
  `span_end` 0.824 versus 0.692 and `span_mean` 1.324 versus 0.983.

The omission hypothesis is consequently most visible for Qwen `span_end`, and
the nested-pair association appears in both models. Neither establishes that
low attention causes the undercount. That question is reserved for the
registered broad-head ablation, head-output patching, and needle-state patching
sweeps with layer-matched random controls.

## Reproducibility and next causal step

The run root is `run_20260731_v4_numeric_presentation_v3`. Each model directory
contains behavior labels, representation captures, raw answer-query attention,
analysis manifests, tables, and figures. The causal implementation uses actual
complete greedy numeric generation and supports:

1. answer-query and global broad-head ablation;
2. pre-output-projection head-slice patching;
3. answer-query, needle-end, and exact tokenwise full-needle residual patching,
   with single-layer and cumulative-from-layer protocols; and
4. discovery-fit centroid transplant/delta, chord/polyline geometric steering,
   and norm-matched random controls.

The full causal sweep must be reported separately. Until it is run, all claims
in this note remain behavioral, representational, or attentional rather than
causal.
