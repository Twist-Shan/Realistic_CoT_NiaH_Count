# V3.1 Observable Counting-Style Codebook

## 1. Purpose and interpretation boundary

This codebook classifies visible structure in a model completion. It does not
identify a latent reasoning process and must not be used to claim that an
observed chain of thought is the model's internal counting algorithm.

The unit is one request completion. Classification is performed without access
to the true `N`, parsed correctness, signed deviation, or empirical-law
residual. The frozen classifier produces:

1. non-exclusive core-style flags;
2. non-exclusive process/failure flags;
3. one exclusive `dominant_style`;
4. an `observability` field describing which reasoning text was available.

The final `Total: <integer>` answer line is excluded when identifying a
counting strategy, but retained for format and accuracy analysis.

## 2. Reasoning text and observability

Use the following source order:

1. a separately returned reasoning field, when the inference backend exposes
   one;
2. otherwise, visible completion text before the final registered answer line;
3. if neither contains content, classify the request as `answer_only` and set
   `empty_reasoning=true`.

Record exactly one observability value:

- `separate_reasoning`: a backend-provided reasoning field is available;
- `inline_reasoning`: reasoning is visible in the ordinary completion text;
- `final_only`: only the final answer is visible;
- `missing_completion`: no completion text is available because of a
  documented runtime failure.

Cross-model differences in observability must be reported because an absent
visible trace is not evidence that no internal reasoning occurred.

## 3. Core style flags

Core flags are non-exclusive. A trace may, for example, contain both an index
enumeration and a final arithmetic grouping.

### 3.1 `index_enumeration`

The reasoning contains at least two candidate records/items marked by explicit
numeric indices or numeric ordinals, such as `1.`, `2.`, `1)`, `2)`, or an
unambiguous equivalent. A single numbered heading is insufficient.

### 3.2 `bullet_enumeration`

The reasoning contains at least two candidate records/items introduced by
bullet markers such as `-`, `*`, or `•`. Ordinary hyphens inside prose or
negative numbers do not count.

### 3.3 `word_enumeration`

The reasoning serially names or describes at least two candidate records in
prose without using a numeric-index or bullet structure. This includes written
cardinal/ordinal counters (`one`, `two`, `first`, `second`, and equivalents)
and prose sequences of city-score records. Merely saying "I counted the
records" is insufficient.

### 3.4 `running_tally`

The reasoning exposes at least two cumulative count updates, for example
`count=1 ... count=2`, `so far 3 ... now 4`, or equivalent incremental tally
language. A single final count is insufficient.

### 3.5 `arithmetic_grouping`

The reasoning divides candidates into two or more groups/subcounts and
combines them arithmetically, for example `4 + 3 + 2 = 9`. An isolated
arithmetic expression unrelated to record counts is insufficient.

### 3.6 `scan_or_retrieval_summary`

The reasoning describes scanning, searching, matching, or retrieving records
and reports a count without exposing at least two enumerated items or tally
updates. Generic filler such as "let me think" is insufficient.

### 3.7 `answer_only`

No visible counting strategy occurs outside the final answer. This flag can
co-occur with `empty_reasoning`, but not with any of the six strategy flags
above.

### 3.8 `mixed`

`mixed` is used as a dominant label, not as a substitute for core flags. It is
assigned when two or more core strategies each make a material contribution
and neither is clearly the organizing strategy of the trace.

### 3.9 `other_unclassifiable`

Visible reasoning exists but cannot be assigned reliably to a registered core
style. The raw text is retained for audit. New confirmatory categories cannot
be created after correctness or bias outcomes have been joined.

## 4. Process and failure flags

These flags describe execution quality rather than the core counting style:

- `repetition`: a record, list segment, or reasoning segment is substantially
  repeated;
- `restart`: the model abandons or explicitly restarts a count/list;
- `self_correction`: an earlier count or item decision is explicitly revised;
- `template_leakage`: placeholders, prompt instructions, tags, or answer
  templates are copied as if they were content;
- `truncated`: generation reached the registered output-token limit;
- `empty_reasoning`: no non-answer reasoning text is observable.

Repetition and restart remain flagged even if the final answer is correct.

## 5. Dominant-style decision rule

Assign exactly one dominant label using the trace's organizing structure:

1. If no core strategy is present, assign `answer_only` when no reasoning is
   visible; otherwise assign `other_unclassifiable`.
2. If exactly one of the six strategy flags in Sections 3.1-3.6 is present,
   assign it.
3. If multiple strategy flags are present, assign the strategy that organizes
   the candidate records and produces the final count. A short arithmetic sum
   after a full indexed list, for example, remains `index_enumeration`.
4. If two or more strategies materially organize the count and no strategy is
   dominant, assign `mixed`.

Process/failure flags do not replace the dominant style. A restarted indexed
list is still `index_enumeration` with `restart=true`.

## 6. Automated classification and human validation

The automated classifier may use deterministic parsing, regular expressions,
and text-structure features. It must not use `N`, correctness, deviation,
model-level accuracy, or empirical-law residuals. Classifier code, rules, and
hashes are frozen before applying it to the outcome-linked analysis table.

Human validation consists of two separately reported samples:

### 6.1 Prevalence-valid random sample

Draw 600 requests using a reproducible seed, stratified to cover all four
prompt modes, five model families (Qwen, Gemma, Nemotron, GLM, and Ministral),
and the registered `N`/`L` ranges. Sampling weights are retained so aggregate
validation estimates can reflect the full registered request set.

### 6.2 Challenge sample

Draw 200 additional requests enriched for classifier uncertainty, unusually
long traces, multiple detected styles, repetition/restart flags, parse
failures, and truncation. This sample evaluates difficult cases and is not
used to estimate population prevalence or headline classifier performance.

Two annotators independently label both samples while blinded to correctness,
true `N`, parsed prediction, and signed error. Disagreements are adjudicated
after the independent pass. Report before-adjudication agreement and final
adjudicated labels.

Minimum validation thresholds for confirmatory automated style reporting are:

- Cohen's kappa >= 0.75 for the exclusive dominant style on the weighted
  random sample;
- macro-F1 >= 0.80 across the non-exclusive core-style flags on the weighted
  random sample.

A core flag with zero human-positive support and zero automated-positive
support in the random validation sample is reported as unsupported and omitted
from the macro-F1 denominator; assigning it an artificial F1 of zero would
penalize agreement on a style that did not occur. A flag with either human or
automated positive support remains in the denominator, including when its F1
is zero.

Also report per-label precision, recall, F1, support, and confusion matrices.
If either threshold is missed, automated style-outcome associations are
labelled exploratory; headline style prevalence is based on an appropriately
weighted manual estimate or reported with an explicit measurement-error
warning. The challenge sample is always reported separately.

## 7. Registered style summaries

Report, at minimum:

- dominant-style and multi-label prevalence by prompt mode and behavior slot;
- observability by model/checkpoint and prompt mode;
- style prevalence over `N` and `L` with uncertainty intervals;
- repetition, restart, self-correction, leakage, and truncation rates;
- prompt adherence: index style under Index Enumeration and bullet style under
  Bullet Enumeration;
- accuracy, parse rate, format compliance, and robust conditional bias by
  observed style.

The last item is descriptive and associational. Prompt mode can affect both
style and outcome, and style is measured after treatment; conditioning on it
does not identify a causal effect of style on counting performance.

## 8. Codebook amendment rule

Before outcome joining, ambiguous examples may motivate clarifying examples or
splitting `other_unclassifiable`. Every change must be dated, justified, and
made while annotators remain outcome-blinded. After outcome joining, new or
revised categories are exploratory and the original registered labels must
still be reported.
