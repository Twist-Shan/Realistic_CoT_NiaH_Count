# Counting-mechanism transfer to Native-thinking

Date: 2026-08-26<br>
Status: exploratory implementation; natural frozen-prompt discovery running

Final empirical results and claim guidance are recorded in
`docs/internal_counter_causal_transfer_results_20260826.md`.

## Scientific scope

This experiment suite transfers four families from *Understanding Counting
Mechanisms in Large Language and Vision-Language Models* to the existing
Qwen3/Gemma Native-thinking setting:

1. CountScope causal decoding;
2. last-`k` to first-`k` continued counting;
3. mean position-difference linear-additivity steering; and
4. separator collapse plus maximum-latent-count interchange.

All selected traces pass the causal-prefix no-enumeration audit.  The primary
natural cohort contains inline evidence sequences rather than numbered or dash
lists: no item carries a numeric index, ordinal label, running subtotal, or
previously stated total.  Consequently, the natural experiment never labels a
nonexistent marker token as a marker intervention.

The primary H100 run uses the naturally occurring no-enumeration rows audited
from the original frozen-prompt archive.  It does **not** add an unnumbered-list
instruction.  Because these traces have variable totals, the runner now treats
the observed trace length dynamically rather than requiring `N=10`.  Within
each seed, the maximum eligible count is selected without reading final-answer
correctness or any intervention outcome.  Discovery evaluation uses seeds 1246
and 1252 (`N=8`); natural confirmation seeds 1256 (`N=10`) and 1261 (`N=8`)
remain untouched.

The 30-row `format_conditioned_n10` cohort remains available only as an
auxiliary standardized-length panel.  It was generated with an explicit
unnumbered-format instruction and must not be described as frozen-prompt
natural behavior.

The transfer is intentionally stricter than the paper in three respects:

- prompt records are same-length scrubbed for behavioral readouts, preventing
  the original input records from bypassing the reasoning trace;
- the paper CI is accompanied by candidate argmax and greedy generation; and
- `k=1` is reported separately from `k=2,3`.

## Shared intervention

The paper's public implementation replaces decoder-block inputs at every
layer and rescales donor activations to the receiver norm.  The new V5 runner
implements the same pre-block all-layer clamp.  Unequal Native-thinking item
lengths are aligned independently within each event using normalized midpoint
coordinates; alignment never crosses an event boundary.

Regions are:

- `marker`: registered marker tokens, only for cohorts that actually expose
  such tokens;
- `opening`: the first token in an event, used as a one-token positional
  control in natural inline traces;
- `payload`: non-marker item tokens excluding the closing token;
- `closing`: the final non-marker token in the item span; in the natural inline
  cohort this token contains the comma/period event boundary;
- `full`: the complete item span.

## 1. CountScope

For each natural donor occurrence `d`, donor block-input states are clamped
into a fixed receiver prefix.  The receiver preserves only punctuation and
sequence grammar; prompt records, city names, scores, and future items are
absent or scrubbed.  Natural discovery tests the event-closing boundary, the
one-token event opening, and the full event.  The target is count `d`.

Every behavioral count readout now appends a newly tokenized minimal
`</think> ... Total:` suffix immediately after the selected event.  It retains
zero tokens from the original post-event reasoning.  This is necessary because
natural traces can recap the inferred total in words after the evidence
sequence; reusing the original terminal suffix would leak the answer.

Primary sufficiency metrics:

- donor count is candidate argmax;
- greedy output equals `d`.

Positive CI without either adoption endpoint is graded causal information, not
a causally sufficient count state.

## 2. Continued counting

For source endpoint `N_s` and patch width `k`, source occurrences
`N_s-k+1,...,N_s` are clamped into target occurrences `1,...,k`.

The paper hypothesis predicts:

\[
\tilde r=N_s+N_t-k.
\]

The runner tests three endpoints:

1. CountScope decode at target occurrence `k+1`, expected `N_s+1`;
2. CountScope decode at `k+2`, expected `N_s+2`;
3. final candidate/greedy count, expected `N_s+N_t-k`.

It additionally scores and greedily generates the native next bullet.  This
separates numerical continuation from donor-successor retrieval.

Interpretation:

- `k=1` plus hop-2 persistence is evidence for a transferable single-state
  recurrent controller;
- success only at `k=2,3` supports a short trajectory or mini-ledger prefix;
- a score increase without argmax/greedy adoption is sensitivity only.

## 3. Linear additivity

Discovery seeds fit per-layer occurrence centroids at item-closing endpoints:

\[
\mu_k^\ell=\mathbb E[h_k^\ell].
\]

For receiver `i` and intended count `j`, layers 20--26 receive:

\[
\Delta_{i\rightarrow j}^\ell=\mu_j^\ell-\mu_i^\ell.
\]

Controls are the opposite delta and a deterministic equal-norm orthogonal
delta at each layer.  The target is an immediate early-stop count after item
`i`, with future events removed.

In the natural variable-length discovery plan, geometry is fit on seven
disjoint discovery seeds and evaluated only for occurrence 2 shifted to 1 or
3, where the fit archive has repeated support.  The project previously tested
count-subspace/centroid steering and transition
equivariance.  This runner is the first exact Native-thinking analogue of the
paper's *mean position-difference applied over a late layer band*.  Even if it
works, it establishes manipulable affine geometry, not that natural inference
executes vector addition.

## 4. Separator collapse

For the natural inline cohort, the block-input state of occurrence 1's closing
boundary is copied into the closing boundary of every later event.  The
one-token event opening is the equal-budget positional control; payload
collapse is not defined for every natural event because a strictly aligned
event can consist of one token, so full-event collapse is the wider
generic-damage control.  The endpoint is the final count in
a prompt-record-scrubbed trace.

A boundary shortcut requires closing-boundary collapse to damage
probability/argmax/greedy accuracy more than both controls.  Generic damage
from many all-layer clamps is not boundary specificity.

## 5. Maximum latent count

For source prefix count `N_s`, target prefix count `N_t`, and width `k`, the
last `k` donor regions are clamped into the target's last `k` regions.  Future
target events are absent.  The paper hypothesis is:

\[
\tilde r=\max(N_s,N_t-k).
\]

The output records the max hypothesis, pure source-copy hypothesis `N_s`,
target-minus-`k`, and the clean target `N_t`.  Both `N_s<N_t` and `N_s>N_t`
directions are required before discussing a max-like rule.

## Reproduction commands

Local/remote GPU run:

```bash
python scripts/run_realistic_niah_v5_counting_mechanism_transfer.py \
  --model Qwen3-8B \
  --cache-dir /path/to/model-cache \
  --generations work/natural_no_enumeration_audit_20260823/Qwen3-8B/eligible_generations.jsonl \
  --config configs/realistic_niah_v5_counting_mechanism_transfer_natural_discovery_v1.json \
  --output outputs/v5_counting_mechanism_transfer_natural_discovery_v1
```

Analysis:

```bash
python scripts/analyze_realistic_niah_v5_counting_mechanism_transfer.py \
  --trials outputs/v5_counting_mechanism_transfer_natural_discovery_v1/trials.jsonl \
  --output outputs/v5_counting_mechanism_transfer_natural_discovery_v1/analysis
```

The discovery config is deliberately small.  New generation seeds and a
frozen claim contract are required before any confirmatory statement.

## Implementation validation

The implementation and its relevant regressions currently pass 66 CPU tests.
These cover all-layer pre-block hook application, unequal-span event-local
alignment, generalized 1--18 candidate scoring, the paper CI formula,
`k=1`/`k=2` analysis separation, and an end-to-end unnumbered trace through
prompt scrubbing, dynamic trace lengths, outcome-blind per-seed row selection,
and CountScope receiver construction.  Both runner entry
points compile and their command-line interfaces load successfully.

No real-model result is recorded in this document yet.  H100 execution began
on 2026-08-26.  Initial smoke runs exposed three pre-analysis implementation
guards: a requested marker region absent from the inline traces, the upstream
ten-candidate scoring assertion, and a post-list verbal-total leak in the old
early-stop suffix.  The plan was corrected to use the observed closing
boundary, dynamic 1--18 scoring, and a zero-retained-token immediate query
before discovery was launched.  The status remains
*implemented, GPU-executed, and empirically analyzed*.  The completed results
support graded causal ordinal information, but do not support a stable
recurrent counter register; see the linked results document.
