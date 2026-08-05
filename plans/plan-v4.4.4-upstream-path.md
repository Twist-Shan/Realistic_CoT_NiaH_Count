# V4.4.4 append-only upstream broad-retrieval → L28 OV path pilot

## Scope and isolation

This is an additive V4.4.4 supplement.  It does not alter V4.4.4 candidate
selection, directions, relay, read/write results, configs, or reports.  New
artifacts live only under `upstream_path_base`, `upstream_path_expanded`, and
`upstream_path_analysis`.  A SHA-256 snapshot protects the completed base,
relay, and read/write manifests.

## Frozen hypothesis

Earlier heads with a cue-stable V4.4.2 broad-retrieval score may supply a
count-bearing state that reaches the already supported L28 H16/H19 OV write
set.  QK localization and OV writing need not be performed by the same heads.

The early ranking is frozen before causal outcomes by descending

`min(S_broad(cue-present), S_broad(cue-absent))`.

Nested top-2/top-4/top-8 sets avoid post-outcome candidate selection.  The
primary late set remains L28 H16/H19.  Additional L28 sets are evaluated only
when H16/H19 fails to mediate either constrained route.

## Three early interventions

1. `slot_edge_qk`: at the answer query, replace only the selected slot-edge
   contribution caused by attention routing, using donor alpha and receiver V.
2. `slot_state`: at registered slot query positions only, replace selected
   early-head pre-O Z with the donor state.  The answer query is untouched at
   these layers; any downstream effect must propagate causally forward.
3. `answer_query_full`: replace full early-head answer-query Z.  This is an
   upper bound and is not interpreted as a QK/V decomposition.

## Serial mediation test

For each early intervention, capture its induced L28 selected-head pre-O
change `delta_z`.  Compare:

- early intervention alone;
- early intervention plus exact L28 restoration `-delta_z`;
- early intervention plus a same-span, equal post-O norm direction orthogonal
  to `W_O delta_z`.

The primary behavior endpoint is donor-vs-receiver candidate-sequence log-odds
gain.  Unlike expected count, it does not saturate when the clean receiver is
already assigned probability near one.  Expected-count transport remains a
secondary effect-size scale.  The seed-level effect is averaged across six
directed donor pairs.  A serial path is supported only if early donor log-odds
gain and L28 log-odds mediation specificity are both positive under two-sided
exact sign-flip tests with Holm correction.

## Evidence boundary

The ten seeds are reused V4.4.4 confirmation seeds.  Results are explicitly
exploratory and require new-seed replication.  No raw attention, QK cache, or
full hidden state is persisted; only scalar summaries and reports are stored.
