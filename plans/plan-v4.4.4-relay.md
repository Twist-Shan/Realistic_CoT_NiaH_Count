# V4.4.4 relay-to-OV serial-mediation supplement

## Scientific claim

This supplement does **not** test whether Qwen L28 H16/H19 use their own QK
circuits to locate raw needles. The frozen claim is narrower:

```text
relay value-state set P
  -> receiver-alpha read into L28 H16/H19 pre-O Z
  -> the already frozen V4.4.4 natural OV axis
  -> answer-count distribution
```

The experiment is an additive subcampaign inside the completed V4.4.4 run.
It does not overwrite the original resolved config, completion marker,
confirmation tables, or report. Its stage roots are `relay_discovery`,
`relay_smoke`, `relay_confirmation`, and `relay_model`; analysis lives under
`analysis/relay`.

## Why alpha need not be count-specific

For a selected head `h`,

```text
z_h(q) = sum_j alpha_h(q,j) v_g(h)(j).
```

A fixed read pointer can retrieve a value whose content changes with count.
H16 and H19 are in the same four-query-head GQA group and share one KV source.
The relay test therefore keeps receiver Q, K, and alpha fixed and changes only
the donor value content on a registered source-position set. To avoid changing
the other query heads in the same GQA group, the intervention computes the
selected edge contribution and applies it only to the H16/H19 pre-O slices:

```text
delta z_h,P = sum_(j in P) alpha_h^receiver(q,j)
              [v^donor(j) - v^receiver(j)].
```

This is exactly the selected-edge V-only change for ordinary attention. It is
not a direct donor-Z patch because positions outside `P` stay at receiver V,
and it does not patch alpha.

## Position-set discovery

The downstream head set and natural OV axis remain frozen from V4.4.4. On the
ten original center seeds, calculate the additive contribution of every source
position along that axis:

```text
rho_j = sum_(h in {16,19}) alpha_h(q,j)
        <W_O^h v_g(h)(j), u_m>.
```

Only aggregate position-set rows are saved. Raw per-token contribution maps
and full V tensors are discarded after each example.

Eligible relay sets are:

- answer-query self;
- non-slot tokens in the last 16 or 64 pre-query positions;
- non-slot top-K positions by absolute natural contribution for
  K in {4,8,16,32,64}.

Active-needle endpoints and active full spans are source controls and cannot be
selected as the primary relay. Selection maximizes the discovery seed-level
positive count-slope t score, with mean slope > 0 and at least 70% positive
discovery seeds. Selection uses no causal output.

## Confirmation estimands

Confirmation reuses the twenty original V4.4.4 confirmation seeds. These seeds
were not used for relay selection, although they are not globally new to the
larger V4.4.4 project. Every donor pair is run in both directions.

### Natural relay signal

Fit a count-zero edge-Z center and one-count edge-Z step on discovery fit
counts. On confirmation seeds, the relay carrier coefficient must have a
positive within-seed slope across counts 1--10.

### Receiver-alpha/donor-V edge patch

The mechanical first stage is the patch-induced L28 set output projected onto
the frozen global OV axis, divided by the donor-receiver count gap. The
behavioral effect is the expected-count change divided by the same signed gap.
Both must be positive.

### Downstream OV mediation

For the patch-induced set output, remove only the component parallel to the
frozen V4.4.4 natural OV axis at pre-O Z. Compare it with a direction in the
same H16/H19 W_O column span that is orthogonal to the axis and has equal
post-O norm. The normalized transport under the orthogonal control must exceed
transport under the natural-axis block.

### Relay necessity

Remove the natural centered edge-output component along the discovery-fitted
relay step. The matched control is in the same selected-head W_O span,
orthogonal to the relay axis, and equal in post-O norm. Removal must increase
expected-count absolute error and reduce correct-count margin relative to the
control.

## Primary decision

Use exact seed-cluster sign-flip tests. The global decision is a four-family
intersection-union test at alpha 0.05:

1. natural relay slope;
2. edge-patch mechanical first stage **and** behavioral transport;
3. downstream OV mediation specificity;
4. relay removal error **and** margin specificity.

The relay claim is supported only if selection qualified, every family passes,
and the artifact audit passes. Injection without removal remains evidence of
accessibility rather than natural use.

## Numerical validity gates

- The selected-module eager/cache rerun candidate-logit delta is persisted as
  a numerical diagnostic. It is not the hard mechanistic gate because later
  layers can amplify a small representation difference.
- The hard attention/value validity gate is direct reconstruction of the
  original selected-layer pre-O state from the captured `sum(alpha * V)`;
  relative L2 must be at most 0.05.

## Interpretation limits

- Passing validates a terminal `relay -> L28 OV -> answer` path.
- It does not identify the upstream heads or MLPs that construct the relay.
- A winning dynamic top-K set supports that registered dynamic rule, not fixed
  absolute token indices.
- A source-control effect at raw needle spans is not counted as relay evidence.
