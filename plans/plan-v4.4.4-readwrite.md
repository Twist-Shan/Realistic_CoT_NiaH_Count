# V4.4.4 supplement: factorized attention read and downstream OV write

## 1. Scientific question

V4.4.4 established that Qwen3-8B L28 H16/H19 has a natural, necessary
pre-O `Z -> W_O -> residual` count channel.  Its relay supplement found that
`tail_64` contains a count-correlated value state, but a receiver-alpha / donor-V
patch of that set did not reliably move the answer and its removal was not
necessary.  The unresolved question is therefore not whether H16/H19 can write
count information, but:

> Which part of the natural attention computation supplies their count-varying
> pre-O state, and how does their output propagate through later residual
> states into count logits?

This is an append-only V4.4.4 supplement.  It reads the frozen V4.4.4 candidate
and natural OV axis and writes only under new `read_write_*` stage names plus
supplement-specific status/report files in the same filestream run root.  It
never overwrites the frozen config, stimuli, candidate selection, natural-OV
artifacts, primary analysis, or base manifest.  SHA-256 hashes of the protected
base artifacts are registered before the supplement and verified by the final
audit.

## 2. Exact read decomposition

For one selected head set and one stable source-position group `G`, write

```text
z_G(alpha, V) = sum_(j in G) alpha(q,j) V(j).
```

For a receiver run `R` and donor run `D`, capture the four counterfactual edge
states

```text
z_RR = z_G(alpha_R, V_R)       z_RD = z_G(alpha_R, V_D)
z_DR = z_G(alpha_D, V_R)       z_DD = z_G(alpha_D, V_D).
```

Use the symmetric two-player Shapley decomposition

```text
Delta z_value = 1/2 [(z_RD-z_RR) + (z_DD-z_DR)]
Delta z_route = 1/2 [(z_DR-z_RR) + (z_DD-z_RD)]
Delta z_full  = z_DD-z_RR = Delta z_value + Delta z_route.
```

`value` measures changing the state content while averaging over receiver and
donor routing.  `route` measures changing alpha while averaging over receiver
and donor values.  Their exact closure is checked before any behavioral
interpretation.  The receiver-anchored and donor-anchored effects are also
saved; their difference is the alpha-by-value interaction.

For the all-key primary analysis, `z_RR` and `z_DD` are the actual fused pre-O
states captured from the model, while the crossed `z_RD` and `z_DR` use the
explicit alpha-V calculation.  Consequently `Delta z_value + Delta z_route`
equals the exact donor-minus-receiver pre-O patch, not merely its eager
reconstruction.  The eager receiver/donor endpoint discrepancies are reported
separately and gated at relative L2 `0.05`; this prevents a small but
count-aligned numerical residual from masquerading as a read component.

The full-forward versus attention-cache candidate-logit delta is retained as a
non-fatal numerical diagnostic, with raw and candidate-centered maxima plus a
flag relative to the reference tolerance (`0.5`). It is not a mechanistic
identity at the selected pre-O boundary because later blocks can amplify small
backend/cache differences. The hard numerical gate remains the direct all-key
eager `alpha x V` reconstruction of the actual selected-layer pre-O endpoint at
relative L2 `0.05`.

The primary decomposition uses all keys and therefore exactly reconstructs the
already successful donor-Z patch.  A disjoint position partition attributes
the eager alpha-V portion of that movement to:

1. all registered slot tokens;
2. early pre-query non-slot tokens;
3. the last 64 pre-query non-slot tokens;
4. the answer-query token itself.

The four groups are required to be disjoint and to cover the complete attention
key axis.  Position-group results are attribution, not four independent circuit
claims.

Shapley closure onto the actual donor-Z patch is an algebraic identity and is
gated at relative L2 `1e-5`.  Position-group attribution remains an eager,
additive decomposition; the separately reported endpoint bridge is therefore
a numerical sensitivity check, not a biological/circuit component.

## 3. Causal read tests

At the true L28 pre-O boundary, add each exact component to the receiver's
natural H16/H19 state and score all count candidates.  For each directed donor
pair, report:

```text
mechanical transport = <W_O Delta z, u_OV> / (donor_count-receiver_count)
behavioral transport = Delta E[count] / (donor_count-receiver_count).
```

For both value and route components, remove only the component parallel to the
frozen natural OV axis and compare against an equal-post-O-norm direction that
is orthogonal inside the same H16/H19 `W_O` span.  This separates "the component
can perturb the answer" from "it writes through the already validated natural
OV channel."

The read-mode classification is frozen before seeing results:

- `value-dominant`: value transport passes and exceeds route transport;
- `routing-dominant`: route transport passes and exceeds value transport;
- `mixed`: both pass without a decisive contrast;
- `unresolved`: the full patch transports but neither factor is stable;
- `no transport`: even the exact full patch fails to replicate V4.4.4.

No claim that H16/H19 locate raw needles follows from any classification.

## 4. Downstream write propagation

Fit a natural answer-query count step `s_l` independently at every post-block
layer `l=28..35` on discovery seeds.  On held-out seeds, intervene at L28 pre-O
with `+beta d_z` and `-beta d_z`.  The central finite difference

```text
J_l d_z ~= [h_l(+beta)-h_l(-beta)] / (2 beta)
```

is projected onto `s_l`.  Record its norm, cosine with `s_l`, amplification
relative to the realized L28 post-O step, and the corresponding expected-count
and correct-margin slopes.  Repeat with an equal-post-O-norm H16/H19-span
direction orthogonal to the natural OV axis.

This distinguishes three possibilities:

- the write disappears immediately after L28;
- it persists in residual space but does not align with the downstream natural
  count state;
- later blocks preserve/amplify it and the final count distribution moves.

## 5. Splits and interpretation

- Frozen mediator: Qwen3-8B L28 H16/H19 and the V4.4.4 natural OV axis.
- Read/write-axis discovery: seeds 1264--1273.
- Mechanistic evaluation: seeds 1274--1293.
- Counts: 1--10; directed pairs `(1,6)`, `(6,1)`, `(3,8)`, `(8,3)`,
  `(5,10)`, `(10,5)`; write traces at counts 2/5/8.
- The evaluation seeds were used by the primary V4.4.4 campaign.  Consequently
  this supplement is a registered mechanistic extension, not a globally independent
  replication.  A publication-strength new-seed replication should use
  1294--1313 after the read-mode decision is frozen.
- Primary analyses use all paired trials and continuous candidate-distribution
  metrics.  Baseline-correct and baseline-wrong strata are secondary; PCA or
  count axes are never refit separately by outcome.
- Persist only scalar summaries, fitted axes, and atomic seed shards.  Do not
  persist full hidden-state tensors, full V tensors, or raw attention maps.

## 6. Decision boundary

The strongest supported conclusion available from this campaign is:

```text
state component (V content and/or alpha routing)
    -> H16/H19 pre-O Z
    -> frozen natural OV output axis
    -> downstream answer-query count state
    -> count distribution.
```

It does not identify the upstream attention/MLP modules that constructed the
source value state.  That requires a subsequent layerwise residual path-patch
campaign after the read mode is known.
