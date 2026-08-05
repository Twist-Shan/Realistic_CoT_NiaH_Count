# V4.4.4 natural-OV causal protocol

## Scientific boundary

The protocol does not require QK-localizer heads and OV-transporter heads to be
the same set.  A direct attention head has both circuits internally, but causal
roles may be distributed across different heads or layers.  The primary claim
tested here is narrower:

> A frozen transporter set naturally carries a prompt-count direction through
> its V channel and O projection into a downstream count-relevant residual
> direction, and removing that channel damages counting.

QK localization is tested separately with source-position interventions and
can later be connected to the transporter set by edge/path patching.

## Natural OV direction

For an ordinary attention layer, let `s_P` be the fitted one-count prompt-state
slope after the real pre-attention norm.  For query head `h` and its GQA KV head
`g(h)`, define the pre-O count step

```text
d_z,h = W_V[g(h)] s_P.
```

For a nonlinear or shared-KV value path (Gemma), `d_z,h` is the empirical
one-count slope after the real V projection and value normalization at the
resolved provider layer.  The set output step is

```text
m_S = sum_h W_O[h] d_z,h.
```

No answer direction is used to construct the intervention.

## Injection

At the answer query and before O projection, apply

```text
z_h <- z_h + beta * d_z,h,  h in S.
```

The model then realizes `beta * m_S` through its own O projection.  The primary
dose-response is the change in expected count per beta.  Report both the
natural-scale effect and the realized post-O delta norm.  A post-O projection
of the answer direction is only a layer steerability control and is not counted
as OV evidence.

## Removal

Let `o_S` be the actual selected-set output and `u_m = normalize(m_S)`.  Remove
the selected set's actual component along its natural OV axis:

```text
c_S = dot(o_S, u_m)
delta_z,h = -(c_S / ||m_S||) d_z,h.
```

This guarantees

```text
sum_h W_O[h] delta_z,h = -c_S u_m,
```

so the intervention is realizable by the selected heads.  The control is an
equal-output-norm direction orthogonal to `u_m` but inside the same selected
W_O column span, also applied at pre-O Z.

The implementation accepts an optional count-neutral `z_center`, estimated on
independent discovery seeds, and then removes the coefficient of
`o_S - W_O^S z_center`.  Without that center the operation is a zero-ablation
and can remove a static offset as well as count-varying signal; this uncentered
variant should be reported as a sensitivity analysis rather than the primary
necessity test.

## Evidence required for natural use

1. The frozen set has positive held-out `cos(m_S, u_A)` relative to a null.
2. Natural pre-O OV injection has a signed, approximately monotone dose effect.
3. Natural-axis removal harms count error/margin more than the in-span control.
4. Donor Z transport is mediated by the same downstream axis: blocking that
   axis suppresses the donor-patch effect.
5. Candidate effects exceed a distribution of matched sets, matched on output
   delta norm, reachable cosine, GQA KV composition, and baseline output norm.
6. For a proposed multi-head core, single-head, joint, and leave-one-out runs
   distinguish synergy, dominance, and redundancy.

Use fresh confirmation seeds and freeze the model/layer/head set before looking
at these causal outcomes.  Nested K exploration belongs to discovery and should
not be counted as multiple independent confirmations.

## Frozen V4.4.4 campaign instantiation

- Primary candidate: Qwen3-8B, L28, heads `{16,19}` only.
- Direction discovery: seeds 1234--1253, inherited from the completed V4.4 run.
- Count-neutral Z center and matched-control selection: new seeds 1264--1273.
- Causal confirmation: new seeds 1274--1293.
- Four K=2 controls share the candidate's GQA within-group offsets and are
  chosen before causal outcomes using natural-step norm, natural-answer cosine,
  reachable-answer cosine, and baseline set-output norm.
- Primary evidence is a four-family intersection-union test: natural carrier,
  pre-O injection, centered removal, and donor-Z mediation must all pass at
  alpha 0.05, including candidate-minus-control-mean specificity.
- Previously discovered L28 K=3/4/6/8 nested sets are secondary robustness
  analyses only.  Their family p-values are Holm-adjusted across K and cannot
  replace the K=2 primary decision.
