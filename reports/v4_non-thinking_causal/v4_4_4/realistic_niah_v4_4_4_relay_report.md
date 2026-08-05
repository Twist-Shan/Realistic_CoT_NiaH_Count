# Realistic NIAH V4.4.4 relay-to-OV supplement

## Registered claim

This supplement does not require L28 H16/H19 to locate raw needles through
their own QK circuit. It tests the narrower serial path
`relay value set -> receiver-alpha read -> L28 H16/H19 pre-O Z -> answer`.

The selected position set is `pre_query_non_slot_tail_64`. Selection
used only natural source-contribution rows from the discovery seeds and did not
use confirmation causal outcomes. The result is **NOT SUPPORTED** under the frozen
four-family intersection-union rule (global p=0.998087).

## Results

- Natural relay carrier slope: 0.840411 [0.756004, 0.924721], p=9.53674e-07.
- Receiver-alpha/donor-V first-stage transport: 0.002825 [0.001283, 0.004457], p=0.000405312.
- Receiver-alpha/donor-V behavioral transport: 0.001620 [-0.000262, 0.003652], p=0.0692835.
- L28 natural-OV mediation specificity: -0.000014 [-0.001416, 0.001455], p=0.507699.
- Relay removal error specificity: -0.036391 [-0.060461, -0.012210], p=0.994787.
- Relay removal margin specificity: 0.214583 [0.087501, 0.337499], p=0.998087.

The natural carrier and the mechanical OV-axis first stage are supported, but
the answer-level transport interval crosses zero and the registered OV block
does not outperform its matched orthogonal control. Both removal estimands are
in the direction opposite to natural necessity. The defensible conclusion is
therefore that the selected late value-state carries count information and is
mechanically accessible through H16/H19's OV subspace, not that the model
naturally depends on this terminal relay-to-OV path to produce the answer.

## Interpretation boundary

The V-only edge patch keeps receiver Q, K, and alpha fixed. A positive result
therefore supports transport of content already present in a relay value state;
it does not identify the upstream heads that created that state. The natural
axis block tests whether the patch effect passes through the frozen L28
H16/H19 OV channel. Removal is required to distinguish natural use from mere
intervention accessibility.

## Audit

Relay audit: `True` across
13 checks.
No raw per-token contribution maps or full V-state tensors are persisted.
The eager/cache final-logit delta is retained as a non-fatal numerical
diagnostic. Direct reconstruction of the original L28 pre-O z from
`sum(alpha * V)` is the hard validity gate (relative L2 <= the registered
threshold, 0.05 in this campaign).
