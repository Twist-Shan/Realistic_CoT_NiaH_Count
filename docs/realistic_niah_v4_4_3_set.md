# Realistic NIAH V4.4.3-Set: preregistered small-head-set causal test

## Hypothesis

The prompt running-index trajectory and the answer-count trajectory encode the
same latent count variable in different residual directions. A small set of
attention heads, rather than one individually sufficient head, may jointly
implement the change of carrier: QK selects the relevant prompt locations, V
extracts count-related features, and the sum of the selected O blocks writes
those features toward the answer-count subspace.

**Section conclusion.** The causal unit is a frozen head set. The earlier K=1
experiment is retained as a baseline and is not treated as a final rejection
of the set hypothesis.

## Set definition and geometry

For a layer-local head set `S`, define

`m_S = sum_(h in S) M_OV^(h) u_prompt`,

`r_S = cos(m_S, u_answer)`.

`u_prompt` is fit after the actual pre-attention normalization. For Gemma's
shared-KV layers, `M_OV^(h) u_prompt` is the empirical count slope after the
actual provider-layer V projection and `v_norm`, followed by the target-layer
O block. Fit and held-out count directions are estimated separately.

Nested candidate sets are selected greedily: Qwen uses K=1,2,3,4,6,8 and Gemma
uses K=1,2,3,4, where K=1 is the frozen earlier baseline. Gemma has eight query
heads in each registered layer, so K>4
would leave no disjoint same-size control set. Starting from the
empty set, each step adds the head that maximizes the fit-count cosine of the
summed mapped vector. Only discovery seeds and odd fit counts are used.
Held-out even counts do not affect selection. Each candidate has a disjoint,
same-layer, same-size control chosen from the closest mapped-norm pool and then
closest to the pool's median mapping score.

**Section conclusion.** A positive held-out `r_S` tests geometric
generalization; it is not reused to choose the set.

## Staged set patch

For donor count `d` and receiver count `r`, all members of `S` are patched in a
single forward pass. The five interventions are:

1. donor attention rows with receiver values (`alpha_receiver_v`);
2. position-scrambled donor rows with receiver values;
3. simultaneous donor pre-O aggregates (`z_donor`);
4. replacement by the actual post-O vector captured from the Z-patched forward
   (`o_donor`);
5. an equal-norm residual direction control.

The transport statistic is

`T = (E[N | intervention] - E[N | baseline]) / (d - r)`.

The registered value-transport contrast is `T_z - T_norm`; the QK localization
contrast is `T_alpha - T_scramble`. Candidate-minus-matched set contrasts are
computed within the same seed and donor/receiver pair. Z/O equality is an
implementation audit, not duplicate causal evidence.

**Section conclusion.** Positive candidate-minus-matched Z transport supports
set-specific count transport; alpha is auxiliary evidence about localization.

## Set-specific removal and injection

Let `o_S = sum_(h in S) W_O^(h) z_h` at the answer query. Removal subtracts
`Proj_(u_answer)(o_S)` from the layer's post-O residual. The control subtracts
an equal-norm deterministic direction orthogonal to `u_answer`.

For injection, concatenate the selected output blocks:

`B_S = [W_O^(h1), ..., W_O^(hK)]`.

Define the set-reachable answer direction

`u_answer,S = P_col(B_S) u_answer / ||P_col(B_S) u_answer||`.

The signed intervention is `beta * s_layer * u_answer,S`, where `s_layer` is
the discovery answer-count step norm. Unlike the first V4.4.3 implementation,
this direction is constrained to the selected set's O-output span. Candidate
and matched sets therefore receive different, set-specific reachable
directions.

**Section conclusion.** Removal tests whether the observed set output carries
answer-direction signal; injection tests whether that set's output span can
causally steer count in the registered direction.

## Evidence split and interpretation

- Discovery: seeds 1234--1253; odd counts fit, even counts held out.
- Screen: seeds 1254--1258; staged set patch.
- Confirmation: seeds 1259--1263; removal and injection.
- Models/layers: Qwen3-8B L28--L30; Gemma4-E4B L36--L38.
- Candidate answer scores use the full answer plus chat-termination sequence.
- Original hidden states stay in the frozen V4.4 filestream. No raw attention
  rows or full hidden states are written by V4.4.3-Set.

Evidence is reported by family: mapping generalization, Z transport,
set-reachable injection, and answer-direction removal. A descriptive
raw-triangulation flag requires held-out mapping in the predicted direction and
matched-set exact `p <= 0.05` in at least two of the three distinct intervention
families. Because the enlarged K grid creates multiple tests, the report's main
triangulation flag instead uses Benjamini-Hochberg `q <= 0.05`, corrected within
each model and causal family across all layer-by-K sets. It is a synthesis rule,
not a claim that all set members are
individually necessary. Establishing irreducibility would require member-wise
leave-one-out tests.

**Section conclusion.** The experiment can support set sufficiency and
specificity; it cannot by itself prove member irreducibility or exclude a
cross-layer/MLP-mediated circuit.
