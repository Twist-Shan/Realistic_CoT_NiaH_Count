# Realistic NIAH V4.4.3: OV vertical-geometry causal test

## 1. Scientific question

V4.4.3 tests the conjecture in `realistic_niah_v4_ov_vertical_geometry_report.pdf`:
the prompt running-index trajectory and the answer-count trajectory may encode the
same count variable in different residual subspaces because one or a few attention
heads read a prompt-count direction through `V` and write it into an answer-count
direction through `O`.

For query head `h` in layer `l`, with its grouped-query KV head `g(h)`, the fitted
mapping is

`M_OV^(l,h) = W_O^(l,h) W_V^(l,g(h))`.

The registered score is

`r_(l,h) = cos(M_OV^(l,h) u_prompt^(l), u_answer^(l))`.

All layer/head indices are zero based, matching the existing V4 artifacts and
the `L29H3` notation in the report.

## 2. Frozen evidence split

- Models: `Qwen3-8B`, `Gemma4-E4B`.
- Condition: numeric, non-thinking V4.4 only.
- Discovery direction fit: seeds 1234--1253.
- Count split used inside discovery:
  - fit/ranking counts: 1, 3, 5, 7, 9;
  - held-out geometry counts: 2, 4, 6, 8, 10.
- Staged patch screen: seeds 1254--1258.
- Directed removal/injection confirmation: seeds 1259--1263.
- Candidate output layers:
  - Qwen: 28, 29, 30;
  - Gemma: 36, 37, 38.
- One fit-score-selected head per layer. Qwen `L29H3` is retained as a
  preregistered sentinel even if it is not selected.

Held-out-count geometry is reported but is not used to choose heads. Screen and
confirmation seeds are never used to fit directions or select mapping heads.

## 3. Test 1: geometry mapping

For a candidate head in output layer `l`:

1. Load the frozen N=10 prompt endpoint states from post-block layer `l-1`.
2. Apply the actual block-`l` pre-attention RMSNorm. This produces the correct
   input space for `W_V`; multiplying an unnormalized residual direction by
   `W_V` would be the wrong estimand.
3. Fit a within-seed OLS count slope. Per-seed centering prevents passage/seed
   offsets from masquerading as a count direction.
4. Fit the answer direction from the frozen post-block layer-`l` answer-query
   states.
5. Respect grouped-query attention: query heads sharing a KV head use the same
   `W_V` slice but different `W_O` column slices.
6. Report fit-count and held-out-count mapping cosine, mapped-vector norm,
   same-layer empirical rank, norm-matched empirical rank, and same-KV-group
   empirical rank.

Gemma4 has an additional architectural constraint: output layers 36--38 do not
own V matrices. In the pinned `google/gemma-4-E4B-it` checkpoint,
`num_kv_shared_layers=18`, so these sliding-attention layers reuse values
produced by layer 22; those values pass through a per-head `v_norm`. The Gemma
estimand therefore follows the actual graph
`L22 input -> V22 -> v_norm -> shared value -> O_l`
while retaining answer direction and intervention boundary at output layer
`l in {36,37,38}`. The provider is resolved from the runtime model graph rather
than hard-coded. The value-space count slope is fit empirically after
`v_norm`; treating the missing target-layer `v_proj` as an identity or silently
moving the target output layer would be incorrect.

The control head paired to each candidate is in the same layer and is selected
by closest mapped-vector norm with a near-median mapping score. This pairing is
frozen before any causal result is read.

## 4. Test 2: staged patch

Donor/receiver count pairs are `(1,6)`, `(3,8)`, `(5,10)` in both directions.
For each candidate and its matched control, V4.4.3 applies:

1. `alpha_receiver_v`: donor attention pattern with receiver values;
2. `alpha_position_scramble`: the aligned donor row is circularly scrambled
   across causal-visible key positions while preserving total mass and the
   self-key;
3. `z_donor`: donor pre-O aggregate replaces receiver `z_h`;
4. `o_donor`: donor isolated post-O head output replaces receiver `o_h`;
5. `output_norm_control`: a deterministic random residual direction with the
   same norm as the donor-output delta.

Qwen pairs normally have exact token-index alignment. Gemma may tokenize an
active/control slot to a slightly different width. Attention rows are therefore
piecewise aligned using all ten registered slot boundaries before they are
combined with receiver values. The aligned row preserves attention mass.
The one-token eager-cache attention path is compared against the full SDPA
prefill on the unique first-count-token logits. Both raw discrepancy and the
discrepancy after removing the common shift over count-token logits are saved.
The registered 0.5 tolerance is applied to the centered discrepancy because a
common shift of all count candidates cancels from expected count and count
margin; using its raw magnitude as a causal failure criterion would be
incorrect. Formal staged-patch shards retain both audits.

`z_h` and `o_h = W_O z_h` are connected by a fixed linear map. Replacing donor
`z_h` before `W_O` and adding the corresponding donor-minus-receiver `o_h`
after `W_O` should therefore be numerically equivalent. V4.4.3 treats their
maximum count-logit discrepancy as an implementation audit (tolerance 0.05),
not as two statistically independent pieces of evidence.

Primary patch outcome: change in expected count over the ten registered numeric
candidates, normalized by the donor-receiver count shift. Candidate probabilities
use the frozen V4 joint score: summed log probability of the complete answer plus
chat-termination sequence. This is necessary because `10` is two tokens in both
registered tokenizers.

## 5. Test 3: directed causal intervention

On held-out confirmation seeds and gold counts 2, 5, and 8:

- remove the candidate head output component parallel to `u_answer`;
- remove an equal-norm component along a deterministic direction orthogonal to
  `u_answer`;
- inject `beta * s_l * u_answer`, where `s_l` is the discovery answer-count OLS
  step norm and `beta` is `-2, -1, -0.5, 0, 0.5, 1, 2`.

The injection is implemented at the layer post-O residual, not inside a
particular attention head. Candidate and matched-head rows within the same
layer are deliberately duplicated so that the output schema remains
rectangular; their exact equality is an implementation audit. Consequently,
injection tests whether the layer-level answer subspace is causally steerable,
but it cannot localize that steerability to the registered head.

The registered directional predictions are:

- injection has a positive expected-count slope in `beta` and Spearman
  correlation above 0.5;
- answer-direction removal increases expected-count absolute error and reduces
  the correct-count margin more than the equal-norm orthogonal removal;
- the head-specific staged-patch and removal effects are stronger for the
  mapping-selected head than for its same-layer norm-matched head.

V4.4.3 also records KL divergence on the vocabulary after excluding the unique
first-token IDs used by the ten count candidates. This is a local answer-query
specificity control. It is not a
complete generic-language benchmark, and the final report must preserve that
boundary.

## 6. Claim rule

The frozen directional rule records a candidate as jointly passing if all of
the following hold:

1. fit-count and held-out-count mapping cosines are positive;
2. donor-Z normalized transport exceeds the equal-norm output control;
3. signed injection has positive slope and monotonic direction;
4. answer-direction removal is more harmful than equal-norm orthogonal removal.

Alpha-vs-scramble is reported separately as evidence about QK localization.
It is not required for the OV writeback rule.

Because item 3 is a layer-level rather than head-level intervention, a frozen
joint pass is not by itself a statistically specific single-head result. The
final single-head claim additionally requires the selected-minus-matched
contrast to have the predicted sign for donor-Z transport and both removal
outcomes, with each seed-level one-sided exact sign-flip test at `p <= 0.05`.
These exact tests are reported without multiplicity correction; if the joint
criterion already fails before correction, correction cannot reverse that
conclusion.

Failure of every registered head does not show that OV is irrelevant. It rejects
the narrower explanation that a single registered head near the reported layer
is sufficient. The next explanation to test is a small head set, MLP-mediated
re-encoding, or a multi-layer distributed circuit.

## 7. Filestream isolation and restart contract

Frozen input (read only):

`/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260731_v4_numeric_presentation_v3`

Exclusive output namespace:

`/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_3_ov_causal/<run_id>`

The V4.4.3 instance uses a local code/runtime copy at
`/home/ubuntu/v443_ov_causal`. It never edits the shared `repo`, the frozen
V4.4 run, or any `run_20260803_v4_4_causal_v2_*` directory. Every causal unit
is a gzip CSV shard written to a temporary file in the destination directory
and installed with `os.replace`. `--resume` accepts only a byte-identical
resolved config and skips validated shards.

Raw attention rows and full hidden states are never persisted. Existing frozen
hidden states remain in the source filestream; V4.4.3 writes directions, scalar
head scores, compact intervention outcomes, manifests, logs, analysis tables,
and the final report.

## 8. Execution

From the isolated instance copy:

```bash
tmux new-session -d -s v443_ov \
  bash /home/ubuntu/v443_ov_causal/repo/scripts/launch_realistic_niah_v4_4_3.sh \
  run_20260803_v4_4_3_ov_causal_a100
```

Resume after an interruption:

```bash
bash /home/ubuntu/v443_ov_causal/repo/scripts/launch_realistic_niah_v4_4_3.sh \
  run_20260803_v4_4_3_ov_causal_a100 --resume
```

The launcher runs an actual-model smoke test before formal shards. Final
completion requires both model completion files, exact dynamic shard counts,
analysis JSON files, the Markdown report, and a passing `audit.json`.
