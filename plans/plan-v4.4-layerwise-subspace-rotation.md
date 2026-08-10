# V4.4 layerwise rank-3 removal and rotation-map extension

## Scientific questions

This extension continues report section 5.4 and separates three questions that
must not be conflated:

1. **Layerwise necessity:** does removing the discovery-frozen rank-3 prompt
   count component from every active needle endpoint damage counting more than
   an exactly norm-matched orthogonal removal, and how does that specificity
   vary with prompt layer?
2. **Layerwise local sufficiency:** does a discovery-frozen transport-aligned
   answer-query perturbation cross each tested adjacent layer boundary more
   strongly than an equal-norm orthogonal perturbation, and how does this
   causal transport vary with depth?
3. **Coordinate change:** can adjacent-layer count coordinates be related by a
   stable linear map, and which part of that map is rotation versus anisotropic
   stretch?

The unit of inference is always a confirmation seed after averaging the
registered counts or directed count pairs. PCA pictures are descriptive and
are never substituted for causal endpoints.

## Frozen data split

- Geometry/map fitting: discovery seeds 1234--1253 only.
- Causal evaluation: confirmation seeds 1254--1263 only.
- Prompt removal counts: 2--10.
- Answer-query transport pairs: 1->2, 2->1, 5->6, 6->5.
- Rank: 3 primary; ranks 1 and 2 are CPU-only sensitivity analyses.

## Experiment A: landmark prompt-layer removal

The primary intervention is the existing `actual_rank3_remove`, now refitted
independently at each registered prompt layer using discovery centroids. At a
given layer, all active needle endpoints are modified simultaneously:

`H' = H - (H - rowmean(H)) U_l U_l^T`.

The matched control uses a rank-3 basis from within-count residual variation,
orthogonalized against `U_l`. Its removed tensor is rescaled separately for
every prompt so that its Frobenius norm equals the candidate removed tensor to
numerical tolerance. Clean generation is computed once per prompt and reused
across layers.

Norm matching is audited after the replacement is quantized back to the model
residual dtype. Because a very small bf16 intervention has discrete realizable
norms, the control scalar is searched and the closest value must be within
2.5% of the candidate norm; the realized ratio remains an output.

Outcome-blind landmark layers:

- Qwen3-8B: 0, 4, 8, 12, 16, 20, 24, 28, 32, 35.
- Gemma4-E4B: 0, 4, 8, 9, 12, 16, 20, 24, 28, 32, 36, 40, 41.

The uniform four-layer grid characterizes depth; L8/L9 preserve the prior
anchor, and the final layer is included explicitly. The primary endpoint is
rank-3-minus-orthogonal absolute-error increase over all samples. Accuracy
drop and clean-correct-only analyses are secondary. Holm correction is applied
within model and endpoint across the registered layers. A seed-level linear
depth slope is reported, but the curve is not assumed monotone.

## Experiment B: landmark adjacent-layer transport causal sweep

For each source layer `l` and target layer `l+1`, discovery source centroids are
ridge-regressed onto the target rank-3 count coordinates. QR of the regression
weights freezes a source transport basis `B_l`. For receiver `r` and donor `d`:

`delta_l(r,d) = B_l B_l^T (mu_l,d - mu_l,r)`.

At the answer-query position, conditions are aligned dose 1, aligned dose 2,
and an equal-norm direction in the orthogonal complement of `B_l`. The primary
endpoint is target-layer donor fraction. Donor-vs-receiver log-odds is
secondary. The required causal contrasts are aligned-1 minus orthogonal and
aligned-2 minus aligned-1.

Planned fp32 and realized model-dtype patch norms are both recorded. The
closest quantized control must be within 2.5% of the realized aligned 1x norm,
and the realized 2x/1x norm ratio within 2.5% of 2.

Outcome-blind boundaries:

- Qwen3-8B: 0->1, 3->4, 7->8, 11->12, 15->16, 19->20, 23->24,
  27->28, 28->29, 31->32, 34->35.
- Gemma4-E4B: 0->1, 3->4, 7->8, 8->9, 11->12, 15->16, 19->20,
  23->24, 27->28, 31->32, 35->36, 36->37, 39->40, 40->41.

The known L28->L29 and L36->L37 effects are prior anchors, not independent
replications. Holm correction is applied within model, metric, and contrast
across all registered boundaries. Dense follow-up is allowed only after this
landmark sweep and must be labeled exploratory unless evaluated on fresh
seeds.

## Experiment C: linear map, rotation, and stability

For each role and adjacent layers, let centered discovery count centroids have
rank-3 bases `U_l` and `U_{l+1}`, with coordinates `Z_l` and `Z_{l+1}`. Fit

`A_l = argmin_A ||Z_l A - Z_{l+1}||_F^2 + lambda ||A||_F^2`,

where `lambda = 1e-3 * trace(Z_l^T Z_l) / 3`. The right polar decomposition

`A_l = R_l S_l`

separates an orthogonal factor `R_l` from positive-semidefinite stretch `S_l`.
Because PCA coordinates have a sign/rotation gauge, raw Euler angles are not a
primary result. Bootstrap bases are first Procrustes-aligned to the all-
discovery reference basis. Primary stability measures are:

- seed-grouped cross-validated coordinate R2 and normalized RMSE;
- rotation-invariant centroid RDM Spearman correlation;
- source/target subspace principal angles;
- bootstrap relative Frobenius dispersion of the gauge-aligned map;
- bootstrap geodesic dispersion of the proper-rotation factor;
- stretch singular values and condition number;
- count-pairing permutation control.

This analysis is run for every adjacent layer of `prompt_running` and
`answer_query`. It is descriptive unless paired with Experiment B at the same
boundary.

## Output and audit rules

- Large hidden states remain on `/lambda/nfs` and are not committed.
- GPU rows are append/resume safe and may be sharded by model/layer.
- Every output records resolved layers, seeds, counts/pairs, rank, model
  revision, elapsed time, command, and norm-matching diagnostics.
- Missing cells, duplicate cells, non-finite values, incomplete conditions, or
  norm mismatch fail the analysis rather than being silently dropped.
- Report section 5.4 is updated only after both analyzers emit `PASS` audits.
