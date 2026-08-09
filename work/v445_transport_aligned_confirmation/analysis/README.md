# V4.4.5 transport-aligned subspace patch

## Question

The earlier rank-3 source-projected donor patch was nearly inert. This experiment asks whether that result came from (i) a PCA subspace that was descriptive but not aligned with downstream transport, or (ii) patching the wrong source support.

## Discovery-frozen transport basis

For each model and source support, discovery count centroids at the source are paired with discovery count centroids at a downstream answer-query layer. A rank-3 PCA basis is fitted only at the downstream target. Let `Y` be the ten standardized target-centroid coordinates and `X` the ten centered source centroids. Ridge regression fits `X W ≈ Y`; QR factorization of the columns of `W` yields a rank-3 source basis `B_transport`. Thus the source subspace is selected for its ability to predict downstream count coordinates, not for source variance alone.

The matched control is the leading within-count source direction after removing `B_transport`. It is orthogonal to the fitted transport subspace and receives the same source-state perturbation norm as dose 1.

## Intervention

For receiver count `r` and donor count `d`, the aligned source step is

`delta_aligned = B_transport B_transport^T (mu_source[d] - mu_source[r])`.

The source state is replaced by `h_source[r] + beta * delta_aligned`, with `beta=1` or `2`. The orthogonal control uses `h_source[r] + ||delta_aligned|| v_orth`.

Two supports are compared:

- `legacy_endpoint`: the last prompt needle endpoint, post-L27 for Qwen and post-L36 for Gemma.
- `answer_query_relay`: the answer-query position immediately upstream of the target, post-L28→post-L29 for Qwen and post-L36→post-L37 for Gemma.

The primary outcome is downstream target donor fraction:

`<h_patch - h_clean, mu_target[d] - mu_target[r]> / ||mu_target[d] - mu_target[r]||^2`.

A value of 1 means one complete receiver-to-donor centroid step at the downstream target; 2 means approximately two steps. Donor-vs-receiver first-token log-odds is secondary.

## Confirmation design

- Seeds: 1254–1263 (10 held-out confirmation seeds).
- Directed count pairs per seed: 1→2, 2→1, 5→6, 6→5.
- Conditions: aligned dose 1, aligned dose 2, norm-matched orthogonal.
- Rows: 2 models × 10 seeds × 4 pairs × 2 supports × 3 conditions = 480.
- Inference unit: the seed mean across four directed pairs.
- Confidence interval: 50,000-draw nonparametric bootstrap over 10 seed means.
- P value: exact two-sided sign-flip test over 10 seed-mean paired contrasts.

## Results

| Model | Support | Condition | Downstream donor fraction | Donor log-odds gain | Argmax changed |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | answer-query relay | aligned dose 1 | 0.949 | +15.263 | 82.5% |
| Qwen3-8B | answer-query relay | aligned dose 2 | 1.810 | +17.206 | 92.5% |
| Qwen3-8B | answer-query relay | matched orthogonal | 0.007 | +3.306 | 25.0% |
| Qwen3-8B | legacy endpoint | aligned dose 1 | 0.00021 | -0.013 | 0% |
| Qwen3-8B | legacy endpoint | matched orthogonal | 0.00012 | -0.013 | 0% |
| Gemma4-E4B | answer-query relay | aligned dose 1 | 0.976 | +10.728 | 67.5% |
| Gemma4-E4B | answer-query relay | aligned dose 2 | 1.801 | +13.428 | 77.5% |
| Gemma4-E4B | answer-query relay | matched orthogonal | 0.002 | +1.380 | 2.5% |
| Gemma4-E4B | legacy endpoint | aligned dose 1 | 0.000 | 0.000 | 0% |
| Gemma4-E4B | legacy endpoint | matched orthogonal | 0.000 | 0.000 | 0% |

For downstream donor fraction, the dose-1 aligned-minus-orthogonal contrast is:

- Qwen: 0.942, seed-bootstrap 95% CI [0.913, 0.967], exact two-sided p=0.00195.
- Gemma: 0.974, seed-bootstrap 95% CI [0.962, 0.986], exact two-sided p=0.00195.

Dose 2 adds a further 0.861 [0.842, 0.879] donor step for Qwen and 0.825 [0.818, 0.833] for Gemma; both p=0.00195. The legacy endpoint dose-1 contrast is not significant (Qwen p=0.607; Gemma is exactly zero, p=1).

## Supported conclusion

Both models carry a direction-specific, approximately dose-responsive count channel in the answer-query residual across the tested adjacent layer boundary. The old source-projected null was therefore not evidence that every low-rank subspace patch must fail: it combined a non-transport-aligned PCA basis with a prompt endpoint support that is causally disconnected from the downstream answer-query state under single-token patching.

This experiment does **not** establish direct transport from the last prompt endpoint to the answer. It establishes local answer-query residual transport. Connecting the prompt running counter to this relay still requires a distributed/span-level or path-specific intervention upstream.
