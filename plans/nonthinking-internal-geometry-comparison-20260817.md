# Non-thinking internal geometry comparison archive (2026-08-17)

## Frozen baseline

- Report: `reports/NiaH_Geometry_Comparison.html`
- Baseline Git commit before this addition: `f626d57` (`Drop low-dimensional transfer aggregate`)
- Registered panel: 10 counts × 30 seeds = 300 trajectories; confirmation is 10 counts × 10 held-out seeds = 100 trajectories.
- Non-thinking running site: prompt needle `span_end` at each observed occurrence `k`.
- Non-thinking final site: `answer_query_v3`, i.e. the prompt-final `:` immediately before generation of the numeric answer; the answer digit itself is not included.

## Decision recorded

Add a supportive within-non-thinking comparison between:

1. prompt needle-end states labeled by running index `k`; and
2. the pre-answer query state labeled by final count `N`.

Both endpoints use discovery-only, seed-grouped layer selection and frozen confirmation evaluation. Each endpoint selects its own best layer for each metric.

This is intentionally not presented as a strict paired contraction estimand. The trajectory panel is shared, but the statistical units differ: running contributes multiple ragged states per trajectory and has triangular class support, whereas the pre-answer endpoint contributes one state per trajectory and has balanced final-count support. Class-balanced metrics reduce the support imbalance but do not make the endpoints identical.

## Frozen supporting evidence (PCA16 covariance geometry)

| model | running SNR | pre-answer SNR | running ordinal RSA | pre-answer ordinal RSA |
|---|---:|---:|---:|---:|
| Qwen3-8B | -4.04 dB @ L19 | -1.78 dB @ L35 | 0.782 @ L0 | 0.977 @ L17 |
| Gemma4-E4B | -6.85 dB @ L39 | -0.75 dB @ L41 | 0.738 @ L11 | 0.960 @ L16 |

For both models, the independently optimized pre-answer endpoint has higher confirmation SNR and higher ordinal RSA. Mahalanobis silhouette and Fisher trace are mixed across models, so the allowed wording is:

> At the pre-answer query boundary, final count is represented with a clearer ordinal organization and a higher between-count / within-count signal ratio than running index at prompt needle ends. This is consistent with partial consolidation or reorganization before answer generation, but it does not establish universal cluster contraction or removal of prompt semantics.

The entity-domain appendix remains important: non-thinking pre-answer states retain strong entity-domain information, so “consolidation” must not be equated with complete filtering.

## Deferred comparison

Do not yet claim a native-thinking running-to-answer consolidation effect. Its answer-side capture requires the planned experiment and an endpoint whose semantics are appropriate for broad retrieval. Add that comparison only after the new native-thinking data are available.

## Metric interpretation recorded

The report must define all six criteria used around this comparison: Logistic balanced accuracy, nearest-centroid balanced accuracy, isotropic SNR, frozen Fisher trace, Mahalanobis silhouette, and held-out ordinal RSA. Each definition includes the exact calculation, practical question, a worked example, and a non-claim boundary.

The mixed Qwen result is retained as an important real case rather than hidden: pre-answer SNR and ordinal RSA rise, while Mahalanobis silhouette and Fisher trace fall. This supports clearer global/ordinal count organization but not universal pointwise cluster tightening. Gemma moves upward on all four covariance-geometry criteria, so its within-non-thinking evidence is broader, while still remaining a supportive endpoint comparison rather than a strict paired contraction estimand.

## Report placement amendment

The main report no longer presents this as a primary section. Appendix D shows the Qwen-only running-index versus pre-answer comparison, because its mixed metric directions make it useful as an interpretive diagnostic rather than a headline result. The Gemma values above remain archived here and in the frozen covariance CSV, but are not used to broaden Appendix D's claim.

Appendix B (native-thinking upper/lower bands) is explicitly labeled descriptive and exploratory. It has no p-value, confidence interval, or seed/trajectory-aware permutation test, and state rows are nested within trajectories. Qwen provides a coherent descriptive pattern consistent with marker/boundary-format offset; Gemma instead follows running position and survives trajectory centering. Therefore Appendix B supports no shared cross-model two-band mechanism and must not be described as statistically significant.
