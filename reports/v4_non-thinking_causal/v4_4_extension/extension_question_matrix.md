# V4.4 non-thinking extension question matrix

This file freezes how every question in `reports/non-thinking extension.md` is
answered.  A negative or scope-limited result is still an answer; no item is
silently omitted.

| Step | Question | Primary evidence | Inference / boundary |
|---|---|---|---|
| 1 | hidden-state rank and rank-k variance | `analysis/geometry/rank_and_compression_by_layer.csv` | sample-matrix stable/effective/numeric rank is kept distinct from rank of the ten count centroids |
| 1 | linear/nonlinear regression R2 | `analysis/geometry/count_regression_summary.csv` | discovery-to-confirmation for prompt; seed-grouped CV for answer |
| 1 | count classification | six fixed algorithms for every layer in `classification_prompt_*` and existing `classification_all_*` | no per-layer algorithm cherry-picking; one global fixed primary is reported and all curves are shown |
| 1 | cluster quality | `analysis/geometry/clustering_summary.csv` | PCA fit on train rows, silhouette/CH/DB evaluated on held-out rows |
| 1 | source of prompt noise | `prompt_noise_two_way_decomposition.csv` | balanced Frobenius decomposition into count, seed/context, and count-by-seed deformation; does not pretend to identify lexical causality |
| 1 | all-token projection and empirical formula | `analysis/all_token/*` | discovery endpoint basis applied to held-out endpoint/interior/hard-negative/ordinary tokens; compares endpoint-gated, span-gated, and ungated formulas |
| 1 | cue present vs absent | existing V4.4.2 pooled-basis full-space tests | topology is preserved but the full state is modulated; causal path remains cue-present only |
| 1 | induction / earlier-span attention | `analysis/endpoint_attention_mask/earlier_span_head_confirmation.csv` | discovery ranks prior full-span mass; confirmation compares same-depth, same-length ordinary spans; “induction-like aggregation” only if it passes |
| 1 | suppress outside-needle attention | `analysis/endpoint_attention_mask/attention_mask_statistics.csv` | strict needle-only query mask versus equal-key-budget non-needle mask at occurrences 2/4/6/8/10 |
| 1 | needle token corruption | `analysis/token_corruption/*` | replace every active needle token by ordinary passage tokens; equal-token-budget ordinary corruption is the matched control; all and clean-correct-only reported |
| 1 | hidden-state / subspace ablation | existing single-endpoint null plus `analysis/prompt_subspace_ablation/*` | new intervention acts jointly on every active endpoint and removes a discovery-frozen rank-3 component; equal-norm orthogonal control |
| 2 | broad retrieval attention and head necessity | existing full-span literal ranking and fresh top-K matched ablation | full-span, not endpoint-key, is the primary retrieval definition |
| 2 | delayed OV and residual identity | existing Qwen natural pre-O OV, Gemma source-bank residual mediation, plus transport-aligned adjacent-layer patch | claims are localized to the tested set/layers; prompt and answer axes need not be parallel |
| 3 | answer geometry, rank, compression, classification, clustering | existing answer PCA and patching plus new layerwise geometry/classification tables | all-sample primary with correct-only classifier/patching sensitivity |
| 3 | running-to-answer-to-output causal link | existing full-span upstream paths, answer full-state patching, natural OV/residual mediation | local answer relay is confirmed; the old final prompt endpoint is not asserted as a direct source |
| 3 | layerwise consolidation | new rank/compression/regression/classification curves plus existing causal write traces | “consolidation” means increasing held-out decodability / lower noise and executable late-state effects, not visual PCA compression alone |
