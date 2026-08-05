# V4.4 full-span top-k sensitivity campaign

## Scope

“Top-k” in this campaign means attention-head banks.  It does not change the
generation sampler's `top_k` parameter or token-position shortlist sizes.

The primary discovery score is

\[
m_{i,h}=\sum_{j\in S_i}\alpha_h(q,j),\qquad
S_h=\mathbb E\left[\left(\sum_i m_{i,h}\right)
\frac{\exp H(m_{1,h},\ldots,m_{N,h})}{N}\right],
\]

where `q` is the answer query and `S_i` is every token in active needle span
`i`.  Endpoint-only mass and first-needle localization are excluded from the
primary ordering.  The source registries are SHA-256 locked in the selection
manifest.

## Experiment A: answer-query ablation dose response

- Models: Qwen3-8B and Gemma4-E4B.
- Frozen K grid: 1, 2, 4, 8, 16, 32 for both models.
- Seeds: 1316--1335; counts: 1--5; 100 prompts per model.
- Intervention: zero the ranked bank's answer-query head outputs.
- Control: three head sets with identical per-layer cardinalities.
- All-sample endpoint: ranked-minus-random absolute generated-count shift.
- Clean-correct endpoint: ranked-minus-random correct-to-wrong probability.
- Uncertainty: 10,000 seed-cluster bootstrap draws.
- Tests: enumerate all `2^20` sign assignments; Holm correction across the
  twelve model-by-K comparisons separately for each primary endpoint.

## Experiment B: Qwen source-bank to L28 path

The same ranking is filtered to layers strictly before L28.  Nested source
banks K=1,2,4,8,16,32 are patched through each existing route definition and
tested against the fixed L28 H16/H19 natural-OV channel.  This uses the
previous V4.4.4 evaluation seeds as a ranking-sensitivity analysis, not a new
independent confirmation.

## Experiment C: Gemma source-bank residual path

The same ranking is filtered to layers strictly before the first candidate
residual mediator (layer <36).  K=1,2,4,8,16 prefixes are evaluated on common
discovery seeds 1630--1639 and confirmation seeds 1640--1659.  Each control is
a deterministic +1/+2/+3 cyclic head rotation within every selected layer,
so layer composition and set size are exact even where large K makes
candidate-control overlap unavoidable.  The path endpoints remain source
transport, exact residual mediation, count-axis mediation, and L41 terminal
adoption.  Global intersection-union p-values are Holm corrected across five
K values.  K=32 is retained in behavioral ablation but excluded here because
the global top-32 contains L41 heads and therefore cannot be a cause of an
L36--40 mediator.

## Isolation and immutable inputs

- FileStream behavior run:
  `/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260805_v4_4_full_span_topk_k1_2_4_8_16_32`
- Qwen path namespace:
  `/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_full_span_topk/qwen_upstream_k_sweep/`
- Gemma path namespace:
  `/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_full_span_topk/gemma_residual_k_sweep/`

Old endpoint-ranked or cue-stability-ranked outputs remain historical inputs;
they are not overwritten or silently relabeled as results from this campaign.
