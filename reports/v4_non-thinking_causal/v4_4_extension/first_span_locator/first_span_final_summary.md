# V4.4 complete-first-span locator: final audit summary

## Question and definitions

At the final `Total:` answer query, the absolute attention mass of head `h` on needle span `i` is

`m_i,h = sum_{j in S_i} alpha_h(q,j)`.

Heads were frozen on N=10 discovery examples (seeds 1234--1253) by descending mean `m_1`. We also report `M10 = sum_i m_i` and `share1 = mean(m1) / mean(M10)`. Causal confirmation used fresh seeds 1336--1355 and counts 1--5 (20 seed clusters; 100 prompts/model). For K in {1,2,4,8,16,32}, selected answer-query pre-O head slices were zeroed. Positive ranked-minus-control means that first-span-ranked heads damage counting more than the control set.

## Representation

- Qwen top head: L9H19. `m1=0.272003`, `M10=0.495591`, `share1=0.548847`. Ten-span profile: `[0.272003, 0.080269, 0.047466, 0.033059, 0.020520, 0.014325, 0.013559, 0.005590, 0.005224, 0.003576]`.
- Gemma top head: L11H6. `m1=0.040711`, `M10=0.057652`, `share1=0.706158`. Ten-span profile: `[0.040711, 0.007719, 0.002388, 0.001577, 0.001311, 0.000905, 0.000955, 0.000718, 0.000694, 0.000673]`.

Both models therefore contain heads with strong first-span concentration as an observational attention phenotype.

## Layer-matched causal control

No Qwen K was significant for either endpoint after within-model/endpoint Holm correction. Gemma K=32 was significant only in the opposite direction: ranked-minus-control absolute-error increase `-0.2033` (Holm `p=0.00423`) and correct-to-wrong excess `-0.1269` (Holm `p=0.0363`). Thus first-span ranking did not identify a set more necessary than layer-matched heads.

## Layer + M10-nearest causal control

No positive first-span-specific effect was significant. Qwen had no significant K. Gemma again showed significant effects only in the opposite direction: K=16 absolute-error difference `-0.1467` (Holm `p=0.00532`); K=32 absolute-error difference `-0.2333` (Holm `p=0.00135`) and correct-to-wrong difference `-0.1714` (Holm `p=0.00738`).

The M10 match is nearest-neighbor rather than exact. For Qwen it is weak because L9H19 is an extreme outlier (`M10=0.4956` versus K=1 control mean `0.0651`); therefore this branch cannot cleanly isolate Qwen first-span concentration from total retrieval strength. Gemma matching is materially closer at small K. Gemma ranked/control overlap is zero through K=8, 12.5% at K=16, and 37.5% at K=32; large-K estimates are consequently conservative and partially dependent.

## Conclusion

The representation result is positive: both models have heads whose answer-query attention is strongly concentrated on the complete first needle span. The causal specificity result is negative: across two control families, ablating heads selected by first-span absolute mass does not damage counting more than matched alternatives. The current evidence therefore supports a first-span attention phenotype, not a uniquely necessary first-locator circuit. This negative result should be reported briefly and should not be used as a core mechanism claim.

## Artifacts

- `discovery_analysis/*first_span*representation*.csv`: all-head and ranked representation tables.
- `full/analysis/first_span_ablation_statistics.csv`: layer-matched statistics.
- `m10_full/analysis/first_span_M10_matched_statistics.csv`: layer+M10-nearest statistics.
- `m10_full/analysis/first_span_M10_matching_audit.csv`: replicate-level matching and overlap audit.
- `m10_full/analysis/first_span_M10_matching_summary.csv`: K-level matching summary.
