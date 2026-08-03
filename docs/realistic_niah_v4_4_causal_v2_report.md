# Realistic NIAH V4.4 causal-v2 formal report

The primary artifact is the standalone Chinese HTML report:
[`reports/realistic_niah_v4_4_causal_v2_report.html`](../reports/realistic_niah_v4_4_causal_v2_report.html).
It integrates the two completed formal runs, rather than mixing them into the
earlier V4.4 representation report.

## Scope and status

The report covers the formal Qwen3-8B and Gemma4-E4B V4.4 numeric
non-thinking causal-v2 campaigns for `k={1,3,5}`. Both campaigns are complete
and each passed the strict audit with 302/302 checks and zero errors. The
implementation commit is `dd409f2dff82ccd6400dfc3d7704025cb6939940`.

**本节结论：** This is completed, audited evidence. It supersedes statements
in the older mechanism-report companion that causal-v2 had not yet run; it
does not supersede that report's separate representation analyses.

## Frozen design

- Counts are 0 through 10.
- The nine unordered count pairs are `(0,1)/(4,5)/(9,10)` for `k=1`,
  `(0,3)/(3,6)/(7,10)` for `k=3`, and `(0,5)/(2,7)/(5,10)` for `k=5`.
  Both directions are run, for 18 directed pairs.
- Centroids use seeds 1234--1253; screening uses 1254--1258; confirmation
  uses the disjoint held-out seeds 1259--1263.
- Prompt and answer patching use `single_layer` and
  `cumulative_from_layer`. Prompt sites are `toggled_needle_end` and
  `toggled_needle_span`; answer patching uses the final `Total:` query.
- Steering uses alpha 1 count-centroid deltas and a norm-matched orthogonal
  random direction. Its confirmation also includes a screen-frozen
  multi-layer plan for each k.
- Selection requires at least 4/5 positive screen seeds, positive effects in
  both directions, at least 2/3 positive anchor pairs, mean adjusted
  transport at least 0.15, and valid rate at least 0.95.

**本节结论：** Confirmation estimates are held out from layer/condition
selection. Empty layers in a confirmation plot mean "not selected and not
confirmed," not zero effect.

## Prompt full-span alignment

The frozen policy is
`monotonic_endpoint_preserving_nearest_neighbor_v1`. For receiver length R
and donor length S, with R,S > 1, receiver index j uses

```text
a(j) = floor((2*j*(S-1) + (R-1)) / (2*(R-1))).
```

This is deterministic round-half-up, monotone, endpoint preserving, and the
identity when R=S. Qwen preflight is 540 exact / 0 remapped / 0 unsupported;
Gemma is 178 exact / 362 remapped / 0 unsupported.

**本节结论：** Both models satisfy the registered support requirement. The
Gemma estimates legitimately include deterministic remapping, so tokenizer
alignment remains a boundary in absolute cross-model comparisons.

## Estimand and inference

With receiver gold count `r`, target count `t`, clean receiver prediction
`y0`, and treatment prediction `y1`, strict normalized transport is

```text
T = (y1 - y0) / (t - r).
```

Target conformity is `1 - |y1-t|/|t-r|`. Invalid numeric generations remain
in the denominator as strict zero effect. Patching subtracts the within-example
mean of self-copy and same-count controls; steering subtracts the norm-matched
orthogonal random control. Paired effects are averaged within seed, then over
seed clusters. Confidence intervals use 10,000 seed-cluster bootstraps; exact
two-sided sign-flip p-values are Holm-corrected within evidence scope.

With only five held-out clusters, the smallest possible two-sided exact p is
`2/2^5 = 0.0625`. No primary condition can therefore attain p < .05 even when
all five seed effects are positive.

**本节结论：** The positive bootstrap intervals are stable descriptive and
resampling evidence, but they do not replace the preregistered exact test. The
HTML report explicitly reports zero Holm-significant primary conditions.

## Results

### Prompt patching

Only full-span conditions pass screening: 126 Qwen and 102 Gemma conditions.
All have positive held-out bootstrap lower bounds and all five held-out seed
effects are positive. No endpoint condition passes screening.

**本节结论：** A coordinated full-needle-span state is sufficient for count
transport under the tested interventions; one endpoint token is not shown to
be sufficient. This is not a necessity result.

### Answer-query patching

There are 149 Qwen and 177 Gemma primary conditions. All have positive
bootstrap lower bounds and 5/5 positive held-out seeds. Cumulative clamping is
not used for layer localization because it overwrites all downstream layers.

**本节结论：** The final answer-query residual is a strong sufficient carrier
of the donor count direction. It can transport an incorrect donor prediction,
so this is not identical to transporting the donor gold count.

### Steering

All 45 Qwen conditions and 53/54 Gemma conditions have 5/5 positive held-out
seeds; every bootstrap lower bound is positive. The outcome is directional
transport, not exact set-to-count control, and multi-layer plans do not
establish an advantage over single layers.

**本节结论：** A population-level count direction causally controls the tested
answer-query states relative to an orthogonal norm-matched control, within the
registered alpha and layer plans.

### Head-bank ablation

The top-1--32 sweep is discovery-only. Ranked-minus-random accuracy and error
effects are non-monotone and can reverse sign across models and banks.

**本节结论：** The sweep does not support a stable cross-model necessity claim
for either ranked bank. It is exploratory candidate-localization evidence.

## Baseline boundary

Both models produce valid numeric completions on all 330 baselines. Held-out
confirmation accuracy is 49.1% for Qwen and 42.7% for Gemma, with systematic
under-counting at high gold counts.

**本节结论：** Intervention transport must not be paraphrased as universally
correct counting. The report keeps validity, accuracy, signed error, and
transport as separate quantities.

## Machine-readable outputs

The directory `reports/v4_non-thinking_causal/v4_4_causal_v2/` contains:

- primary condition, family, and protocol-by-k confirmation tables;
- baseline-by-split and baseline-by-count tables;
- prompt-alignment, selection, stage, ablation, and audit summaries;
- local/FileStream archive verification;
- `source_ledger.csv`, with size and SHA-256 for every report input;
- `report_summary.json`, used by tests for headline-value traceability.

**本节结论：** Displayed numbers are generated from the exported machine
tables. The checked report is not maintained by manually editing its values.

## Rebuild

```bash
python scripts/build_realistic_niah_v4_4_causal_v2_report.py \
  --qwen-run-root /path/to/qwen/run/Qwen3-8B/numeric/causal_v2 \
  --gemma-run-root /path/to/gemma/run/Gemma4-E4B/numeric/causal_v2 \
  --qwen-export /path/to/Realistic_CoT_NiaH_Count_20260803_v4_4_causal_v2_qwen \
  --gemma-export /path/to/Realistic_CoT_NiaH_Count_20260803_v4_4_causal_v2_gemma \
  --output reports/realistic_niah_v4_4_causal_v2_report.html \
  --data-dir reports/v4_non-thinking_causal/v4_4_causal_v2
```

Run the causal-v2 audit on each source run before rebuilding. Then run the
report test and open the HTML in a browser at desktop and mobile widths.

**本节结论：** Reproduction requires the two small causal-v2 exports, not a
new download of the pre-existing 32 GB corpus.
