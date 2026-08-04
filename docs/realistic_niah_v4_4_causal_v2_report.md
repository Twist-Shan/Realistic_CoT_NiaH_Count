# Realistic NIAH V4.4 causal-v2 integrated report

The primary artifact is the standalone Chinese HTML report:
[`reports/realistic_niah_v4_4_causal_v2_report.html`](../reports/realistic_niah_v4_4_causal_v2_report.html).
It integrates the original audited Qwen/Gemma causal-v2 campaigns with the
2026-08-04 clean-correct patching and dual-population ablation extension.

## Scope and claim status

The report is intentionally limited to two mechanistic claims:

1. The final answer-query hidden state contains donor-associated correct-count
   information that downstream computation can use.
2. Ranked attention-head banks have measurable functional effects on counting
   behavior relative to layer-matched random-head controls.

It does not claim a unique counting circuit, an explicit integer register, an
exact number of causal heads, an additive head ranking, or a monotone
ablation dose response.

**本节结论：** Clean-correct answer patching is sufficient for the bounded
hidden-state claim. Ablation supplies fresh-seed discovery evidence for head
function, but a particular reusable bank is not yet independently confirmed
because `n` was inspected on the same seeds used to select the candidate.

## Audited inputs

- Original Qwen causal-v2: 302/302 checks, zero errors.
- Original Gemma causal-v2: 302/302 checks, zero errors.
- Correct-interventions extension: 98/98 checks, zero errors.
- Original causal implementation: `dd409f2dff82ccd6400dfc3d7704025cb6939940`.
- Correct-interventions execution commit:
  `cda0d092db424d4bcb712a1402b899df1bee793b`.
- Correct-interventions definition SHA-256:
  `6f7f7760f53a2bab08e5b840aa765dbf70d853a75952eabb5282d108b4315f5e`.
- Qwen/Gemma extension stage design hashes: `4c3cdeb48cbf` and
  `d419daff86de`.
- The extension inventory independently verified 798/798 file size+SHA
  entries and 180/180 gzip files.

**本节结论：** Every displayed result is read from an audited CSV/JSON source;
the report generator writes a source path/size/SHA ledger.

## Clean-correct patching design

An eligible pair requires both the clean receiver and clean donor generation
to equal their own gold count. The original screen-selected patch conditions
are retained, and patching is then evaluated only on eligible pairs. Each
model × `k={1,3,5}` × direction group targets five seed clusters.

Initial shortages and deterministic supplements were:

- Qwen: `k=3` lacked one seed cluster per direction and `k=5` lacked three
  per direction. Added pair seeds were 1274, 1275, 1276, and 1278.
- Gemma: `k=5` lacked four seed clusters per direction. Added pair seeds were
  1275, 1277, 1281, and 1295.
- All twelve model × k × direction groups reached 5/5 clusters; no shortage
  remained.

The stopping rule scans the ordered reserve-seed prefix and stops when all
predeclared quotas are met. It does not inspect intervention outcomes.

**本节结论：** Supplementation repaired coverage, not effect size. The exact
missing support and added seeds are machine-readable in
`supplement_seed_summary.csv`.

## Patching estimand

For patch pair-condition instance `i`, with donor gold `d_i` and patched
numeric output `y_i`, strict donor-target accuracy is

```text
PatchAcc = (1/N) * sum_i 1[patched output is valid and y_i = d_i].
```

Invalid generations remain in the denominator and contribute zero. Each
`k × direction` estimate is pair-condition-instance weighted. Confidence
intervals use 10,000 seed-cluster bootstrap repetitions. The report's pooled
average is the descriptive ratio of total successes to total denominator
across the six groups; it is not assigned a synthetic pooled confidence
interval.

Headline pooled values:

- Qwen prompt: 1161/1424 = 81.53%.
- Gemma prompt: 1000/1088 = 91.91%.
- Qwen answer query: 1628/1686 = 96.56%.
- Gemma answer query: 1809/1884 = 96.02%.

**本节结论：** Answer-query replacement very reliably sets a clean-correct
receiver to the clean-correct donor's gold count. This supports downstream-
usable correct-count information, not an explicit scalar code or unique
circuit.

## Dual-population ablation

Both populations use counts 1–5, seed clusters 1274–1283, the same ranked
broad-aggregation banks, and at least three layer-matched random controls.
The clean-correct population is an exact subset of the all-example output.

For all examples, with clean prediction `y0`, ranked-ablation prediction
`yR(n)`, and random-control predictions `yj(n)`, the primary magnitude effect
is

```text
D_abs(n) = E|yR(n)-y0| - mean_j E|yj(n)-y0|.
```

The signed counterpart is also reported to show whether the count shifts up
or down. For clean-correct examples (`y0 = gold`), the primary endpoint is

```text
D_cw(n) = P(yR(n) != gold | y0 = gold)
          - mean_j P(yj(n) != gold | y0 = gold).
```

Positive values mean ranked-head ablation is more disruptive than random
ablation. Effects are clustered by seed and use 10,000 bootstrap repetitions.
The main comparison is `n=1..5`; `n=6..32` remains an explicitly exploratory
diagnostic and is not used for candidate selection in the report.

The largest main-range effects are:

- Qwen all examples: `n=2`, `D_abs=0.133`, CI `[0.013, 0.300]`, overlap 0.
- Qwen clean-correct: `n=4`, `D_cw=0.083`, CI `[0.023, 0.158]`, mean overlap 1.
- Gemma all examples: `n=1`, `D_abs=0.147`, CI `[0.060, 0.233]`, overlap 0.
- Gemma clean-correct: `n=1`, `D_cw=0.102`, CI `[0.000, 0.205]`, overlap 0.

The Gemma clean-correct lower bound equals zero, so it does not strictly
exclude zero.

**本节结论：** Model/population-specific candidates are Qwen 2/4 and Gemma
1/1. Qwen has no single `n` that maximizes both estimands. These are
discovery candidates, not frozen confirmation results.

## Non-monotonicity and random overlap

Non-monotonic effects prevent additive ranking and dose-response claims. They
do not invalidate a predeclared point effect. Random overlap may attenuate a
contrast in an ideal shared-removal setting, but it is not guaranteed to be
conservative because it also changes the control intervention and variance.

**本节结论：** The Qwen clean-correct `n=4` effect remains observed evidence,
but a clean confirmation should use random controls with zero overlap.

## Minimal confirmation if a reusable bank claim is needed

Before observing new outcomes, freeze:

- Qwen all-example `n=2`;
- Qwen clean-correct `n=4`;
- Gemma all-example `n=1`;
- Gemma clean-correct `n=1`.

Use a completely untouched seed suffix, retain counts 1–5, use at least 10
(preferably 20) seed clusters per model, and enforce at least three
layer-matched random banks with zero overlap. Keep `D_abs` and `D_cw` as the
population-specific primary endpoints and Holm-correct the four frozen tests.
Do not rescan `n`.

If the manuscript only needs the clean-correct head-usefulness claim, confirm
only Qwen `n=4` and Gemma `n=1`; treat the all-example analysis as descriptive
support.

**本节结论：** No further patching is needed for the bounded hidden-state
claim. Only a small frozen-n/new-seed/zero-overlap ablation confirmation is
needed to call a particular head bank reproducible and confirmed.

## Machine-readable outputs

The directory `reports/v4_non-thinking_causal/v4_4_causal_v2/` includes the
original causal-v2 machine tables plus:

- `correct_patching_aggregate.csv`;
- `correct_patching_pooled.csv`;
- `dual_population_ablation_top_n_1_5.csv`;
- `dual_population_ablation_diagnostics.csv`;
- `ablation_candidate_summary.csv`;
- `supplement_seed_summary.csv`;
- `correct_prompt_alignment_summary.csv`;
- `correct_interventions_audit.csv`;
- `correct_interventions_stage_summary.csv`;
- `source_ledger.csv` and `report_summary.json`.

**本节结论：** Headline values can be independently recomputed from the
checked CSV files without parsing the HTML.

## Rebuild

```bash
python scripts/build_realistic_niah_v4_4_causal_v2_report.py \
  --qwen-run-root /path/to/qwen/numeric/causal_v2 \
  --gemma-run-root /path/to/gemma/numeric/causal_v2 \
  --correct-run-root /path/to/run_20260804_v4_4_correct_interventions \
  --qwen-export /path/to/qwen_export \
  --gemma-export /path/to/gemma_export \
  --correct-export /path/to/correct_interventions_export \
  --output reports/realistic_niah_v4_4_causal_v2_report.html \
  --data-dir reports/v4_non-thinking_causal/v4_4_causal_v2
```

Run all source audits before rebuilding, then run the report tests and inspect
the HTML at desktop/mobile widths and in print mode.

**本节结论：** Reproduction needs only the small causal/intervention exports;
it does not require re-downloading the pre-existing 32 GB corpus.
