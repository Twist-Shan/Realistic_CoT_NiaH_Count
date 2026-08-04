# V4.4 correct-pair patching and dual-population ablation

## Purpose and relationship to V4.4

This is an additive confirmation extension. It does not overwrite, relabel, or
reselect any result in the audited V4.4 causal-v2 runs. Prompt/answer patching
keeps the condition set selected by the original five-seed overall screen.
Head ablation keeps the independently frozen model-specific intervention sizes:
Qwen3-8B broad-aggregation top-8 and Gemma4-E4B broad-aggregation top-6.

The extension answers two narrower questions:

1. When both clean examples are answered correctly, does residual patching move
   the receiver to the donor's gold count?
2. What is the effect of the frozen head bank (a) over all registered examples
   and (b) conditional on a clean-correct baseline?

No result from this extension is evidence that the selected heads constitute a
unique counting circuit.

## Correct-pair patching population

For model (m), seed (s), receiver count (r), and donor count (d), a
directed patch pair is eligible exactly when both unmodified greedy generations
are format-valid and exact:

\[
E_{m,s,r,d}=
\mathbf 1\{\hat y_{m,s,r}=r\}\,
\mathbf 1\{\hat y_{m,s,d}=d\}.
\]

Receiver and donor use the same seed, so they share the nested V4.4 passage
family. The registered count gaps remain (k=|d-r|\in\{1,3,5\}), with both
increase and decrease directions. The implementation rejects an intervention
call if either clean baseline is ineligible; filtering only at analysis time is
not accepted for new rows.

### Existing support and predeclared shortage

The target is at least five distinct eligible seed clusters for every
model-by-(k)-by-direction group. Correctness is symmetric in direction, so the
increase/decrease counts match within each model and (k).

| Model | (k) | Existing eligible seed clusters | Target | Initial shortage |
|---|---:|---:|---:|---:|
| Qwen3-8B | 1 | 5 | 5 | 0 |
| Qwen3-8B | 3 | 4 | 5 | 1 |
| Qwen3-8B | 5 | 2 | 5 | 3 |
| Gemma4-E4B | 1 | 5 | 5 | 0 |
| Gemma4-E4B | 3 | 5 | 5 | 0 |
| Gemma4-E4B | 5 | 1 | 5 | 4 |

Fresh reserve seeds 1274--1473 are inspected in ascending order. The stopping
rule may read only unmodified greedy baseline correctness. It may not read any
patching or ablation outcome. For each shortage, the runner chooses the first
minimum set of distinct eligible seeds and one registered anchor pair per seed.
Thus extra baseline-correct pairs found during the scan do not inflate the
number of interventions.

The machine-readable `supplement_selection.json` records:

- the initial shortage for every (k) and direction;
- every scanned seed, including seeds that did not contribute an eligible row;
- the clean-correct counts found for each scanned seed;
- the exact added directed pairs and their contributing seeds;
- the unused reserve suffix; and
- final support after supplementation.

### Average patching accuracy

For one exact group

\[
g=(m,\text{family},\text{site},\text{protocol},\ell,k,\text{direction}),
\]

the primary average patching accuracy is donor-target accuracy:

\[
\operatorname{PatchAcc}(g)
=\frac{1}{|I_g|}\sum_{i\in I_g}
\mathbf 1\{\hat y_i^{\mathrm{patch}}=d_i\}.
\]

Here (I_g) contains eligible donor-transport pair instances, (d_i) is the
donor gold count, and an invalid patched generation contributes zero. This is
stored as `average_patching_acc`; its numerator and denominator are stored as
`patching_acc_successes` and `patching_acc_denominator`. A seed-cluster
bootstrap gives the 95% interval: seeds are sampled with replacement, every
pair instance belonging to a sampled seed is retained, and the instance-level
ratio is recomputed. Thus the point estimate is pair-instance weighted while
the uncertainty calculation respects within-seed dependence.

Because "patching accuracy" can otherwise be ambiguous, every group also
reports

\[
\operatorname{ReceiverAcc}(g)
=\frac{1}{|I_g|}\sum_{i\in I_g}
\mathbf 1\{\hat y_i^{\mathrm{patch}}=r_i\},
\]

which measures retention of the receiver's original gold count. Successful
transport generally increases donor-target accuracy and can decrease receiver
accuracy; these quantities must not be interchanged.

Exact-group and model/family/(k)/direction aggregate tables are both written.
Every reported number is recomputed by the audit from the combined raw detail.

## Two ablation populations

Both analyses use answer-query-only ablation and the same frozen bank/top-(n)
per model. The three layer-matched random controls are retained. Ranked/random
head overlap remains allowed under the original registered control definition
and is not changed by this extension.

### Population A: all examples, signed effect

This population reuses the original independent confirmation without filtering
on baseline correctness. For example (i), the signed shift is

\[
\Delta_i^{\mathrm{signed}}
=\hat y_i^{\mathrm{ablate}}-\hat y_i^{\mathrm{clean}}.
\]

Negative values mean the generated count moved downward; positive values mean
it moved upward. The table also reports

- mean absolute count shift, (\mathbb E|\Delta_i^{\mathrm{signed}}|
  \), for magnitude without direction;
- accuracy change;
- absolute-error change relative to the gold count;
- prediction-change rate; and
- ranked-minus-random versions of these endpoints.

Signed count shift is undefined for invalid numeric generations. Such rows
remain failures for strict accuracy, while signed-shift means use the valid
numeric denominator and report patched valid rate.

### Population B: clean-correct baselines only

This population includes an example only if its unmodified greedy output is
format-valid and exactly correct before ablation. Counts remain 7, 8, 9, and 10,
matching the original frozen ablation confirmation domain. The primary
descriptive endpoint is failure induction:

\[
\operatorname{CorrectToWrong}
=\frac{1}{n}\sum_i
\mathbf 1\{\hat y_i^{\mathrm{clean}}=y_i,
             \hat y_i^{\mathrm{ablate}}\ne y_i\}.
\]

The existing independent confirmation contains 3 eligible Qwen seed clusters
and 0 eligible Gemma seed clusters. The target is 10 independent eligible seed
clusters per model, so the initial shortages are 7 and 10 respectively. The
same ascending reserve scan supplements only these shortages. All correct
count-7--10 examples in the selected baseline prefix are retained; heads and
top-(n) are never reselected.

The signed shift, absolute shift, error change, prediction-change rate, strict
accuracy, and ranked-minus-random comparisons are also reported for this
population. The all-example and clean-correct populations are stored as
separate rows; their denominators must never be pooled.

## Reproducible execution

The independent run is launched with
`scripts/launch_realistic_niah_v4_4_correct_interventions.sh`. It writes to a
new run root, checkpoints each baseline seed and intervention shard, and refuses
to start from a dirty repository. Its design hash includes the definition,
source selections, source confirmation details, rankings, frozen stimuli, and
implementation fingerprint.

The strict audit is
`scripts/audit_realistic_niah_v4_4_correct_interventions.py`. It verifies the
ordered seed prefix, initial/final quotas, clean-correct eligibility, unchanged
model-specific top-(n), complete ranked/random controls, explicit patching
accuracy denominators, and numerical reproduction of both patching and
ablation summaries from raw detail.

Until `correct_interventions.complete` exists and the audit reports `PASS`, the
extension is an implemented design rather than a completed empirical result.
