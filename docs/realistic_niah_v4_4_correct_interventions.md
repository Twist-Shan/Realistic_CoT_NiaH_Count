# V4.4 correct-pair patching and dual-population ablation

## Purpose and relationship to V4.4

This is an additive confirmation extension. It does not overwrite, relabel, or
reselect any result in the audited V4.4 causal-v2 runs. Prompt/answer patching
keeps the condition set selected by the original five-seed overall screen.
Head ablation does **not** freeze an intervention size in this extension.
Instead, broad-aggregation top-(n), (n=1,\ldots,32), is run once on a shared
fresh-seed count-1--5 population. The all-example analysis retains every row;
the clean-correct analysis is the exact baseline-correct subset of those same
rows. The resulting tables are discovery diagnostics for human review, not
confirmed choices.

The extension answers two narrower questions:

1. When both clean examples are answered correctly, does residual patching move
   the receiver to the donor's gold count?
2. How do broad-aggregation head-ablation effects vary with top-(n), separately
   (a) over all registered examples and (b) conditional on a clean-correct
   baseline?

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

Both analyses use answer-query-only ablation and the broad-aggregation bank.
Every top-(n) from 1 through 32 is retained. The three layer-matched random
controls are retained. Ranked/random head overlap remains allowed under the
original registered control definition and is not changed by this extension.
No model- or population-specific (n) is frozen at this stage.

### Shared count and seed population

Both ablation analyses use counts 1, 2, 3, 4, and 5. Count 0 is excluded
because it is a qualitatively special no-target case; counts 6--10 are excluded
from this extension because the available Gemma run yielded no clean-correct
examples in the previous hard-count 7--10 domain. This scope makes the two
population estimates directly comparable, but it also limits the resulting
head-ablation claim to counts 1--5. Nothing in this extension establishes
generalization to counts 6--10.

Fresh seeds are inspected in ascending order. The registered discovery prefix
ends at the seed that supplies the tenth seed cluster with at least one exact,
format-valid count-1--5 baseline. If patching requires scanning farther, those
later seeds do not enter ablation discovery. Every count-1--5 stimulus in the
registered prefix is ablated once at every dose and control. Population B is
then obtained only by filtering Population A; interventions are not rerun.

### Population A: all examples, signed effect

This population contains every count-1--5 example in the registered fresh-seed
prefix without filtering on baseline correctness. The original hard-count
1--32 sweep is retained only as legacy context and is not pooled with or used
to select a dose for this extension. For example (i), the signed shift is

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
format-valid and exactly correct before ablation. It is the exact row subset of
Population A, so count domain, fresh-seed pool, intervention outputs, ranked
bank, and random controls are identical. The primary descriptive endpoint is
failure induction:

\[
\operatorname{CorrectToWrong}
=\frac{1}{n}\sum_i
\mathbf 1\{\hat y_i^{\mathrm{clean}}=y_i,
             \hat y_i^{\mathrm{ablate}}\ne y_i\}.
\]

Previous fixed-(n) confirmation rows cannot support an unbiased new 1--32
comparison because they used a different count domain and were not evaluated
at every (n). They are kept only as legacy reference. The target is therefore
10 **fresh** independent eligible seed clusters per model. All clean-correct
count-1--5 examples in the registered shared prefix are retained. If patching
support requires scanning farther into the reserve, later correct seeds are
recorded but excluded from top-(n) discovery, preventing the dose sweep from
expanding post hoc.

The signed shift, absolute shift, error change, prediction-change rate, strict
accuracy, and ranked-minus-random comparisons are also reported for this
population. The all-example and clean-correct populations are stored as
separate rows; their denominators must never be pooled.

## Top-n discovery diagnostics and deferred selection

The two populations use different primary discovery endpoints.

For the all-example population, direction alone is not a criterion for a
"better" ablation dose. Its primary diagnostic is the extra magnitude of
change produced by ranked heads relative to layer-matched random heads:

\[
D_n^{\mathrm{all}}
=\mathbb E\lvert\Delta_{i,n}^{\mathrm{ranked}}\rvert
-\mathbb E\lvert\Delta_{i,n}^{\mathrm{random}}\rvert.
\]

Larger positive values mean that ranked-head ablation changes generated counts
more than the random control. The corresponding signed ranked-minus-random
shift is reported separately to show whether the movement is predominantly up
or down.

For the clean-correct population, the primary diagnostic is the extra failure
induction relative to random:

\[
D_n^{\mathrm{correct}}
=\Pr(\mathrm{correct}\rightarrow\mathrm{wrong}\mid\mathrm{ranked},n)
-\Pr(\mathrm{correct}\rightarrow\mathrm{wrong}\mid\mathrm{random},n).
\]

Both diagnostics use a seed-cluster bootstrap. The output gives all 32 doses,
95% intervals, examples, seed clusters, valid rates, random overlap, secondary
endpoints, and a descriptive within-model rank. That rank is not an automatic
selection. After reviewing effect size, uncertainty, stability, and dose
parsimony, a model- and population-specific (n) may be frozen. Confirmation
must then use an untouched suffix of fresh seeds; discovery seeds may not be
reused as confirmatory evidence.

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
1--32 candidate set, explicitly unfrozen selection status, the shared count-1--5
seed grid, exact subset identity of the correct-only rows, complete ranked/random
controls, explicit patching accuracy denominators, and numerical reproduction
of both patching and ablation summaries from raw detail.

Until `correct_interventions.complete` exists and the audit reports `PASS`, the
extension is an implemented design rather than a completed empirical result.
