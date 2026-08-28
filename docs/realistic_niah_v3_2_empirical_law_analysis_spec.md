# Realistic CoT NiaH V3.2 Empirical-Law Analysis Specification

Status: **frozen post-generation analysis amendment**  
Frozen on: **2026-08-26 (America/Chicago)**  
Inference-data commit: `939410edde9885fea5c791e90fdb632254bd327c`  
Machine-readable companion:
`configs/realistic_niah_v3_2_empirical_law_analysis.json`

Final candidate/estimand extensions:
`configs/realistic_niah_v3_2_inverse_n_candidate_extension.json` and the
audited raw-scale conditional-MAE analysis manifest. These extensions reuse
the same frozen requests, folds, gates, and LOMO procedure; no selected law
changed merely because `1/N` was added.

## 1. Scope and timing

V3.2 changes only the empirical-law analysis of the completed V3.1 inference
run. It does not generate new model responses and does not change any stimulus,
prompt, parser, model revision, request ID, or output.

The immutable input contains 161,280 unique requests from 14 fixed physical
model revisions, represented as 48 model-mode slots and 12 comparison slots.
The registered grids remain:

- `N = {1,2,3,4,5,6,7,8,9,10,12,15,18,20}`;
- `L = {1000,2000,3000,5000,8000,10000,15000,20000}`;
- 30 complete seed IDs per `model x prompt mode x N x L` cell;
- prompt modes `direct`, `enumeration_index`, `enumeration_bullet`, and
  `native_thinking`.

This amendment was written after V3.1 response generation and after an earlier
ordinary-mean exploratory analysis, but before running the V3.2 accuracy and
trimmed-bias candidate fits. Therefore the V3.2 report must call this a frozen
post-generation analysis specification, not an outcome-blind preregistration.

## 2. Formal estimands

### 2.1 Parsed exact accuracy

For request `i`, define

```text
Y_i = 1  if the response parses to an integer equal to the true N,
      0  otherwise.
```

Parse failures, truncations, and parseable wrong counts therefore contribute
zero to the primary accuracy estimand. Parse rate, conditional numerical
accuracy, format-compliance rate, truncation rate, and strict accuracy remain
descriptive decomposition outcomes; they are not additional V3.2 empirical-law
selection targets.

### 2.2 Ten-percent trimmed signed bias

For a cell with `m` parseable predicted counts, let

```text
e_i = predicted_count_i - N,
k = floor(0.10 * m),
trimmed_signed_bias_10 = mean(e_(k+1), ..., e_(m-k)),
```

where `e_(i)` denotes the ordered signed errors. A cell is eligible for the
formal bias law only when `m >= 20`. Eligible cells enter with equal weight.
The identity-link model is fit to this cell-level outcome.

The ordinary mean signed deviation is not a V3.2 law outcome and must not be
used for model selection, coefficient interpretation, or headline figures.
Raw extreme errors may still be retained in audit or tail-risk diagnostics.

### 2.3 Conditional mean absolute error

For the same eligible parseable responses, define

```text
conditional_mae = mean(abs(predicted_count_i - N)),  m >= 20.
```

Fit this outcome directly in count units by identity-link OLS. Do not apply a
`log1p` transform and do not clip negative fitted values. Because observed MAE
is nonnegative, any negative fitted value is reported as a diagnostic that the
identity model is a local descriptive approximation rather than a generative
distribution.

## 3. Predictor definitions and bounded structural registry

Use

```text
L_k  = L / 1000,
logN = ln(N),
logL = ln(L_k).
```

The frozen base registry contains the following 13 structures, in this
tie-breaking order:

1. intercept only;
2. `N`;
3. `L_k`;
4. `logN`;
5. `logL`;
6. `N + L_k`;
7. `logN + logL`;
8. `N + logL`;
9. `logN + L_k`;
10. `N + L_k + N:L_k`;
11. `logN + logL + logN:logL`;
12. `N + logL + N:logL`;
13. `logN + L_k + logN:L_k`.

The final V3.2 specification additionally audits five `1/N` structures, in
this order: `1/N`; `1/N + L_k`; `1/N + logL`; `1/N + L_k + (1/N):L_k`; and
`1/N + logL + (1/N):logL`. The complete final registry therefore has 18
structures. Every interaction retains both parent terms. The audit selected
no inverse-count structure and changed none of the base selections.

Every interaction includes both parent main effects. Density terms,
count-count interactions, length-length interactions, higher-order
polynomials, splines, and post-hoc formulas are excluded from V3.2 selection.

The same candidate structure is selected across the 12 comparison slots
within a prompt mode, while coefficients are fit separately for each
comparison slot. Structures are selected separately by prompt mode and formal
estimand.

## 4. Accuracy-law families

### 4.1 Headline family: request-level Bernoulli-logit

For comparison slot `m`, prompt mode `p`, and candidate features `x_j`, fit

```text
logit(P(Y_i = 1 | m,p,N,L))
    = alpha_(m,p) + sum_j beta_(m,p,j) x_j(N,L).
```

The Bernoulli-logit result determines the headline V3.2 accuracy structure.

### 4.2 Link-function robustness families

Repeat the complete final-registry search using request-level Bernoulli GLMs
with:

- probit link;
- complementary-log-log (`cloglog`) link.

These families test whether the selected `N/L` structure depends on the
sigmoid/link shape. They must be reported beside the logit result, but cannot
replace the logit headline merely because they produce a more favorable
substantive conclusion.

### 4.3 Overdispersion robustness family

After the Bernoulli-logit structure has been selected, refit that fixed
structure to cell success counts using a Beta-Binomial-logit likelihood with a
separate dispersion parameter for each comparison slot and prompt mode. This
check reports overdispersion and predictive calibration; it does not rerun
structure selection.

A separate Binomial-logit search is omitted because, with identical cell
covariates and denominators carried correctly, it duplicates the aggregated
likelihood of the request-level Bernoulli-logit model. It may be emitted as a
numerical parity audit, not as an additional scientific family.

## 5. Model selection: focused empirical-law procedure

V3.2 uses the earlier focused empirical-law procedure rather than the V3.1
nested/bootstrap design.

### 5.1 Fixed held-condition folds

For the 14 ordered `N` levels and eight ordered `L` levels, assign each complete
condition to

```text
fold = (index(N) + index(L)) mod 5.
```

All 30 seeds for a held `N x L` condition remain together. There is no nested
held-seed, leave-one-N, or leave-one-L candidate reselection.

### 5.2 Per-slot fits and cross-slot summaries

Fit every candidate separately for every comparison slot and prompt mode.
Summarize candidate performance across comparison slots using the median and
25th percentile (`Q25`).

For accuracy families:

- primary loss: held-condition request-level log loss;
- primary higher-is-better score:
  `CV_D2 = 1 - candidate_log_loss / intercept_log_loss`;
- secondary metrics: Brier score, calibration intercept/slope, and accuracy.

For trimmed signed bias:

- primary loss: held-condition MAE;
- primary higher-is-better score: held-condition `R^2`;
- secondary metric: RMSE.

Conditional MAE uses the same held-condition MAE loss, held-condition `R^2`,
and RMSE on the untransformed count-error scale.

### 5.3 Practical and special-term gates

The focused-method thresholds are retained:

- every non-intercept term has median absolute standardized effect at least
  `0.10` across comparison slots;
- an interaction term has BH-adjusted `q < 0.05` in at least 50% of comparison
  slots;
- the interaction improves the primary CV score over its matched additive
  parent by a median of at least `0.02`;
- the one-sided paired Wilcoxon test of slot-level CV-score gains has
  BH-adjusted `q <= 0.05` within the prompt mode.

The intercept-only candidate is always eligible. If no non-intercept candidate
passes the practical gate and improves on it, report `no reliable shared law`.

### 5.4 Near-best and complexity rule

Among eligible candidates:

1. retain structures whose median primary CV score is within `0.02` of the
   best median;
2. retain structures whose Q25 primary CV score is within `0.05` of the best
   Q25 among the remaining candidates;
3. prefer fewer predictors;
4. then prefer lower median primary loss;
5. then prefer higher median and Q25 primary CV scores;
6. break any remaining tie by the registry order in Section 3.

### 5.5 Leave-one-model-out structure stability

For the headline Bernoulli-logit accuracy law, trimmed-bias law, and final
raw-scale conditional-MAE law, repeat
the cross-slot structure selection after omitting each comparison slot. Report
the fraction recovering the full-data structure and the held-slot predictive
score. LOMO is a model-set sensitivity analysis, not an uncertainty interval
for an unseen model.

Probit, cloglog, and Beta-Binomial robustness fits do not trigger their own
LOMO loops.

## 6. Uncertainty and multiplicity

No bootstrap is run in V3.2.

- OLS and Bernoulli GLM coefficients use HC3 sandwich covariance within each
  comparison-slot fit.
- Non-intercept coefficient p-values are Benjamini-Hochberg adjusted across
  comparison slots within each `outcome family x prompt mode x candidate x
  term` family.
- Cross-slot special-term CV gains use a one-sided paired Wilcoxon test and BH
  correction across the registered interaction structures within a prompt
  mode.
- Beta-Binomial robustness coefficients use Hessian/Wald intervals and are
  explicitly labelled model-based rather than HC3.
- LOMO recovery frequency is reported as structural sensitivity, not as a
  confidence interval.

There are zero interaction, validation, coefficient, or reselection bootstrap
replicates.

## 7. Required reporting

The V3.2 report must include:

1. the immutable request and revision audit;
2. descriptive parse, exact-accuracy, strict-accuracy, format, and truncation
   decomposition;
3. the Bernoulli-logit selected law for every prompt mode;
4. probit and cloglog selected structures and predictive metrics as link
   robustness checks;
5. the fixed-structure Beta-Binomial overdispersion check;
6. the eligible-cell 10% trimmed signed-bias law for every prompt mode;
7. every attempted candidate and all selection gates;
8. HC3/BH coefficient tables, interaction-gain tests, and primary-law LOMO;
9. clear separation of same-checkpoint and checkpoint-confounded comparisons;
10. an explicit statement that ordinary mean signed deviation was discarded
    as a law estimand.

Claims are limited to the registered `N/L` grid and frozen model set. V3.2
describes predictive regularities; it does not identify a neural mechanism or
establish extrapolation to new models or longer contexts.
