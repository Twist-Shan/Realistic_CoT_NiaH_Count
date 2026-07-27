# Counting-mechanism empirical-law search plan (frozen before fitting)

Date: 2026-07-25

## Scope and invariants

- Source: the canonical 6,300-row `request_level_report.csv` in the
  `niah_eight_model_report_20260724_model_bias` report.
- Analyze `direct` (nonthinking), `enumeration`, and `native_thinking` (CoT)
  separately.
- Do not use model parameter count as a predictor. Each model may have its own
  intercept, length coefficient, needle coefficient, interaction, and
  query-order nuisance coefficient, but candidate functional families are
  shared within a prompt mode.
- Keep every request. Parse failures, format failures, and truncations remain
  failures for the primary exact-correct target. Numeric-error targets are
  explicitly conditional on a successfully parsed numeric answer; their
  denominator is always reported.
- No point deletion and no post-hoc candidate invention after seeing held-out
  scores. All candidates and convergence failures are retained.

## Coordinates and bounded candidate grid

Let `l = log(L / 5000)` and `n = log(N / 5)`. The finite grid is:

1. intercept only;
2. log density `n - l`;
3. log burden `n + l`;
4. separable power coordinates `l + n`;
5. power coordinates with `l*n`;
6. quadratic log surface `l + n + l^2 + n^2 + l*n`;
7. piecewise log surface with knots at `L=5000` and `N=8`;
8. separable raw/exponential coordinates `(L/5000-1) + (N/5-1)`;
9. raw coordinates with interaction;
10. separable square-root coordinates;
11. separable inverse coordinates;
12. hybrid `log L + raw N`;
13. hybrid `raw L + log N`.

Every design includes model-specific parameters and a model-specific
query-last nuisance term. Binary/binomial targets compare logistic and
survival-log-log links. Continuous targets compare the same coordinate grid
with ordinary least squares on a predeclared target transform.

## Outcomes

Primary outcome in every mode:

- exact correctness on all requests.

Shared mechanism outcomes:

- parse success;
- truncation;
- exact correctness conditional on successful numeric parsing;
- within-one count accuracy conditional on parsing;
- undercount and overcount probabilities conditional on parsing;
- signed relative error `(predicted-N)/N`;
- `asinh(signed error)`;
- `log1p(absolute error)` and `log1p(relative absolute error)`;
- signed log count ratio `log((predicted+0.5)/(N+0.5))` for nonnegative
  parsed predictions;
- output-token usage.

Enumeration-specific outcomes:

- per-needle retrieval success, fitted as a grouped binomial with
  successes `N-missing_pairs` and trials `N`;
- probability all gold pairs were retrieved;
- probability of zero hallucinated pairs;
- probability of zero duplicate listed pairs;
- probability the listed-record count equals `N`;
- missing-pair fraction, hallucinated-pair rate, duplicate rate, and
  listed-record ratio;
- the compounding diagnostic `q_hat^N`, where `q_hat` is held-out predicted
  per-needle retrieval probability, compared with held-out all-pairs-found
  and exact correctness.

## Validation and selection

- Five leave-one-seed-out folds, keeping same-seed stimuli together.
- Five blocked `(L,N)` cell folds, keeping every model, seed, and query order
  for a held-out cell together.
- Leave-one-needle-level-out and leave-one-length-level-out diagnostics for
  the final selected laws.
- Primary exact-law selection uses nested validation: outer blocked cell
  folds, inner leave-one-seed-out candidate selection.
- Candidate rankings use held-out log loss/Brier score for binary/binomial
  outcomes and held-out normalized RMSE/MAE/R2 for continuous outcomes.
  In-sample R2 is descriptive only.
- Prefer the simplest candidate within one standard error of the best
  validation score. A more complex surface is selected only when the held-out
  improvement exceeds that tolerance.
- Report request-level metrics and held-out cell-level observed-versus-
  predicted R2 separately; the latter must not be described as request-level
  accuracy.

## Mechanistic laws to test

1. **Failure-hazard law**

   `p_exact = exp[-A_m (L/5000)^r_m (N/5)^s_m]`

   or its logistic/Hill analogue. This tests multiplicative burden.

2. **Retrieval-compounding law for enumeration**

   `logit(q_m) = a_m + r_m log(L/5000) + s_m log(N/5)`

   and `P(all N retrieved) ≈ q_m^N`.

3. **Count-error law**

   `E[log(1+|error|)] = a_m + r_m log(L/5000) + s_m log(N/5)`

   with parallel signed-relative and log-ratio diagnostics.

4. **Mechanism decomposition**

   Exact failure is compared against parsing failure, truncation, retrieval
   failure, hallucination/duplication, and numeric counting error. This is a
   predictive decomposition, not a causal claim.

## Decision rule

A “counting mechanism law” is claimed only if a shared functional family has
stable signs/orders across multiple models, improves grouped held-out
prediction over intercept and density-only baselines, and survives blocked
cell plus leave-level diagnostics. If no such family exists for a target or
mode, the result is “no reliable unified law in the tested domain”; a high
training fit is not sufficient.
