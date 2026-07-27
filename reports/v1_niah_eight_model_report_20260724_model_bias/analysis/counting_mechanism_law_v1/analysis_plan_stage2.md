# Counting-mechanism stage-2 refinement plan

Frozen: 2026-07-25, after completing and retaining the full stage-1 grid.

## Why this stage was triggered

Stage 1 produced two prespecified diagnostic failures:

1. For enumeration, blocked-cell prediction from the independent-retrieval
   formula `q_hat^N` had cell R2 around 0.33 for the event that all gold pairs
   were retrieved, much lower than the direct held-out response surface.
   Therefore needle retrieval failures are not well described as independent
   and identically difficult.
2. A single signed-relative-bias surface had low held-out R2 because
   undercounts and overcounts are a mixture whose signs cancel. Conditional
   absolute-error and under/over probabilities were substantially more
   predictable.

These diagnostics are retained. Stage 2 does not replace them.

## Refinement A: correlated retrieval through an effective needle count

Keep the stage-1 per-needle retrieval model `q_m(L,N,o)`. Compare this finite
compounding grid:

1. independence: `P(all)=q^N`;
2. model-specific scale: `P(all)=q^(kappa_m N)`;
3. shared scale and order: `P(all)=q^(kappa N^tau)`;
4. model-specific scale and shared order:
   `P(all)=q^(kappa_m N^tau)`;
5. model-specific scale and order:
   `P(all)=q^(kappa_m N^tau_m)`;
6. model-specific scale/order plus query-order correction in the effective
   exponent.

The effective independent needle count is
`N_eff,m = kappa_m N^tau_m`. Values below N indicate positively correlated
retrieval success or heterogeneous common difficulty; they are an empirical
summary, not a proof of a particular internal architecture.

Validation:

- outer five blocked `(L,N)` cell folds;
- within each outer training set, cross-fitted leave-one-seed-out `q` values
  are used to estimate compounding parameters;
- per-candidate outer OOF log loss, Brier score, and cell R2 are retained;
- the simplest candidate within one standard error of the best is selected.

## Refinement B: hurdle law for count bias

For each prompt mode and model, decompose parsed numeric responses:

`E[(predicted-N)/N] =
 P(over) E[|error|/N | over] -
 P(under) E[|error|/N | under]`.

- Under/over probabilities use grouped logistic response surfaces.
- Conditional magnitudes use `log(1+|error|/N)` and the same frozen
  coordinate grid from stage 1.
- Magnitudes are retransformed with a training-only Duan smearing factor.
- Outer blocked-cell predictions are combined request by request, then
  evaluated against held-out cell mean signed-relative bias and held-out cell
  mean relative absolute error.
- Compare against the single-surface signed-relative-bias baseline and a
  model-only hurdle baseline.

Also retain separate conditional undercount- and overcount-magnitude candidate
tables, coefficients, and leave-seed/cell metrics.

## Refinement C: parse × conditional counting decomposition

For each mode:

`P(exact) = P(parse) × P(exact | parse)`.

Each component is fit only on the appropriate training denominator, then
predicted for an outer held-out `(L,N)` cell. This exact probability identity
is used as a hurdle prediction; the component models remain empirical.
Compare its held-out log loss, Brier score, and cell R2 with the direct exact
surface from stage 1.

## Decision rules

- Do not call a refinement successful solely from in-sample fit.
- Prefer a mechanism law only if blocked-cell OOF prediction improves and
  model parameter signs are stable enough to interpret.
- Retain all stage-2 candidates, including independence and model-only
  baselines.
- If the flexible effective-exponent candidate wins but its parameters vary
  wildly under leave-seed or leave-level checks, report that it is predictive
  but not a stable empirical law.
