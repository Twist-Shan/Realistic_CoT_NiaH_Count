# V4.4 layerwise subspace removal, transport, and map stability

This directory contains the compact, report-facing outputs for the layerwise
extension of Section 5.4. Large packed hidden states, raw GPU rows, and the
76,000-row map bootstrap table remain in the run archive and are not committed.

## Frozen design

- Discovery seeds: 1234–1253.
- Confirmation seeds: 1254–1263.
- Prompt-removal counts: 2–10.
- Transport donor/receiver pairs: 1→2, 2→1, 5→6, and 6→5.
- Rank 3 is primary; map ranks 1 and 2 are sensitivity analyses.
- Layer and boundary landmarks, norm tolerance, multiplicity families, and the
  map-to-causal stability rule are frozen in
  `configs/realistic_niah_v4_4_layerwise_subspace.json`.

Prompt removal reports candidate absolute-error damage minus the realized-
norm-matched orthogonal-control damage. Transport reports aligned 1× minus
orthogonal target donor fraction and aligned 2× minus aligned 1×. Inference is
at the confirmation-seed level; registered layer or boundary families use
Holm correction.

## Linear map and gauge

For adjacent layers, centered rank-3 coordinates are fit with ridge regression,

`Z_(l+1) ≈ Z_l A_l`,

and `A_l = R_l S_l` is the right polar decomposition. Raw PCA-coordinate
rotation angles are gauge dependent. Bootstrap bases are therefore
Procrustes-aligned within a boundary. Cross-boundary continuity uses the
gauge-invariant ambient-space operator

`T_l = U_l A_l U_(l+1)^T`,

reported through consecutive-operator cosine and relative Frobenius drift.

The causal transport basis `B_l` is related but not identical: it is obtained
by directly ridge-regressing source ambient centroids onto target rank-3
coordinates and taking a QR basis of the regression weights. The registered
map-to-causal test asks whether held-out causal transport defined by `B_l` is
stronger where the discovery-only coordinate map `A_l` is locally stable. It
does not claim to intervene directly on the polar rotation matrix.

## Confirmatory results

All eight raw and analysis audits pass. Rank-3 prompt removal does not show a
control-adjusted counting-specific effect at any registered layer in either
model after Holm correction (0/92 model-by-population-by-endpoint layer tests
significant). The corresponding depth slopes are also non-significant. Thus,
removing the prompt count subspace at a single layer is not sufficient evidence
for a layer-localized necessary channel under this intervention and control.

Answer-query transport shows the opposite pattern. The aligned-dose-1 minus
orthogonal target-donor-fraction contrast rises strongly with depth: the
seed-level slope is 0.961 per unit normalized depth for Qwen3-8B and 1.201 for
Gemma4-E4B (both exact sign-flip `p=0.001953`). Ten of 11 Qwen boundaries and
all 14 Gemma boundaries survive their registered Holm families. Gemma's
earliest `L0→L1` effect is negative (-0.452), whereas the largest later effects
are 0.942 for Qwen (`L28→L29`) and 0.976 for Gemma (`L36→L37`). Dose 2 adds a
positive depth-dependent increment in both models.

The frozen stable-minus-unstable primary contrast is positive for both models:
0.532 for Qwen (95% bootstrap CI [0.521, 0.545]) and 0.661 for Gemma
([0.567, 0.783]); both have exact seed sign-flip `p=0.001953` and Holm-adjusted
`p=0.011719`. This supports the registered claim that locally stable
discovery-only maps identify boundaries with stronger held-out causal
transport. It does not isolate map stability from late-layer maturation,
because stability status is strongly aligned with depth.

The realized-norm audit has maximum control/dose-1 ratio error 0.0179 and
maximum absolute dose-2/dose-1 ratio error 0.0450, within the frozen 2.5%
relative tolerance. The maximum deviation from the continuous-space planned
norm is 0.0395 and is reported diagnostically; BF16 causal matching is defined
on realized paired doses. Full code history, model revisions, raw hashes, and
the preserved failed partial row are recorded in `run_provenance.json`.

## Report-facing files

- `layer_maps/`: adjacent-layer map summaries and audit.
- `prompt_removal/`: paired layerwise removal statistics, trends, and audit.
- `transport/`: layerwise transport contrasts, trends, and audit.
- `map_causal_link/`: frozen stability labels and stable-minus-unstable tests.
- `run_provenance.json`: runtime, model revisions, code history, raw hashes,
  and formal audit status.

All four component audits must be `PASS` before the integrated report builder
will render Section 5.4C–E.
