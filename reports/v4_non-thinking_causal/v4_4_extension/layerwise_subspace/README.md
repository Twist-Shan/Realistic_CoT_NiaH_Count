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

## Report-facing files

- `layer_maps/`: adjacent-layer map summaries and audit.
- `prompt_removal/`: paired layerwise removal statistics, trends, and audit.
- `transport/`: layerwise transport contrasts, trends, and audit.
- `map_causal_link/`: frozen stability labels and stable-minus-unstable tests.

All four component audits must be `PASS` before the integrated report builder
will render Section 5.4C–E.
