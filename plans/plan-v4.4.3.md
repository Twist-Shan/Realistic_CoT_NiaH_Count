# V4.4.3 implementation checklist

1. Keep V4.4 input read only and create a unique V4.4.3 filestream namespace.
2. Reuse frozen prompt running-index and answer-query states for direction fit.
3. Fit LayerNorm-correct, GQA-correct OV mapping scores on discovery only.
4. Freeze candidate and same-layer/norm-matched control heads.
5. Run actual-model smoke for alpha/Z/O patches and directed interventions.
6. Run staged-patch screen on seeds 1254--1258.
7. Run removal/injection confirmation on seeds 1259--1263.
8. Aggregate by seed, report exact sign-flip tests and seed bootstrap intervals.
9. Audit dynamic shard counts and write the final mechanism report.

No raw attention maps or full hidden-state dumps are part of V4.4.3.
