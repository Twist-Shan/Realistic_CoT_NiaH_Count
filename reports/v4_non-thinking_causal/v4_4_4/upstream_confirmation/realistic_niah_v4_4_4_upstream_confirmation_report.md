# V4.4.4 independent upstream-path confirmation

**Primary result: CONFIRMED.**

## Frozen hypothesis and design

The frozen early broad-retrieval top-4 is patched only at registered slot-query positions. The primary L28 mediator is H16--H19. Exact restoration of the induced pre-O Z change is compared with an equal-post-O-norm, same-W_O-span orthogonal control. Seeds 1294--1313 were not used for head selection or the exploratory upstream-path analysis.

Primary endpoint: donor-vs-receiver candidate-sequence log-odds gain, averaged over six directed donor pairs within each seed. The serial-chain claim uses an intersection-union test: both the early intervention and L28 mediation specificity must be positive with two-sided exact sign-flip p < 0.05.

## Primary result

- Early slot-state donor log-odds gain: 0.1057 (95% bootstrap CI [0.0412, 0.1683], exact p=0.005884).
- L28 H16--H19 mediation specificity (orthogonal control minus exact natural block): 0.1709 (95% CI [0.1127, 0.2353], exact p=0.000025).
- Intersection-union p: 0.005884.

Conclusion for this section: the previously exploratory early top-4 slot-state → L28 H16--H19 → answer chain replicated on independent seeds.

## Leave-one-out member analysis

`full − leave-one-out` is the paired loss of mediation specificity after removing one L28 head. Positive values indicate an incremental contribution by that head. Holm correction is across the four heads.

| Removed head | LOO mediation | Full−LOO decrement | 95% CI | exact p | Holm p | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| H16 | 0.1014 | 0.0696 | [0.0120, 0.1336] | 0.039143 | 0.117428 | no unique decrement resolved |
| H17 | 0.1895 | -0.0186 | [-0.0884, 0.0467] | 0.609287 | 1.000000 | no unique decrement resolved |
| H18 | 0.1813 | -0.0104 | [-0.0687, 0.0495] | 0.738483 | 1.000000 | no unique decrement resolved |
| H19 | 0.0171 | 0.1538 | [0.0783, 0.2291] | 0.001011 | 0.004044 | necessary within the tested set |

Conclusion for this section: leave-one-out identifies incremental membership within the tested H16--H19 set; it does not claim that any single head implements counting alone.

## Audit

All audit checks passed: **True**.

- primary_row_count: True — `{'observed': 120, 'expected': 120}`
- leave_one_out_row_count: True — `{'observed': 480, 'expected': 480}`
- independent_seed_registry: True — `[1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313]`
- frozen_route_and_early_set: True — `{'routes': ['slot_state'], 'early_sets': ['top4']}`
- frozen_late_sets: True — `['full_h16_h19', 'minus_h16', 'minus_h17', 'minus_h18', 'minus_h19']`
- exact_l28_block_closure: True — `0.0`
- same_span_orthogonal_control: True — `3.281607874328074e-08`
- deterministic_prefill_reproducibility: True — `0.0`
- early_intervention_identical_across_late_comparisons: True — `0.0`
- no_persisted_raw_states_or_attention: True — `[]`
