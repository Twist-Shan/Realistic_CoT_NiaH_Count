# V4.4.4 independent confirmation: early slot-state → L28 OV → answer

## Confirmatory claim

The exploratory campaign identified one supported serial route:

`early broad-retrieval top-4 slot-state patch → L28 H16--H19 → answer`.

This campaign freezes that route before seeing any new-seed model outcomes. It
does not retest unsupported stronger claims such as a unique counter head, a
single sufficient prompt endpoint, or identical QK and OV heads.

## Frozen design

- Model: Qwen3-8B, non-thinking, V4.4 prompts.
- Independent evaluation seeds: 1294--1313. These are disjoint from the
  exploratory upstream-path seeds 1284--1293 and all direction/center seeds.
- Counts: 1--10.
- Six directed donor pairs: 1↔6, 3↔8, and 5↔10.
- Early set: L27H18, L23H28, L23H29, L26H20, frozen from the cue-stable V4.4.2
  broad-retrieval ranking.
- Early intervention: donor pre-O Z at registered slot-token query positions;
  answer-query activations at the early layers are not directly patched.
- Primary mediator: L28 H16,H17,H18,H19.
- Exact mediator block: restore the induced L28 selected-set pre-O Z to its
  clean receiver value.
- Matched control: same selected-head `W_O` span, same post-O norm, orthogonal
  to the induced natural output.
- Primary endpoint: donor-vs-receiver candidate-sequence log-odds gain.
- Raw hidden states and raw attention are not persisted.

## Primary inference

Within each seed, all six directed pairs are averaged. Define:

- `G_early`: donor log-odds gain under the early slot-state patch.
- `G_block`: gain after the early patch plus exact L28 natural-channel block.
- `G_orth`: gain after the early patch plus the matched orthogonal control.
- `M = G_orth - G_block`: L28 mediation specificity.

The serial chain is independently confirmed only if:

1. mean `G_early > 0`;
2. mean `M > 0`;
3. both two-sided exact paired sign-flip p-values are below 0.05;
4. block closure, orthogonality, deterministic replay, row-count, and seed
   audits all pass.

This is an intersection-union decision with global `p = max(p_early, p_M)`.

## Secondary member analysis

Run four frozen leave-one-out mediator sets: remove H16, H17, H18, or H19.
For each head, compare the paired seed-level decrement

`D_h = M_full - M_minus_h`.

Positive `D_h` means that the head contributes incrementally within H16--H19.
Two-sided exact sign-flip p-values are Holm-corrected across the four heads.
This analysis distinguishes incremental, redundant, and set-necessary members;
it does not require any head to implement counting alone.
