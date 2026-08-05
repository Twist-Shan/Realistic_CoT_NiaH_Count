# V4.4.4 Gemma independently selected natural-OV candidate

Candidate: Gemma4-E4B L35 H2.
Selection source: correct-only broad-retrieval ablation frozen before natural-OV outcomes.
The candidate identity was fixed before any outcome in this campaign.

Global intersection-union p: 1.
Full four-family support: False.

| family | IUT p | passes alpha=0.025 |
|---|---:|---|
| natural_signal | 0.95284939 | False |
| pre_o_injection | 1 | False |
| centered_removal | 0.99453068 | False |
| path_mediation | 0.99513817 | False |

The candidate is accepted only if natural carrier, true pre-O 
sufficiency, centered z-space necessity, and donor-z mediation all 
pass for both the candidate and candidate-minus-four-matched-controls.