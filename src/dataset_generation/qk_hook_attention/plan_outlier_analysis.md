# Codex task plan: outlier-token analysis with Qwen3 Q/K caches

## Goal

Continue coding in the existing repo to expand `analysis_hidden_states_v4.ipynb` (and slightly modify existing python scripts if necessary) for **uncontrolled prompts only**: needles are inserted at every requested position. Study **outlier tokens** at selected layers of `Qwen/Qwen3-8B`. Here, outlier tokens include:

1. **attention sink tokens**: token positions that receive unusually large attention probability from later query tokens;
2. **massively activating tokens**: token positions whose hidden-state norms or scalar hidden-state activations are unusually large.

Use the existing Q/K cache pipeline:

- `capture_qk_qwen3.py`
- `analyze_qk_qwen3.py`

Do **not** use `output_attentions=True` for long contexts. Reusable logic should go into Python utility functions or scripts; notebook cells should mostly call these utilities and display results.

## Constraints and variables

Mainly work in `analysis_hidden_states_v4.ipynb`, revising scripts only when needed. Respect existing variables and folder conventions, especially `RUN_DIR`, `LAYERS`, example IDs, hidden-state tensor paths, and existing `figures`, `tables`, and `tensors` folders.

Add:

```python
HEADS = None  # None means all heads; otherwise a list such as [0, 1, 7, 15]
MASSIVE_NORM_RATIO_THRESHOLD = 10.0
MASSIVE_TOP_K_PER_LAYER = 50
N_CRITICAL_EDGE_TOKENS = 10
N_AFTER_NEEDLE = 10
```

Save tensors under `RUN_DIR/tensors/`, tables under `RUN_DIR/tables/`, and figures under `RUN_DIR/figures/`.

## 1. Generate or reuse Q/K caches

Add a notebook block that checks whether Q/K cache exists for each uncontrolled example and each layer in `LAYERS`. If missing, call `capture_qk_qwen3.py` to generate the cache. Save files under a structure like:

```text
RUN_DIR/tensors/qk_cache/input_{example_idx}/
  metadata.json
  input_ids.pt
  attention_mask.pt
  position_ids.pt
  tokens.json
  layer_00_q_raw.pt
  layer_00_k_raw.pt
```

Print shape checks after capture. For Qwen3-8B, expect:

```text
q_raw: [1, T, 4096]
k_raw: [1, T, 1024]
```

Downstream analysis should support all heads by default and respect `HEADS` when specified.

## 2. Attention-sink and attention-statistics analysis

Extend offline analysis so the notebook can run across:

```text
examples × LAYERS × HEADS
```

without rerunning model inference.

Be precise about attention direction:

- **emitted attention** from query token `q`: `A[q, :]`, where token `q` attends;
- **received attention** to key token `k`: `A[:, k]`, how much later tokens attend to token `k`.

Attention sink analysis is mainly **received-attention** analysis.

For each example, layer, and head, compute:

```text
received_mean[t] = mean_{q > t} A[q, t]
received_sum[t] = sum_{q > t} A[q, t]
received_count[t] = number of valid later queries
```

Also compute a position-normalized sink ratio:

```text
uniform_baseline[t] = mean_{q > t} 1 / (q + 1)
received_uniform_ratio[t] = received_mean[t] / uniform_baseline[t]
```

Use `NaN` for positions with no later queries. Save these tensors under:

```text
RUN_DIR/tensors/attention_stats/input_{i}/
```

For **critical tokens**, include BOS/special tokens, Qwen chat or thinking markers if present, first/last `N_CRITICAL_EDGE_TOKENS`, tokens after each needle span, and tokens later identified as massive activations. For each critical position `c`, save both:

```text
A[c, :]  # emitted row
A[:, c]  # received column, invalid q <= c as NaN
```

Save a compact dictionary containing positions, labels, rows, and columns.

For each needle span, compute the average attention mass the whole needle receives from later tokens. Save a per-head JSON and a combined CSV, e.g. `RUN_DIR/tables/needle_attention_mass.csv`.

Also expose existing statistics from `analyze_qk_qwen3.py`: top-k attended positions, entropy for selected query rows, entropy by position when feasible, local-window mass, and span mass for named spans. Save compact JSON/CSV summaries.

## 3. Massively activating token analysis

For each example and layer in `LAYERS`, load existing hidden states and compute:

```text
hidden_norm[t] = ||h_t||_2
norm_ratio_to_median[t] = hidden_norm[t] / median(hidden_norm)
robust_z[t] = (hidden_norm[t] - median(hidden_norm)) / MAD(hidden_norm)
```

Select massive tokens if they pass `MASSIVE_NORM_RATIO_THRESHOLD` or appear in the top `MASSIVE_TOP_K_PER_LAYER` fallback. Save per-layer tensors and combined event files:

```text
RUN_DIR/tensors/massive_activations/input_{i}/hidden_norms_layer_{l:02d}.pt
RUN_DIR/tables/massive_tokens_all.csv
RUN_DIR/tables/massive_tokens_all.jsonl
RUN_DIR/tables/massive_tokens.txt
```

Each event row should include: example index, layer, position, token ID/string, hidden norm, median norm, norm ratio, robust z, whether the token is special/early/late/needle/after-needle, left/right context, and trigger rule.

Also add scalar massive-activation analysis. For each token/layer compute:

```text
max_abs_activation[t] = max_d |h_t[d]|
argmax_abs_dim[t] = argmax_d |h_t[d]|
```

Save `massive_scalar_activations_all.csv` and `massive_activation_dim_counts.csv`. Summarize whether the same hidden dimensions repeatedly produce large activations across examples or layers.

## 4. Relate attention sinks to massive activations

Join attention and activation statistics by example/layer/head/position. Compare norm ratio, scalar max-activation ratio, received attention, received uniform ratio, entropy if available, and whether the position is a top-k sink or massive token.

Save:

```text
RUN_DIR/tables/outlier_attention_join.csv
RUN_DIR/tables/outlier_overlap_summary.csv
```

Include overlap fractions and correlations between `norm_ratio_to_median` and `received_uniform_ratio`.

## 5. Figures and notebook summaries

Enhance or recreate `inputs_{i}.png`. Add two subplots above the existing plots:

1. token position vs. received attention, preferably `received_uniform_ratio`, one line per layer and averaged over `HEADS`;
2. token position vs. attention entropy, again layer-wise and head-averaged.

Overlay vertical dashed lines at massive-token positions, using the same color as the corresponding layer. If modifying the original figure code is fragile, create new figures such as:

```text
RUN_DIR/figures/outlier_attention_input_{i}.png
RUN_DIR/figures/norm_vs_attention_sink_input_{i}.png
```

Add concise notebook summaries and save them to:

```text
RUN_DIR/tables/attention_statistics_summary.txt
RUN_DIR/tables/massive_activation_summary.txt
```

The task is complete when the notebook runs on uncontrolled prompts, reuses or generates Q/K caches, computes attention-sink and massive-activation statistics offline, saves tensors/tables/figures, and never uses `output_attentions=True` for long contexts.
