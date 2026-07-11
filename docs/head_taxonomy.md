# Attention-head taxonomy

The implementation screens Qwen3 attention heads for three functional
families. Labels are non-exclusive because a head can satisfy more than one
behavioral criterion.

## Metrics

### Targeted retrieval

For specified answer/retrieval query positions, measure total attention mass
assigned to target or needle spans. The reported lift divides observed mass by
the causal-uniform expectation:

    number of eligible target keys / number of causal keys in that row

A useful NIAH control is to compare the correct needle with equally sized
non-target spans and with prompts where the query is changed.

### Induction

For query position q, candidate key positions k satisfy:

    token[k - 1] == token[q] and 1 <= k < q

This is the standard A B ... A to B attention pattern. The score is attention
mass on all matching k positions, normalized by the same causal-uniform
baseline.

### Successor candidate

The Q/K proxy tests whether a successor token attends to configured predecessor
tokens. Supply either a predecessor-token-ID to successor-token-ID map or
explicit [query_position, predecessor_key_position] pairs.

This is screening evidence, not sufficient proof of a successor head. A
mechanistic successor-head claim should additionally show that the head OV
circuit increases the correct successor logit, or that causal head ablation
selectively harms a curated successor task.

## Notebook workflow

Open:

    notebooks/attention_head_taxonomy.ipynb

Then:

1. Point CACHE_DIR to a Qwen3 Q/K cache produced by the existing capture code.
2. Inspect the token-position table.
3. Fill target spans and retrieval query positions.
4. Leave induction query positions as None for automatic detection, or restrict
   them to a controlled repeated-pattern region.
5. Add explicit successor pairs or a token-ID map from a curated ordered list.
6. Run the scan and inspect both aggregate scores and per-query evidence.

The default safety limit is 4096 tokens because this first implementation
reconstructs one full attention matrix at a time. It never retains all heads at
once. For longer contexts, use a shorter diagnostic prompt or deliberately
raise max_full_attention_tokens after checking memory requirements.

## Command-line workflow

Create an analysis-spec JSON with these fields:

    target_spans
    retrieval_query_positions
    induction_query_positions
    successor_query_positions
    successor_token_map
    successor_pairs

Then run:

    python scripts/classify_attention_heads.py --cache-dir PATH_TO_CACHE \
      --analysis-spec PATH_TO_SPEC_JSON \
      --config configs/head_taxonomy.json \
      --output-dir outputs/head_taxonomy --device cuda

Optional comma-separated layer and head filters are available through
--layers and --heads.

## Outputs

- head_scores.csv: all scanned layer/head metrics.
- head_labels.csv: heads that pass at least one configured threshold.
- head_evidence.json: per-query candidate keys, mass, baseline, and lift.
- run_metadata.json: resolved settings, cache path, elapsed time, and the
  successor interpretation caveat.

Thresholds are hypothesis-screening parameters, not universal constants.
Validate candidates across multiple examples, matched controls, and seeds
before treating them as a stable head family.
