# Native-reasoning model extension for Realistic NIAH V2

This extension adds five analysis groups without changing either the completed
14,500-request V2 panel or the completed OLMo 3 extension. It reuses the exact
frozen 500-stimulus V2 grid and the same Direct, Index, Bullet, and Native
Thinking query text.

## Registered checkpoints

| Analysis group | Checkpoint label | Hugging Face checkpoint | Immutable revision | Modes | Requests |
| --- | --- | --- | --- | --- | ---: |
| Nemotron Nano v2 9B | `Nemotron-Nano-v2-9B` | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `6533e8de2c68e4536bf7c411d7a3ce5734111476` | all four | 2,000 |
| Nemotron 3 Nano 4B | `Nemotron-3-Nano-4B` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` | all four | 2,000 |
| Granite 3.3 8B | `Granite-3.3-Instruct-8B` | `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | all four | 2,000 |
| Cogito v1 Preview 8B | `Cogito-v1-Preview-8B` | `deepcogito/cogito-v1-preview-llama-8B` | `64c42369b3f322fbffb277bfff146551dd2823cc` | all four | 2,000 |
| Ministral 3 8B | `Ministral-3-Instruct-8B` | `mistralai/Ministral-3-8B-Instruct-2512` | `5b26027e7b19eeb4b7352e1fed3926375dd2cb4d` | Direct, Index, Bullet | 1,500 |
| Ministral 3 8B | `Ministral-3-Reasoning-8B` | `mistralai/Ministral-3-8B-Reasoning-2512` | `81eaece1948f3875421d9a45bc55487d10e2d894` | Native Thinking | 500 |

The first four groups use one checkpoint for all four modes; their
non-thinking versus thinking comparison therefore holds weights fixed.
Ministral is intentionally different: Instruct and Reasoning are separate
post-trained checkpoints, so that comparison does not hold weights fixed.
Raw labels, revisions, manifests, and request IDs retain this distinction.

The source grid remains:

- passage lengths: 2k, 3k, 5k, 10k, and 20k canonical tokens;
- needle counts: 1, 2, 3, 4, 5, 6, 8, 10, 20, and 30;
- seeds: 1234 through 1243;
- stimuli SHA256:
  `b739122c96adf73ec6df4abe0266af239a026b4de6f09f309933231f604c7f71`.

This yields 20 independent 500-request shards and 10,000 total requests.

## Model-specific controls

The user query text is unchanged. Native reasoning is activated through each
checkpoint's documented interface:

- Nemotron Nano v2 9B: a template-level `/think` or `/no_think` system signal;
- Nemotron 3 Nano 4B: `enable_thinking=True/False`;
- Granite 3.3 8B: `thinking=True/False`;
- Cogito v1 Preview 8B: `enable_thinking=True/False`;
- Ministral 3 Reasoning: the official structured reasoning system prompt;
- Ministral 3 Instruct: no reasoning control.

Every resolved `ModelSpec`, rendered prompt, decoding configuration, engine
profile, checkpoint revision, and generated token sequence is preserved in
the request rows or manifests.

The registered Native Thinking sampling settings follow the model cards where
they specify a recommendation:

| Checkpoint | Temperature | Top-p | Output budget |
| --- | ---: | ---: | ---: |
| Nemotron Nano v2 9B | 0.6 | 0.95 | 4,096 |
| Nemotron 3 Nano 4B | 1.0 | 0.95 | 4,096 |
| Granite 3.3 8B | 0.0 | 1.0 | 4,096 |
| Cogito v1 Preview 8B | 0.6 | 0.95 | 4,096 |
| Ministral 3 Reasoning 8B | 0.7 | 0.95 | 4,096 |

Direct remains greedy with 64 output tokens. Index and Bullet remain greedy
with 1,536 output tokens. Registered success still requires an exact count, a
parseable and mode-compliant response, and no length truncation.

Nemotron hybrid checkpoints run with
`mamba_ssm_cache_dtype=float32`, as required by NVIDIA for output quality.
Ministral runs with vLLM's `mistral` tokenizer/config/load profiles and
requires `mistral-common>=1.8.6`. These resolved engine overrides are recorded
in each run manifest.

## Environment and hardware

Install the registered environment:

```bash
python -m pip install -r requirements-inference.txt
```

The launcher checks `transformers==5.5.3`, `vllm==0.25.1`, and
`mistral-common>=1.8.6`. GPU validation is still mandatory. An H100 or H200 is
the safest smoke-test target because the Ministral Instruct checkpoint is
published in FP8; do not start the formal run until all six checkpoint
initializations and all 80 smoke requests pass structural audit.

## Smoke test

Commit the implementation first. The launchers require a clean worktree so
that every row can record an immutable Git commit.

```bash
SOURCE=/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/eight_models_formal_20260726T190349Z
SMOKE=/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/reasoning_models_smoke_$(date -u +%Y%m%dT%H%M%SZ)
bash scripts/run_realistic_niah_reasoning_models_smoke.sh "$SOURCE" "$SMOKE"
```

The smoke grid uses L={2k,20k}, N={6,30}, and seed 1234. Each prompt mode has
four requests, for 80 requests total. Inspect both the aggregate audit and raw
outputs:

```bash
python -m json.tool "$SMOKE/orchestration/smoke_audit.json"
find "$SMOKE/models" -path '*/smoke/requests.jsonl' -print
```

Parse, format, wrong-count, or truncation failures are scientific outcomes;
the smoke audit reports rather than removes them. A model-initialization or
tokenizer incompatibility is an operational failure and must be fixed before
formal launch.

## Formal launch

The optional third argument selects 1--8 workers, bounded by visible GPUs. The
default uses at most six workers.

```bash
SOURCE=/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/eight_models_formal_20260726T190349Z
RUN=/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/reasoning_models_extension_$(date -u +%Y%m%dT%H%M%SZ)
bash scripts/launch_realistic_niah_reasoning_models_extension.sh "$SOURCE" "$RUN" 4
```

Workers claim mutually exclusive shards and checkpoint each completed request
by request ID. Relaunching the same command after a confirmed worker stop
preserves failed attempts and continues only missing IDs.

Monitor:

```bash
tmux list-sessions
find "$RUN/orchestration/shard_state/completed" -type f -size +0c
find "$RUN/orchestration/shard_state/failed" -type f -size +0c
wc -l "$RUN"/shards/*/main/requests.jsonl
nvidia-smi
```

## Final audit

After all 20 shards complete, the finalizer writes checkpoint-specific
canonical outputs under `models/<checkpoint-label>/main/`, one logical
`family_manifest.json` per analysis group, and
`orchestration/final_shard_audit.json`.

The final audit must report:

- `passed=true`;
- 10,000 JSONL rows and 10,000 unique request IDs;
- exact frozen stimuli and plan hashes;
- the same immutable Git commit and checkpoint revisions;
- no missing or duplicate task membership.

All wrong counts, parse failures, format failures, and truncations remain in
the canonical output.
