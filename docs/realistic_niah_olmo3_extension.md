# OLMo 3 7B extension for Realistic NIAH V2

This extension adds OLMo 3 7B without changing the completed V2 panel. The
original plan remains 29 shards and 14,500 requests.

## Registered checkpoints

| Analysis label | Checkpoint | Immutable revision | Modes | Requests |
| --- | --- | --- | --- | ---: |
| `Olmo3-7B-Instruct` | `allenai/Olmo-3-7B-Instruct` | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` | Direct, Index, Bullet | 1,500 |
| `Olmo3-7B-Think` | `allenai/Olmo-3-7B-Think` | `d97e442d7cc678210054dbcc9b440894d62c89a4` | Native Thinking | 500 |

Reports may group these checkpoint-specific rows under the logical analysis
label `Olmo3-7B`, but the raw rows, manifests, and request IDs retain the exact
checkpoint label and revision.

The extension reuses the exact 500 formal V2 stimuli:

- passage lengths: 2k, 3k, 5k, 10k, and 20k canonical tokens;
- needle counts: 1, 2, 3, 4, 5, 6, 8, 10, 20, and 30;
- seeds: 1234 through 1243;
- frozen stimuli SHA256:
  `b739122c96adf73ec6df4abe0266af239a026b4de6f09f309933231f604c7f71`.

The four prompts are unchanged from the current V2 code. Instruct modes use
the existing deterministic decoding (`temperature=0`; 64 tokens for Direct,
1,536 for Index/Bullet). Think uses `temperature=0.6`, `top_p=0.95`, and a
4,096-token output budget. Registered success still requires an exact count,
a parseable and mode-compliant response, and no length truncation.

## Environment

The OLMo 3 architecture requires `transformers>=4.57.0`. Install the pinned
inference environment before launching:

```bash
python -m pip install -r requirements-inference.txt
```

The launchers default to:

```text
repo:   /lambda/nfs/Twist-CoT-Count-Multi-Model-v2/code/Realistic_CoT_NiaH_Count
cache:  /lambda/nfs/Twist-CoT-Count-Multi-Model-v2/hf-cache
python: /home/ubuntu/venvs/realistic-niah-vllm/bin/python
```

Override them with `REALISTIC_NIAH_REPO_ROOT`,
`REALISTIC_NIAH_HF_CACHE`, and `REALISTIC_NIAH_PYTHON`.

## Smoke test

Commit the implementation first: both smoke and formal launchers require a
clean Git worktree and record the commit in every manifest.

```bash
SOURCE=/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/eight_models_formal_20260726T190349Z
SMOKE=/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/olmo3_7b_smoke_$(date -u +%Y%m%dT%H%M%SZ)
bash scripts/run_realistic_niah_olmo3_extension_smoke.sh "$SOURCE" "$SMOKE"
```

The smoke grid contains L={2k,20k}, N={6,30}, and seed 1234: 12 Instruct
requests plus 4 Think requests. Inspect:

```bash
python -m json.tool "$SMOKE/orchestration/smoke_audit.json"
```

Proceed to the formal extension only after checking raw failures and confirming
that truncation/format behavior is acceptable. The smoke audit is structural;
it intentionally does not hide scientifically valid failures.

## Formal launch

One GPU is sufficient. With two or more visible GPUs, the launcher defaults to
two workers so Instruct and Think can run concurrently. An explicit third
argument selects 1--4 workers, bounded by the visible GPU count.

```bash
SOURCE=/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/eight_models_formal_20260726T190349Z
RUN=/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/olmo3_7b_extension_$(date -u +%Y%m%dT%H%M%SZ)
bash scripts/launch_realistic_niah_olmo3_extension.sh "$SOURCE" "$RUN" 1
```

Monitor without mutating state:

```bash
tmux list-sessions
find "$RUN/orchestration/shard_state/completed" -type f -size +0c
find "$RUN/orchestration/shard_state/failed" -type f -size +0c
wc -l "$RUN"/shards/*/main/requests.jsonl
nvidia-smi
```

If a worker stops, inspect its attempt-specific log under
`orchestration/logs/`, ensure the old GPU process is gone, and relaunch the
same command with the same run root. The worker preserves prior attempt
records and resumes from the existing `requests.jsonl` request-ID checkpoint.

## Final outputs and audit

The finalizer merges only after all four shards finish and writes:

```text
models/Olmo3-7B-Instruct/main/{requests.jsonl,run_manifest.json,qc_report.json}
models/Olmo3-7B-Think/main/{requests.jsonl,run_manifest.json,qc_report.json}
models/Olmo3-7B/family_manifest.json
orchestration/final_shard_audit.json
```

`final_shard_audit.json` must report `passed=true`, 2,000 rows, and 2,000
unique request IDs. It also verifies exact task membership, stimuli SHA,
checkpoint revisions, and the frozen Git commit. Parse failures, wrong formats,
wrong counts, and truncations remain in the merged data.
