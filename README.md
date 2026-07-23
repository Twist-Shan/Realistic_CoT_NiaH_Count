# Realistic CoT NIAH Count

This repository consolidates `NIAH_repo_and_local_runs_001` and `NIAH_repo_and_local_runs_002` without deleting or modifying either source.

Start with `notebooks/`, `src/`, `scripts/`, and `configs/`. The original instructions are preserved in `docs/README.upstream.md`, and the exact merge mapping is in `provenance/MERGE_MANIFEST.md`.

Large retained outputs and environment snapshots live in `artifacts/` and `archive/`; they remain local and are excluded from normal Git tracking.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Remote: `https://github.com/Twist-Shan/Realistic_CoT_NiaH_Count.git`

## Registered Realistic NIAH pilot

The multi-model counting experiment is defined in `plans/outline.md`. Its
master grid uses post-insertion passage lengths of 2K, 5K, and 10K tokens,
needle counts `1,2,3,4,5,6,8,10,20,30`, and paired seeds `1234..1238`.

Freeze and audit the 150 shared stimuli once:

```bash
PYTHONPATH=src python scripts/freeze_realistic_niah.py \
  --output-dir /path/to/runs/realistic_niah_v1/dataset \
  --cache-dir /path/to/hf-cache
```

Run the 36-request Qwen3-8B smoke test with offline vLLM:

```bash
PYTHONPATH=src python scripts/run_realistic_niah.py \
  --stimuli /path/to/runs/realistic_niah_v1/dataset/stimuli.jsonl \
  --output-dir /path/to/runs/realistic_niah_v1/Qwen_Qwen3-8B_smoke \
  --model Qwen3-8B \
  --passage-lengths 2000,10000 \
  --needle-counts 5,6,30 \
  --seeds 1234 \
  --cache-dir /path/to/hf-cache
```

Runs are resumable by stable request ID. Archive and verify a completed run
on a configured rclone Google Drive remote with
`scripts/sync_run_to_gdrive.py`. Model caches and run outputs belong outside
the Git checkout.

For production inference, install `requirements-inference.txt`; it pins the
vLLM version used by the registered runner. On the registered Lambda image,
invoke Python through `bash scripts/lambda_python.sh ...`; the wrapper also
sets the persistent Hugging Face and pip cache locations.

## Attention-head taxonomy

Use `notebooks/attention_head_taxonomy.ipynb` to screen Qwen3 heads for
targeted retrieval, induction, and successor-candidate behavior from existing
Q/K caches. Reusable scoring lives in
`src/dataset_generation/qk_hook_attention/head_taxonomy.py`; see
`docs/head_taxonomy.md` for metric definitions, CLI usage, and limitations.
