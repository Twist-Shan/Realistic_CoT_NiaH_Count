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

## Registered Realistic NIAH V2

The current counting protocol is specified in
`docs/realistic_niah_v2.md` and `configs/realistic_niah_main.json`. It uses
eight reasoning-capable Qwen, Gemma, DeepSeek, and GLM models and a shared
500-stimulus grid:
five post-insertion passage lengths (`2K,3K,5K,10K,20K`), ten true needle
counts (`1,2,3,4,5,6,8,10,20,30`), and ten seeds (`1234..1243`).
Each model receives four formal prompt modes, producing 2,000 generations
per model and 16,000 generations in the complete panel.

`GLM-4-9B-0414` is registered separately as the matched non-thinking
control for `GLM-Z1-9B-0414`; it is not counted among the eight primary
models or the 16,000 primary-panel generations. `Qwen3-8B` provides the
architecture-matched non-thinking comparison for
`DeepSeek-R1-0528-Qwen3-8B`.

Build the full, content-deduplicated Paul Graham source corpus from the 218
URLs registered by NVIDIA/RULER, then freeze and audit the 500 shared
stimuli once. Put the corpus inside the run root so its text and provenance
are preserved with the experiment:

```bash
PYTHONPATH=src python scripts/sync_paul_graham_full_corpus.py \
  --output-dir /path/to/runs/realistic_niah_v2/source_corpus

PYTHONPATH=src python scripts/freeze_realistic_niah.py \
  --output-dir /path/to/runs/realistic_niah_v2/dataset \
  --haystack-dir /path/to/runs/realistic_niah_v2/source_corpus \
  --haystack-corpus-manifest \
    /path/to/runs/realistic_niah_v2/source_corpus/corpus_manifest.json \
  --haystack-source-mode multi_file_no_repeat \
  --cache-dir /path/to/hf-cache
```

The generator never repeats an individual essay. It deterministically
shuffles content-unique essays per seed and uses nested prefix windows, so
the 2K, 3K, 5K, 10K, and 20K conditions for one seed share a common filler
prefix. If the deduplicated corpus is too short, freezing fails explicitly.

Before the formal run, execute the guarded-CoT truncation smoke test on
Qwen3-8B, Gemma4-12B, DeepSeek-R1-0528-Qwen3-8B, and GLM-Z1-9B-0414.
Each model receives only the registered guarded `native_thinking` prompt on
the same 12 hard-case stimuli. The primary gate is exactly zero truncations;
accuracy, parsing, format compliance, output length, restarts, and duplicate
reasoning are retained as diagnostics.

```bash
PYTHONPATH=src python scripts/run_realistic_niah.py \
  --stimuli /path/to/runs/realistic_niah_v2/dataset/stimuli.jsonl \
  --output-dir /path/to/runs/realistic_niah_v2/smoke/Qwen3-8B \
  --model Qwen3-8B \
  --passage-lengths 2000,20000 \
  --needle-counts 6,20,30 \
  --seeds 2234,2235 \
  --prompt-modes native_thinking \
  --cache-dir /path/to/hf-cache

PYTHONPATH=src python scripts/run_realistic_niah.py \
  --stimuli /path/to/runs/realistic_niah_v2/dataset/stimuli.jsonl \
  --output-dir /path/to/runs/realistic_niah_v2/smoke/Gemma4-12B \
  --model Gemma4-12B \
  --passage-lengths 2000,20000 \
  --needle-counts 6,20,30 \
  --seeds 2234,2235 \
  --prompt-modes native_thinking \
  --cache-dir /path/to/hf-cache

PYTHONPATH=src python scripts/run_realistic_niah.py \
  --stimuli /path/to/runs/realistic_niah_v2/dataset/stimuli.jsonl \
  --output-dir /path/to/runs/realistic_niah_v2/smoke/DeepSeek-R1-0528-Qwen3-8B \
  --model DeepSeek-R1-0528-Qwen3-8B \
  --passage-lengths 2000,20000 \
  --needle-counts 6,20,30 \
  --seeds 2234,2235 \
  --prompt-modes native_thinking \
  --cache-dir /path/to/hf-cache

PYTHONPATH=src python scripts/run_realistic_niah.py \
  --stimuli /path/to/runs/realistic_niah_v2/dataset/stimuli.jsonl \
  --output-dir /path/to/runs/realistic_niah_v2/smoke/GLM-Z1-9B-0414 \
  --model GLM-Z1-9B-0414 \
  --passage-lengths 2000,20000 \
  --needle-counts 6,20,30 \
  --seeds 2234,2235 \
  --prompt-modes native_thinking \
  --cache-dir /path/to/hf-cache
```

Summarize and enforce the zero-truncation gate across the 48 smoke
generations with:

```bash
PYTHONPATH=src python scripts/summarize_realistic_niah_smoke.py \
  --requests \
    /path/to/runs/realistic_niah_v2/smoke/Qwen3-8B/requests.jsonl \
    /path/to/runs/realistic_niah_v2/smoke/Gemma4-12B/requests.jsonl \
    /path/to/runs/realistic_niah_v2/smoke/DeepSeek-R1-0528-Qwen3-8B/requests.jsonl \
    /path/to/runs/realistic_niah_v2/smoke/GLM-Z1-9B-0414/requests.jsonl \
  --config configs/realistic_niah_smoke.json \
  --analysis guarded \
  --output /path/to/runs/realistic_niah_v2/smoke/summary.json
```

Runs are resumable by stable request ID. Archive and verify a completed run
on a configured rclone Google Drive remote with
`scripts/sync_run_to_gdrive.py`. Model caches and run outputs belong outside
the Git checkout.

For production inference, install `requirements-inference.txt`; it pins the
vLLM version used by the registered runner. On the registered Lambda image,
invoke Python through `bash scripts/lambda_python.sh ...`; the wrapper also
sets the persistent Hugging Face and pip cache locations.

### OLMo 3 7B extension

The independently versioned OLMo 3 extension adds the official 7B Instruct
checkpoint for Direct/Index/Bullet and the official 7B Think checkpoint for
Native Thinking. It reuses the frozen 500 V2 stimuli but does not modify the
original 29-shard, 14,500-request formal panel. See
[`docs/realistic_niah_olmo3_extension.md`](docs/realistic_niah_olmo3_extension.md)
for immutable revisions, smoke testing, resumable launch commands, and the
2,000-request final audit.

### Five-group native-reasoning extension

The independently versioned reasoning-model extension adds switchable
Nemotron Nano v2 9B, Nemotron 3 Nano 4B, Granite 3.3 8B, and Cogito v1
Preview 8B checkpoints, plus the separate Ministral 3 8B Instruct/Reasoning
pair. It reuses the same frozen 500 V2 stimuli and four prompts without
changing the original formal panel or OLMo extension. See
[`docs/realistic_niah_reasoning_models_extension.md`](docs/realistic_niah_reasoning_models_extension.md)
for immutable revisions, model-specific thinking controls, the 80-request
smoke, the resumable 20-shard launch, and the 10,000-request final audit.

## Attention-head taxonomy

Use `notebooks/attention_head_taxonomy.ipynb` to screen Qwen3 heads for
targeted retrieval, induction, and successor-candidate behavior from existing
Q/K caches. Reusable scoring lives in
`src/dataset_generation/qk_hook_attention/head_taxonomy.py`; see
`docs/head_taxonomy.md` for metric definitions, CLI usage, and limitations.
