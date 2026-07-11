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

## Attention-head taxonomy

Use `notebooks/attention_head_taxonomy.ipynb` to screen Qwen3 heads for
targeted retrieval, induction, and successor-candidate behavior from existing
Q/K caches. Reusable scoring lives in
`src/dataset_generation/qk_hook_attention/head_taxonomy.py`; see
`docs/head_taxonomy.md` for metric definitions, CLI usage, and limitations.
