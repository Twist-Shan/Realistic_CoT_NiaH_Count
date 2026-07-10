# Merge manifest

The consolidated directory was created non-destructively from `NIAH_repo_and_local_runs_001` and `NIAH_repo_and_local_runs_002`. Neither source directory was moved, rewritten, or deleted.

| Source | Consolidated destination |
| --- | --- |
| `001/NIAH` except nested `.git` and `.venv` | repository root |
| `001/run_results` | `artifacts/run_results` |
| `001/invalid_runs` | `artifacts/invalid_runs` |
| `001/main_findings` | `reports/main_findings` |
| `001/related papers` | `references/related_papers` |
| `001/steering_run_analysis` | `analyses/steering_run_analysis` |
| `001/read_only_files` | `archive/read_only_files` |
| `001/research-plan.md` | `docs/research-plan.md` |
| `001/gather_run_results.md` | `docs/gather_run_results.md` |
| `001/NIAH-counting.html` | `reports/NIAH-counting.html` |
| `001/NIAH/.git` | `archive/source_git_metadata/niah_001_git_dir` |
| `001/NIAH/.venv` | `archive/env001` |
| `002/NIAH/.venv` | `archive/env002` |

The original project README is retained as `docs/README.upstream.md`.

The first 001 environment copy used a long destination path and encountered Windows path-length limits. A second copy was made to the shorter `archive/env001` path; both are retained. Recreate a clean environment from `requirements.txt` for reproducible execution.
