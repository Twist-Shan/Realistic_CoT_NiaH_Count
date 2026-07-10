# Repository cleanup plan

## Goal

Clean up the repo so it is easier to run, understand, and extend. Remove unused files, simplify the experiment pipeline, improve runtime where possible, and reorganize documentation.

## Phase 1: Audit only

Do not delete or rewrite code yet. First inspect the repo and produce the following analysis:

1. a file inventory,
2. likely unused files,
3. duplicated scripts/functions,
4. unclear entry points,
5. slow or redundant pipeline steps,
6. documentation problems,
7. recommended cleanup order,
8. risks.

If I specify some high-level functions to be the core for this repo project, you need to analyze (i) package dependencies, (ii) script / function dependencies, (iii) documentation clarity. You also need to lay out your plan for removing unused files and duplicated scripts/functions/variables, and consolidate loosely connections scripts/functions/variables.

## Phase 2: Safe cleanup

After the audit, make only low-risk changes:

- remove clearly generated files,
- update `.gitignore`,
- move outdated notes to `docs/archive/`,
- simplify README structure,
- identify canonical scripts/configs,
- add or update smoke-test instructions.


## Phase 3: Pipeline simplification

Simplify the main experiment path:

- one canonical entry point for local smoke tests,
- one canonical entry point for Colab/GPU runs,
- clear config files,
- explicit output directories,
- no hard-coded paths.

## Acceptance criteria

- README explains how to set up locally, run a smoke test, and run on Colab.
- Generated artifacts are ignored.
- Main scripts still run.
- Local smoke test passes.
- Codex summarizes all removed/moved files and why.