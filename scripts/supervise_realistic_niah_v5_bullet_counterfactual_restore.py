#!/usr/bin/env python3
"""Resume and supervise the two-model bullet-counter experiment end to end."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_LAYERS = {
    "Qwen3-8B": tuple(range(0, 33, 4)),
    "Gemma4-E4B": tuple(range(0, 41, 4)),
}


def _run(
    command: Sequence[str],
    *,
    log_path: Path,
    env: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n[supervisor-command] " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            list(command),
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _wait_for_external_qwen_cohort(
    cohort_manifest: Path,
    *,
    poll_seconds: int,
    log_path: Path,
) -> None:
    while not cohort_manifest.exists():
        process = subprocess.run(
            ["pgrep", "-af", "assemble_realistic_niah_v5_bullet_counter_cohort.py.*--model Qwen3-8B"],
            check=False,
            capture_output=True,
            text=True,
        )
        active_lines = [
            line
            for line in process.stdout.splitlines()
            if "pgrep -af" not in line and "supervise_realistic" not in line
        ]
        if not active_lines:
            raise RuntimeError(
                "Qwen cohort manifest is absent and its external generator stopped"
            )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[supervisor] waiting for Qwen cohort; active={active_lines[0]}\n"
            )
        time.sleep(int(poll_seconds))


def _cohort_command(
    *,
    python: str,
    model: str,
    cache_dir: Path,
    output: Path,
    existing: Sequence[Path],
) -> list[str]:
    return [
        python,
        "scripts/assemble_realistic_niah_v5_bullet_counter_cohort.py",
        "--model",
        model,
        "--cache-dir",
        str(cache_dir),
        "--device-map",
        "auto",
        "--torch-dtype",
        "bfloat16",
        "--attention-backend",
        "sdpa",
        "--existing-generations",
        *(str(path) for path in existing),
        "--seed-start",
        "1234",
        "--max-seed",
        "10000",
        "--resume",
        "--output",
        str(output),
    ]


def _run_model_trials(
    *,
    python: str,
    model: str,
    cache_dir: Path,
    model_root: Path,
    env: dict[str, str],
) -> None:
    cohort = model_root / "cohort"
    generations = cohort / "eligible_generations.jsonl"
    cohort_manifest = cohort / "frozen_cohort_manifest.json"
    discovery = model_root / "discovery"
    discovery_analysis = discovery / "analysis"
    confirmation = model_root / "confirmation"
    confirmation_analysis = confirmation / "analysis"
    model_log = model_root / "logs" / "supervisor.log"

    if not (discovery / "manifest.json").exists():
        _run(
            [
                python,
                "scripts/run_realistic_niah_v5_bullet_counterfactual_restore.py",
                "--model",
                model,
                "--cache-dir",
                str(cache_dir),
                "--device-map",
                "auto",
                "--torch-dtype",
                "bfloat16",
                "--attention-backend",
                "sdpa",
                "--generations",
                str(generations),
                "--cohort-manifest",
                str(cohort_manifest),
                "--phase",
                "discovery",
                "--source-layers",
                *(str(value) for value in DISCOVERY_LAYERS[model]),
                "--resume",
                "--output",
                str(discovery),
            ],
            log_path=model_log,
            env=env,
        )
    if not (discovery_analysis / "frozen_layers.json").exists():
        _run(
            [
                python,
                "scripts/analyze_realistic_niah_v5_bullet_counterfactual_restore.py",
                "--input",
                str(discovery),
                "--phase",
                "discovery",
                "--output",
                str(discovery_analysis),
            ],
            log_path=model_log,
            env=env,
        )
    frozen_path = discovery_analysis / "frozen_layers.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    layers = tuple(int(value) for value in frozen["source_layers"])
    if len(layers) != 3:
        raise RuntimeError(f"{model} discovery did not freeze exactly three layers")

    if not (confirmation / "manifest.json").exists():
        _run(
            [
                python,
                "scripts/run_realistic_niah_v5_bullet_counterfactual_restore.py",
                "--model",
                model,
                "--cache-dir",
                str(cache_dir),
                "--device-map",
                "auto",
                "--torch-dtype",
                "bfloat16",
                "--attention-backend",
                "sdpa",
                "--generations",
                str(generations),
                "--cohort-manifest",
                str(cohort_manifest),
                "--phase",
                "confirmation",
                "--source-layers",
                *(str(value) for value in layers),
                "--resume",
                "--output",
                str(confirmation),
            ],
            log_path=model_log,
            env=env,
        )
    if not (confirmation_analysis / "analysis.json").exists():
        _run(
            [
                python,
                "scripts/analyze_realistic_niah_v5_bullet_counterfactual_restore.py",
                "--input",
                str(confirmation),
                "--phase",
                "confirmation",
                "--frozen-layers",
                str(frozen_path),
                "--output",
                str(confirmation_analysis),
            ],
            log_path=model_log,
            env=env,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--native-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if int(args.poll_seconds) < 10:
        raise ValueError("Supervisor polling interval must be at least ten seconds")

    python = sys.executable
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    supervisor_log = args.output_root / "supervisor.log"
    supervisor_log.parent.mkdir(parents=True, exist_ok=True)

    qwen_root = args.output_root / "Qwen3-8B"
    _wait_for_external_qwen_cohort(
        qwen_root / "cohort" / "frozen_cohort_manifest.json",
        poll_seconds=int(args.poll_seconds),
        log_path=supervisor_log,
    )
    _run_model_trials(
        python=python,
        model="Qwen3-8B",
        cache_dir=args.cache_dir,
        model_root=qwen_root,
        env=env,
    )

    gemma_root = args.output_root / "Gemma4-E4B"
    gemma_cohort = gemma_root / "cohort"
    if not (gemma_cohort / "frozen_cohort_manifest.json").exists():
        gemma_existing = [
            args.native_run_root / "Gemma4-E4B" / "generations.jsonl",
            args.native_run_root
            / "one_to_one_supplement"
            / "Gemma4-E4B"
            / "batches"
            / "seed_1264_1293"
            / "generations.jsonl",
            args.native_run_root
            / "one_to_one_supplement"
            / "Gemma4-E4B"
            / "batches"
            / "seed_1294_1323"
            / "generations.jsonl",
            args.native_run_root
            / "one_to_one_supplement"
            / "Gemma4-E4B"
            / "batches"
            / "seed_1324_1353"
            / "generations.jsonl",
        ]
        if not all(path.exists() for path in gemma_existing):
            missing = [str(path) for path in gemma_existing if not path.exists()]
            raise FileNotFoundError(f"Missing Gemma frozen generation inputs: {missing}")
        _run(
            _cohort_command(
                python=python,
                model="Gemma4-E4B",
                cache_dir=args.cache_dir,
                output=gemma_cohort,
                existing=gemma_existing,
            ),
            log_path=gemma_root / "logs" / "supervisor.log",
            env=env,
        )
    _run_model_trials(
        python=python,
        model="Gemma4-E4B",
        cache_dir=args.cache_dir,
        model_root=gemma_root,
        env=env,
    )
    with supervisor_log.open("a", encoding="utf-8") as handle:
        handle.write("[supervisor] COMPLETE: both models passed all pipeline stages\n")


if __name__ == "__main__":
    main()
