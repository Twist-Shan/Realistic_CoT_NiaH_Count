from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from realistic_niah_v5.cross_mode_geometry import compare_position_geometry


def _write_non_thinking(
    tmp_path,
    *,
    hidden_size=32,
    discovery_seeds=range(1, 5),
    confirmation_seeds=range(5, 8),
):
    base = tmp_path / "nonthinking"
    shards = base / "shards"
    shards.mkdir(parents=True)
    index_rows = []
    rng = np.random.default_rng(1)
    for split, seeds in (
        ("discovery", discovery_seeds),
        ("confirmation", confirmation_seeds),
    ):
        for seed in seeds:
            values = np.stack(
                [
                    np.asarray(
                        [occurrence * 0.8 + rng.normal(scale=0.1) for _ in range(hidden_size)],
                        dtype=np.float16,
                    )
                    for occurrence in range(1, 11)
                ]
            )[None, :, :]
            shard = shards / f"{split}_{seed}.npz"
            np.savez(shard, layer_indices=np.asarray([0]), span_end=values)
            index_rows.append(
                {
                    "design_variant": "v4.4",
                    "count": 10,
                    "split": split,
                    "seed": seed,
                    "stimulus_id": f"stimulus-{split}-{seed}",
                    "model_label": "Toy",
                    "shard_path": str(shard.relative_to(base)),
                }
            )
    index = base / "capture_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in index_rows))
    return index


def _write_native(
    tmp_path,
    *,
    hidden_size=32,
    discovery_seeds=range(1, 5),
    confirmation_seeds=range(5, 8),
    occurrences_by_seed=None,
):
    base = tmp_path / "native"
    shards = base / "shards"
    shards.mkdir(parents=True)
    index_rows = []
    rng = np.random.default_rng(2)
    occurrences_by_seed = occurrences_by_seed or {}
    for split, seeds in (
        ("discovery", discovery_seeds),
        ("confirmation", confirmation_seeds),
    ):
        for seed in seeds:
            occurrences = tuple(occurrences_by_seed.get(seed, range(1, 11)))
            row_dir = shards / f"{split}_{seed}"
            row_dir.mkdir()
            values = np.stack(
                [
                    np.asarray(
                        [occurrence * 1.2 + rng.normal(scale=0.08) for _ in range(hidden_size)],
                        dtype=np.float16,
                    )
                    for occurrence in occurrences
                ]
            )[:, None, :]
            states = row_dir / "states.npz"
            np.savez(states, layer_indices=np.asarray([0]), site_states=values)
            manifest = {
                "site_rows": [
                    {"site_kind": "item_end", "occurrence": occurrence}
                    for occurrence in occurrences
                ]
            }
            manifest_path = row_dir / "capture_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            index_rows.append(
                {
                    "gold_count": 10,
                    "split": split,
                    "seed": seed,
                    "stimulus_id": f"stimulus-{split}-{seed}",
                    "model_label": "Toy",
                    "trace_one_to_one": len(occurrences) == 10,
                    "exact_count": True,
                    "states_path": str(states.relative_to(base)),
                    "manifest_path": str(manifest_path.relative_to(base)),
                }
            )
    index = base / "capture_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in index_rows))
    return index


def test_cross_mode_geometry_uses_matched_seed_panel_and_reports_covariance(tmp_path):
    paths = compare_position_geometry(
        _write_non_thinking(tmp_path),
        _write_native(tmp_path),
        tmp_path / "out",
        # With three confirmation seeds, each delete-one replicate has only
        # two observations per class.  PCA16 therefore exercises the singular
        # within-covariance path used by the real Gemma E3 analysis.
        pca_dim=16,
        layers=[0],
    )
    per_class = pd.read_csv(paths["per_class"])
    assert set(per_class["mode"]) == {"non_thinking", "native_thinking"}
    assert {"min_bhattacharyya", "class_nc1_ratio", "logistic_recall"}.issubset(
        set(per_class["metric"])
    )
    global_metrics = pd.read_csv(paths["global"])
    assert global_metrics["pillai_trace_regularized"].notna().all()
    assert global_metrics["nc1_trace_sigmaw_sigmab_pinv_over_c"].notna().all()
    assert (global_metrics["covariance_ridge"] > 0).all()
    assert (global_metrics["covariance_ridge_attempts"] >= 1).all()
    audit = json.loads(paths["audit"].read_text())
    assert audit["preprocessing_fit_split"] == "discovery only"
    assert audit["analysis_design"] == "fixed_registered_seed_panel_observed_positions"
    assert audit["registered_seed_panel"]["confirmation"] == [5, 6, 7]
    cross = pd.read_csv(paths["cross_mode_trend"])
    assert cross["native_minus_nonthinking"].notna().any()


def test_cross_mode_geometry_keeps_partial_traces_on_full_seed_panel(tmp_path):
    confirmation_seeds = range(5, 9)
    paths = compare_position_geometry(
        _write_non_thinking(tmp_path, confirmation_seeds=confirmation_seeds),
        _write_native(
            tmp_path,
            confirmation_seeds=confirmation_seeds,
            occurrences_by_seed={2: range(1, 9), 6: range(1, 9)},
        ),
        tmp_path / "out_ragged",
        pca_dim=8,
        layers=[0],
    )
    audit = json.loads(paths["audit"].read_text())
    assert audit["registered_seed_panel"] == {
        "discovery": [1, 2, 3, 4],
        "confirmation": [5, 6, 7, 8],
    }
    assert audit["position_support"]["native_thinking"]["discovery"]["10"] == 3
    assert audit["position_support"]["native_thinking"]["confirmation"]["10"] == 3
    assert audit["position_support"]["non_thinking"]["confirmation"]["10"] == 4

    global_metrics = pd.read_csv(paths["global"])
    native = global_metrics.loc[global_metrics["mode"].eq("native_thinking")].iloc[0]
    nonthinking = global_metrics.loc[global_metrics["mode"].eq("non_thinking")].iloc[0]
    assert int(native["n_confirmation"]) == 38
    assert int(native["min_confirmation_per_class"]) == 3
    assert int(nonthinking["n_confirmation"]) == 40

    per_class = pd.read_csv(paths["per_class"])
    late_native = per_class.loc[
        per_class["mode"].eq("native_thinking")
        & per_class["occurrence"].eq(10)
    ]
    assert set(late_native["n_confirmation_class"]) == {3}


def test_cross_mode_geometry_rejects_same_seed_with_different_stimulus(tmp_path):
    nonthinking = _write_non_thinking(tmp_path)
    native = _write_native(tmp_path)
    rows = [json.loads(line) for line in native.read_text().splitlines()]
    rows[0]["stimulus_id"] = "different-stimulus"
    native.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="cross-mode stimulus mismatches"):
        compare_position_geometry(
            nonthinking,
            native,
            tmp_path / "out_mismatch",
            pca_dim=8,
            layers=[0],
        )
