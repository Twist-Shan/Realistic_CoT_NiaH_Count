from __future__ import annotations

import json

import numpy as np
import pandas as pd

from realistic_niah_v5.cross_mode_geometry import compare_position_geometry


def _write_non_thinking(tmp_path, *, hidden_size=16):
    base = tmp_path / "nonthinking"
    shards = base / "shards"
    shards.mkdir(parents=True)
    index_rows = []
    rng = np.random.default_rng(1)
    for split, seeds in (("discovery", range(1, 5)), ("confirmation", range(5, 8))):
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
                    "stimulus_id": f"v4-{split}-{seed}",
                    "model_label": "Toy",
                    "shard_path": str(shard.relative_to(base)),
                }
            )
    index = base / "capture_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in index_rows))
    return index


def _write_native(tmp_path, *, hidden_size=16):
    base = tmp_path / "native"
    shards = base / "shards"
    shards.mkdir(parents=True)
    index_rows = []
    rng = np.random.default_rng(2)
    for split, seeds in (("discovery", range(1, 5)), ("confirmation", range(5, 8))):
        for seed in seeds:
            row_dir = shards / f"{split}_{seed}"
            row_dir.mkdir()
            values = np.stack(
                [
                    np.asarray(
                        [occurrence * 1.2 + rng.normal(scale=0.08) for _ in range(hidden_size)],
                        dtype=np.float16,
                    )
                    for occurrence in range(1, 11)
                ]
            )[:, None, :]
            states = row_dir / "states.npz"
            np.savez(states, layer_indices=np.asarray([0]), site_states=values)
            manifest = {
                "site_rows": [
                    {"site_kind": "item_end", "occurrence": occurrence}
                    for occurrence in range(1, 11)
                ]
            }
            manifest_path = row_dir / "capture_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            index_rows.append(
                {
                    "gold_count": 10,
                    "split": split,
                    "seed": seed,
                    "stimulus_id": f"v5-{split}-{seed}",
                    "model_label": "Toy",
                    "trace_one_to_one": True,
                    "exact_count": True,
                    "states_path": str(states.relative_to(base)),
                    "manifest_path": str(manifest_path.relative_to(base)),
                }
            )
    index = base / "capture_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in index_rows))
    return index


def test_cross_mode_geometry_uses_paired_confirmation_and_reports_covariance(tmp_path):
    paths = compare_position_geometry(
        _write_non_thinking(tmp_path),
        _write_native(tmp_path),
        tmp_path / "out",
        pca_dim=4,
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
    audit = json.loads(paths["audit"].read_text())
    assert audit["preprocessing_fit_split"] == "discovery only"
    assert audit["paired_complete_n10_seeds"]["confirmation"] == [5, 6, 7]
    cross = pd.read_csv(paths["cross_mode_trend"])
    assert cross["native_minus_nonthinking"].notna().any()
