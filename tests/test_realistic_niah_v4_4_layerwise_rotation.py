from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from realistic_niah_v4.layerwise_rotation import (
    align_resampled_map_to_reference,
    evaluate_layer_map,
    fit_centroid_geometry,
    fit_layer_map,
    orthogonal_procrustes,
    polar_factors,
    principal_angles_degrees,
    proper_rotation_geodesic_degrees,
)


def _orthogonal(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(rows, columns)))
    return q[:, :columns]


def _paired_synthetic_states() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(446)
    classes = np.arange(2, 12)
    repeats = 20
    ambient = 12
    source_basis = _orthogonal(rng, ambient, 3)
    target_basis = _orthogonal(rng, ambient, 3)
    angle = np.deg2rad(28.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform = rotation @ np.diag([1.25, 0.9, 0.7])
    class_coordinates = rng.normal(size=(len(classes), 3))
    class_coordinates -= class_coordinates.mean(axis=0, keepdims=True)
    labels = np.repeat(classes, repeats)
    source_coordinates = np.repeat(class_coordinates, repeats, axis=0)
    source_coordinates += rng.normal(scale=0.015, size=source_coordinates.shape)
    target_coordinates = source_coordinates @ transform
    target_coordinates += rng.normal(scale=0.01, size=target_coordinates.shape)
    source = source_coordinates @ source_basis.T
    target = target_coordinates @ target_basis.T
    source += rng.normal(scale=0.002, size=source.shape)
    target += rng.normal(scale=0.002, size=target.shape)
    return source, target, labels


def test_layer_map_predicts_a_known_low_rank_transport() -> None:
    source, target, labels = _paired_synthetic_states()
    fit = fit_layer_map(
        source,
        target,
        labels,
        rank=3,
        ridge_relative_scale=1e-6,
    )
    metrics = evaluate_layer_map(fit, source, target)

    assert fit.training_r2 > 0.999
    assert metrics["r2"] > 0.995
    assert metrics["normalized_rmse"] < 0.08
    assert np.linalg.det(fit.proper_rotation) == pytest.approx(1.0, abs=1e-10)
    assert np.all(np.linalg.eigvalsh(fit.stretch_factor) >= -1e-10)


def test_gauge_alignment_removes_arbitrary_basis_rotations() -> None:
    source, target, labels = _paired_synthetic_states()
    reference = fit_layer_map(source, target, labels, rank=3)
    rng = np.random.default_rng(447)
    source_gauge = _orthogonal(rng, 3, 3)
    target_gauge = _orthogonal(rng, 3, 3)
    transformed_matrix = source_gauge.T @ reference.matrix @ target_gauge
    orthogonal, stretch, proper, determinant = polar_factors(transformed_matrix)
    resampled = replace(
        reference,
        source=replace(
            reference.source, basis=reference.source.basis @ source_gauge
        ),
        target=replace(
            reference.target, basis=reference.target.basis @ target_gauge
        ),
        matrix=transformed_matrix,
        orthogonal_factor=orthogonal,
        stretch_factor=stretch,
        proper_rotation=proper,
        orthogonal_determinant=determinant,
    )

    aligned, source_alignment, target_alignment = align_resampled_map_to_reference(
        resampled, reference
    )

    assert np.allclose(source_gauge @ source_alignment, np.eye(3), atol=1e-10)
    assert np.allclose(target_gauge @ target_alignment, np.eye(3), atol=1e-10)
    assert np.allclose(aligned, reference.matrix, atol=1e-10)


def test_subspace_and_rotation_angles_have_expected_values() -> None:
    left = np.eye(5)[:, :3]
    right = left.copy()
    assert np.allclose(principal_angles_degrees(left, right), 0.0)

    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert proper_rotation_geodesic_degrees(np.eye(3), rotation) == pytest.approx(
        37.0, abs=1e-10
    )


def test_geometry_and_procrustes_reject_invalid_inputs() -> None:
    source, _, labels = _paired_synthetic_states()
    with pytest.raises(ValueError, match="rank deficient"):
        fit_centroid_geometry(np.ones_like(source), labels, rank=3)
    with pytest.raises(ValueError, match="equally shaped"):
        orthogonal_procrustes(np.eye(3), np.eye(2))
    with pytest.raises(ValueError, match="proper rotations"):
        proper_rotation_geodesic_degrees(np.diag([-1.0, 1.0, 1.0]), np.eye(3))
