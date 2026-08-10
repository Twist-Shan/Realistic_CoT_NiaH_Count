from __future__ import annotations

import numpy as np
import pytest
import torch

from realistic_niah_v4.layerwise_removal import (
    fit_prompt_removal_geometry,
    make_prompt_removal_transform,
    normmatched_orthogonal_delta,
    projected_delta,
)


def test_candidate_removal_and_control_have_registered_geometry() -> None:
    rng = np.random.default_rng(448)
    basis, _ = np.linalg.qr(rng.normal(size=(16, 6)))
    target = torch.from_numpy(basis[:, :3].astype(np.float32))
    control = torch.from_numpy(basis[:, 3:6].astype(np.float32))
    selected = torch.from_numpy(rng.normal(size=(2, 7, 16)).astype(np.float32))

    candidate = projected_delta(selected, target)
    orthogonal, candidate_again = normmatched_orthogonal_delta(
        selected, target, control
    )

    assert torch.allclose(candidate, candidate_again)
    assert torch.allclose(
        torch.linalg.vector_norm(candidate, dim=(1, 2)),
        torch.linalg.vector_norm(orthogonal, dim=(1, 2)),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.linalg.vector_norm(orthogonal @ target).item() < 1e-5
    removed = selected - candidate
    assert torch.linalg.vector_norm(projected_delta(removed, target)).item() < 1e-5


def test_geometry_and_transform_measurements() -> None:
    rng = np.random.default_rng(449)
    labels = np.repeat(np.arange(1, 11), 12)
    class_signal = rng.normal(size=(10, 3))
    axes, _ = np.linalg.qr(rng.normal(size=(14, 6)))
    states = class_signal[labels - 1] @ axes[:, :3].T
    states += rng.normal(scale=0.2, size=(len(labels), 14))
    geometry = fit_prompt_removal_geometry(states, labels, rank=3)
    assert geometry.basis.shape == (14, 3)
    assert geometry.control_basis.shape == (14, 3)
    assert np.linalg.norm(geometry.basis.T @ geometry.control_basis) < 1e-7

    selected = torch.from_numpy(states[:5][None].astype(np.float32))
    measurements: dict[str, float] = {}
    transform = make_prompt_removal_transform(
        geometry, "actual_normmatched_orthogonal", measurements
    )
    replacement = transform(selected)
    assert replacement.shape == selected.shape
    assert measurements["norm_ratio"] == pytest.approx(1.0, rel=1e-6)


def test_invalid_condition_is_rejected() -> None:
    rng = np.random.default_rng(450)
    labels = np.repeat(np.arange(1, 11), 5)
    states = rng.normal(size=(len(labels), 12))
    geometry = fit_prompt_removal_geometry(states, labels, rank=3)
    with pytest.raises(ValueError, match="unsupported"):
        make_prompt_removal_transform(geometry, "wrong", {})
