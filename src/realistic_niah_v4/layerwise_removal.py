"""Discovery-frozen count-subspace removal and norm-matched controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, MutableMapping

import numpy as np
import torch


@dataclass(frozen=True)
class PromptRemovalGeometry:
    classes: np.ndarray
    centroids: np.ndarray
    basis: np.ndarray
    control_basis: np.ndarray
    centroid_variance_capture: float


def fit_prompt_removal_geometry(
    states: np.ndarray,
    counts: np.ndarray,
    *,
    rank: int = 3,
    required_classes: np.ndarray | None = None,
) -> PromptRemovalGeometry:
    """Fit the count-centroid basis and an orthogonal within-count control."""

    values = np.asarray(states, dtype=np.float64)
    labels = np.asarray(counts)
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("states and counts must be row matched")
    if not np.isfinite(values).all():
        raise ValueError("states contain non-finite values")
    classes = np.unique(labels) if required_classes is None else np.asarray(required_classes)
    if len(classes) <= rank:
        raise ValueError("too few classes for requested rank")
    missing = [value for value in classes if not np.any(labels == value)]
    if missing:
        raise ValueError(f"missing classes: {missing}")
    centroids = np.stack([values[labels == value].mean(0) for value in classes])
    centered = centroids - centroids.mean(0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular[rank - 1] <= 1e-12:
        raise ValueError("count-centroid geometry is rank deficient")
    basis = vt[:rank].T

    lookup = {value: centroids[index] for index, value in enumerate(classes)}
    residual = values - np.stack([lookup[value] for value in labels])
    residual -= (residual @ basis) @ basis.T
    _, control_singular, control_vt = np.linalg.svd(residual, full_matrices=False)
    if control_singular[rank - 1] <= 1e-12:
        raise ValueError("within-count control geometry is rank deficient")
    control = control_vt[:rank].T
    control -= basis @ (basis.T @ control)
    control, _ = np.linalg.qr(control, mode="reduced")
    control = control[:, :rank]
    if np.linalg.norm(basis.T @ control, ord="fro") > 1e-7:
        raise RuntimeError("control basis is not orthogonal to count basis")
    total = float(np.square(singular).sum())
    capture = float(np.square(singular[:rank]).sum() / max(total, 1e-24))
    return PromptRemovalGeometry(
        classes=np.asarray(classes).copy(),
        centroids=centroids,
        basis=basis,
        control_basis=control,
        centroid_variance_capture=capture,
    )


def projected_delta(selected: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project within-prompt endpoint differences onto a frozen basis."""

    if selected.ndim != 3:
        raise ValueError("selected states must have shape [batch, positions, hidden]")
    axes = basis.to(device=selected.device, dtype=selected.dtype)
    if axes.ndim != 2 or axes.shape[0] != selected.shape[-1]:
        raise ValueError("basis width does not match selected states")
    centered = selected - selected.mean(dim=1, keepdim=True)
    return (centered @ axes) @ axes.T


def normmatched_orthogonal_delta(
    selected: torch.Tensor,
    target_basis: torch.Tensor,
    control_basis: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a control delta with the candidate norm in every batch item."""

    target = projected_delta(selected, target_basis)
    nuisance = projected_delta(selected, control_basis)
    target_norm = torch.linalg.vector_norm(target, dim=(1, 2), keepdim=True)
    nuisance_norm = torch.linalg.vector_norm(nuisance, dim=(1, 2), keepdim=True)
    if torch.any((nuisance_norm <= epsilon) & (target_norm > epsilon)):
        raise RuntimeError("orthogonal control has zero norm for a nonzero target")
    scaled = nuisance * target_norm / torch.clamp(nuisance_norm, min=epsilon)
    return scaled, target


def make_prompt_removal_transform(
    geometry: PromptRemovalGeometry,
    condition: str,
    measurements: MutableMapping[str, float],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a hook transform and record its realized Frobenius norm."""

    basis = torch.from_numpy(geometry.basis.astype(np.float32))
    control = torch.from_numpy(geometry.control_basis.astype(np.float32))
    if condition not in {"actual_rank3_remove", "actual_normmatched_orthogonal"}:
        raise ValueError(f"unsupported removal condition: {condition}")

    def transform(selected: torch.Tensor) -> torch.Tensor:
        if condition == "actual_rank3_remove":
            delta = projected_delta(selected, basis)
            target = delta
        else:
            delta, target = normmatched_orthogonal_delta(selected, basis, control)
        realized = torch.linalg.vector_norm(delta).detach().float().cpu().item()
        target_norm = torch.linalg.vector_norm(target).detach().float().cpu().item()
        measurements["removed_fro_norm"] = float(realized)
        measurements["target_removed_fro_norm"] = float(target_norm)
        measurements["norm_ratio"] = float(realized / max(target_norm, 1e-12))
        return selected - delta

    return transform
