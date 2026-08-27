"""Axial (180-degree periodic) orientation utilities."""

from __future__ import annotations

import math

import torch

from .head import split_structure_logits


def orientation_bin_centers(
    num_bins: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return equally spaced centers in ``[0, pi)``."""

    return torch.arange(num_bins, device=device, dtype=dtype) * (math.pi / num_bins)


def orientation_moments(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return axial angle, concentration, and normalized probabilities.

    Doubling each bin angle maps a line orientation (period pi) onto the unit
    circle (period 2*pi). The resultant-vector length is the confidence ``C``.
    """

    if probabilities.ndim != 4:
        raise ValueError("Orientation probabilities must have shape [B, K, H, W].")
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
    centers = orientation_bin_centers(
        probabilities.shape[1], device=probabilities.device, dtype=probabilities.dtype
    ).view(1, -1, 1, 1)
    vx = (probabilities * torch.cos(2 * centers)).sum(dim=1, keepdim=True)
    vy = (probabilities * torch.sin(2 * centers)).sum(dim=1, keepdim=True)
    theta = torch.remainder(0.5 * torch.atan2(vy, vx), math.pi)
    confidence = torch.sqrt(vx.square() + vy.square()).clamp(0, 1)
    return theta, confidence, probabilities


def decode_orientation(orientation_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode direction logits into ``theta`` (radians), confidence, and ``q_theta``."""

    return orientation_moments(orientation_logits.softmax(dim=1))


def decode_structure(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode raw 1+K Structure Head channels into P, q_theta, theta, and C."""

    presence_logits, orientation_logits = split_structure_logits(logits)
    theta, confidence, probabilities = decode_orientation(orientation_logits)
    presence = presence_logits.sigmoid()
    return {
        "presence": presence,
        "orientation_distribution": probabilities,
        "theta": theta,
        "confidence": confidence,
        "directional_gate": presence * confidence,
    }


def axial_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Smallest absolute angular difference for line orientations, in radians."""

    difference = torch.remainder(first - second + math.pi / 2, math.pi) - math.pi / 2
    return difference.abs()

