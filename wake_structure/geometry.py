"""Circular direction decoding for wake geometry."""

from __future__ import annotations

import math

import torch

from .head import split_geometry_logits


def direction_bin_centers(num_bins: int, *, device=None, dtype=None) -> torch.Tensor:
    return torch.arange(num_bins, device=device, dtype=dtype) * (2 * math.pi / num_bins)


def decode_direction(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = logits.softmax(dim=1)
    centers = direction_bin_centers(
        probabilities.shape[1], device=probabilities.device, dtype=probabilities.dtype
    ).view(1, -1, 1, 1)
    vx = (probabilities * torch.cos(centers)).sum(dim=1, keepdim=True)
    vy = (probabilities * torch.sin(centers)).sum(dim=1, keepdim=True)
    angle = torch.remainder(torch.atan2(vy, vx), 2 * math.pi)
    confidence = torch.sqrt(vx.square() + vy.square()).clamp(0, 1)
    return angle, confidence, probabilities


def decode_geometry(logits: torch.Tensor, num_bins: int) -> dict[str, torch.Tensor]:
    parts = split_geometry_logits(logits, num_bins)
    theta1, confidence1, q1 = decode_direction(parts.arm1)
    theta2, confidence2, q2 = decode_direction(parts.arm2)
    structure = parts.structure.sigmoid()
    return {
        "structure": structure,
        "tip": parts.tip.sigmoid(),
        "offset": parts.offset.sigmoid(),
        "theta1": theta1,
        "theta2": theta2,
        "confidence1": confidence1,
        "confidence2": confidence2,
        "arm1_distribution": q1,
        "arm2_distribution": q2,
        "geometry_gate": structure * torch.sqrt(confidence1 * confidence2),
    }

