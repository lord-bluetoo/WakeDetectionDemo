"""Geometry-guided feature denoising and directional extraction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .geometry import decode_direction
from .head import split_geometry_logits


class PointwiseNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class GeometryGuidedRefinement(nn.Module):
    """Suppress background clutter and extract context along two predicted wake arms."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_bins: int,
        sampling_step: float = 1.0,
        denoise_scale_init: float = 0.0,
        feature_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.sampling_step = float(sampling_step)
        self.reduce = PointwiseNormAct(in_channels, hidden_channels)
        self.noise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
        )
        self.fuse = PointwiseNormAct(3 * hidden_channels, hidden_channels)
        self.signal = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        self.denoise_scale = nn.Parameter(torch.tensor(float(denoise_scale_init)))
        self.feature_scale = nn.Parameter(torch.tensor(float(feature_scale_init)))

    def _sample(self, feature: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = feature.shape
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        dx = torch.cos(theta[:, 0]) * (2 * self.sampling_step / max(width - 1, 1))
        dy = torch.sin(theta[:, 0]) * (2 * self.sampling_step / max(height - 1, 1))
        offset = torch.stack((dx, dy), dim=-1)
        return F.grid_sample(feature, base + offset, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, feature: torch.Tensor, geometry_logits: torch.Tensor) -> torch.Tensor:
        parts = split_geometry_logits(geometry_logits, self.num_bins)
        theta1, confidence1, _ = decode_direction(parts.arm1)
        theta2, confidence2, _ = decode_direction(parts.arm2)
        gate = parts.structure.sigmoid() * torch.sqrt(confidence1 * confidence2)

        clean = feature - torch.tanh(self.denoise_scale) * (1 - gate) * self.noise(feature)
        reduced = self.reduce(clean)
        arm1 = self._sample(reduced, theta1)
        arm2 = self._sample(reduced, theta2)
        signal = self.signal(self.fuse(torch.cat((reduced, arm1, arm2), dim=1)))
        return clean + torch.tanh(self.feature_scale) * gate * signal
