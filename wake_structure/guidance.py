"""Structure-conditioned residual feature extraction for P3 features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .geometry import decode_orientation
from .head import split_structure_logits


class PointwiseNormAct(nn.Sequential):
    """A lightweight 1x1 projection used around directional sampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class StructureGuidedExtractor(nn.Module):
    """Enhance P3 with soft axial context selected by ``P * C``.

    The orientation field selects two bilinear samples one feature pixel away
    along the predicted axial direction. Their symmetric average is fused with
    the local feature, projected back to the P3 width, and added residually.
    A zero-initialized scalar makes the module an exact identity at startup.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_bins: int = 8,
        sampling_step: float = 1.0,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1 or hidden_channels < 1:
            raise ValueError("Guidance channel counts must be positive.")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2.")
        if sampling_step <= 0:
            raise ValueError("sampling_step must be positive.")
        self.num_bins = num_bins
        self.sampling_step = float(sampling_step)
        self.reduce = PointwiseNormAct(in_channels, hidden_channels)
        self.fuse = PointwiseNormAct(2 * hidden_channels, hidden_channels)
        self.output = nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _directional_context(self, feature: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = feature.shape
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

        dx = torch.cos(theta[:, 0]) * (2.0 * self.sampling_step / max(width - 1, 1))
        dy = torch.sin(theta[:, 0]) * (2.0 * self.sampling_step / max(height - 1, 1))
        offset = torch.stack((dx, dy), dim=-1)
        forward = F.grid_sample(
            feature,
            base + offset,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        backward = F.grid_sample(
            feature,
            base - offset,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return 0.5 * (forward + backward)

    def forward(self, feature: torch.Tensor, structure_logits: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"Expected BCHW P3 feature, got {tuple(feature.shape)}")
        presence_logits, orientation_logits = split_structure_logits(structure_logits, self.num_bins)
        if structure_logits.shape[-2:] != feature.shape[-2:]:
            raise ValueError("Structure logits and P3 feature must have the same spatial size.")

        theta, confidence, _ = decode_orientation(orientation_logits)
        gate = presence_logits.sigmoid() * confidence
        reduced = self.reduce(feature)
        directional = self._directional_context(reduced, theta)
        residual = self.output(self.fuse(torch.cat((reduced, directional), dim=1)))
        return feature + torch.tanh(self.alpha) * gate * residual
