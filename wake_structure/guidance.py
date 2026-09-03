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
        sampling_steps: tuple[float, ...] = (1.0, 2.0, 4.0),
        confidence_floor: float = 0.2,
        enable_denoising: bool = True,
        enable_directional_extraction: bool = True,
        denoise_scale_init: float = 0.0,
        feature_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.sampling_steps = tuple(float(step) for step in sampling_steps)
        self.confidence_floor = float(confidence_floor)
        self.enable_denoising = enable_denoising
        self.enable_directional_extraction = enable_directional_extraction
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

    def _sample(self, feature: torch.Tensor, theta: torch.Tensor, step: float) -> torch.Tensor:
        batch, _, height, width = feature.shape
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        dx = torch.cos(theta[:, 0]) * (2 * step / max(width - 1, 1))
        dy = torch.sin(theta[:, 0]) * (2 * step / max(height - 1, 1))
        offset = torch.stack((dx, dy), dim=-1)
        return F.grid_sample(feature, base + offset, mode="bilinear", padding_mode="border", align_corners=True)

    def _sample_arm(self, feature: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        steps = getattr(self, "sampling_steps", (getattr(self, "sampling_step", 1.0),))
        samples = [self._sample(feature, theta, step) for step in steps]
        return torch.stack(samples).mean(dim=0)

    def decode_gates(self, geometry_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = split_geometry_logits(geometry_logits, self.num_bins)
        theta1, confidence1, _ = decode_direction(parts.arm1)
        theta2, confidence2, _ = decode_direction(parts.arm2)
        structure = parts.structure.sigmoid()
        direction_confidence = torch.sqrt(confidence1 * confidence2)
        confidence_floor = getattr(self, "confidence_floor", 0.0)
        effective_confidence = confidence_floor + (1 - confidence_floor) * direction_confidence
        return {
            "structure": structure,
            "theta1": theta1,
            "theta2": theta2,
            "direction_confidence": direction_confidence,
            "denoise_gate": 1 - structure,
            "enhancement_gate": structure * effective_confidence,
        }

    def _components(self, feature: torch.Tensor, geometry_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        gates = self.decode_gates(geometry_logits)
        enable_denoising = getattr(self, "enable_denoising", True)
        enable_directional_extraction = getattr(self, "enable_directional_extraction", True)
        estimated_noise = self.noise(feature) if enable_denoising else torch.zeros_like(feature)
        clean = (
            feature - torch.tanh(self.denoise_scale) * gates["denoise_gate"] * estimated_noise
            if enable_denoising
            else feature
        )
        if enable_directional_extraction:
            reduced = self.reduce(clean)
            arm1 = self._sample_arm(reduced, gates["theta1"])
            arm2 = self._sample_arm(reduced, gates["theta2"])
            signal = self.signal(self.fuse(torch.cat((reduced, arm1, arm2), dim=1)))
            output = clean + torch.tanh(self.feature_scale) * gates["enhancement_gate"] * signal
        else:
            output = clean
        return {**gates, "estimated_noise": estimated_noise, "clean": clean, "output": output}

    def forward(self, feature: torch.Tensor, geometry_logits: torch.Tensor) -> torch.Tensor:
        return self._components(feature, geometry_logits)["output"]

    @torch.no_grad()
    def diagnostic_maps(self, feature: torch.Tensor, geometry_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        components = self._components(feature, geometry_logits)
        return {
            "denoise_gate": components["denoise_gate"],
            "enhancement_gate": components["enhancement_gate"],
            "direction_confidence": components["direction_confidence"],
            "noise_magnitude": components["estimated_noise"].abs().mean(dim=1, keepdim=True),
            "feature_change": (components["output"] - feature).abs().mean(dim=1, keepdim=True),
        }
