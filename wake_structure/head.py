"""Wake geometry prediction head."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


@dataclass
class GeometryLogits:
    structure: torch.Tensor
    tip: torch.Tensor
    offset: torch.Tensor
    arm1: torch.Tensor
    arm2: torch.Tensor


class GeometryHead(nn.Module):
    """Predict wake structure, tip, sub-cell offset, and two Kelvin-arm directions."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_bins: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_bins = num_bins
        layers: list[nn.Module] = [
            ConvNormAct(in_channels, hidden_channels),
            ConvNormAct(hidden_channels, hidden_channels),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.features = nn.Sequential(*layers)
        self.output = nn.Conv2d(hidden_channels, 4 + 2 * num_bins, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)
        self.output.bias.data[:2] = math.log(0.01 / 0.99)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(feature))


def split_geometry_logits(logits: torch.Tensor, num_bins: int) -> GeometryLogits:
    if logits.ndim != 4 or logits.shape[1] != 4 + 2 * num_bins:
        raise ValueError(f"Expected [B,{4 + 2 * num_bins},H,W] geometry logits, got {tuple(logits.shape)}")
    return GeometryLogits(
        structure=logits[:, 0:1],
        tip=logits[:, 1:2],
        offset=logits[:, 2:4],
        arm1=logits[:, 4 : 4 + num_bins],
        arm2=logits[:, 4 + num_bins :],
    )

