"""The deliberately small first version of the wake Structure Head."""

from __future__ import annotations

import math

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class StructureHead(nn.Module):
    """Predict one wake-presence logit and ``K`` axial-orientation logits.

    The network does not directly predict an angle or confidence. Those are
    deterministic quantities decoded from the orientation distribution.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_bins: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2.")
        self.num_bins = num_bins
        blocks: list[nn.Module] = [
            ConvNormAct(in_channels, hidden_channels),
            ConvNormAct(hidden_channels, hidden_channels),
        ]
        if dropout:
            blocks.append(nn.Dropout2d(dropout))
        self.features = nn.Sequential(*blocks)
        self.output = nn.Conv2d(hidden_channels, 1 + num_bins, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)
        # A sparse initial map is safer than starting with P=0.5 everywhere.
        self.output.bias.data[0] = math.log(0.01 / 0.99)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(feature))


def split_structure_logits(logits: torch.Tensor, num_bins: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a ``[B, 1+K, H, W]`` tensor into presence and direction logits."""

    if logits.ndim != 4:
        raise ValueError(f"Expected BCHW logits, got shape {tuple(logits.shape)}")
    inferred_bins = logits.shape[1] - 1
    if inferred_bins < 2:
        raise ValueError("Structure logits need at least one presence and two direction channels.")
    if num_bins is not None and inferred_bins != num_bins:
        raise ValueError(f"Expected {num_bins} direction bins, got {inferred_bins}.")
    return logits[:, :1], logits[:, 1:]

