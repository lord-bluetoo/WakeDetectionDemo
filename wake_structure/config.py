"""Configuration objects for the first structure-head experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _known_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    known = {item.name for item in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} option(s): {sorted(unknown)}")
    return values


@dataclass(frozen=True)
class StructureLossConfig:
    """Weights and weak-target settings for the auxiliary objective."""

    mil_weight: float = 1.0
    background_weight: float = 0.25
    orientation_weight: float = 0.25
    sparse_weight: float = 0.05
    equivariance_weight: float = 0.10
    mil_topk_fraction: float = 0.10
    max_foreground_fraction: float = 0.25
    orientation_kappa: float = 4.0
    roi_margin: float = 0.05
    orientation_presence_floor: float = 0.10

    def __post_init__(self) -> None:
        if not 0 < self.mil_topk_fraction <= 1:
            raise ValueError("mil_topk_fraction must be in (0, 1].")
        if not 0 < self.max_foreground_fraction <= 1:
            raise ValueError("max_foreground_fraction must be in (0, 1].")
        if self.orientation_kappa < 0:
            raise ValueError("orientation_kappa must be non-negative.")
        weights = (
            self.mil_weight,
            self.background_weight,
            self.orientation_weight,
            self.sparse_weight,
            self.equivariance_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("Loss weights must be non-negative.")


@dataclass(frozen=True)
class StructureConfig:
    """Architecture, guidance, and loss configuration for the structure model."""

    num_bins: int = 8
    p3_layer_index: int = 4
    hidden_channels: int = 64
    dropout: float = 0.0
    enable_equivariance: bool = True
    enable_feature_guidance: bool = False
    guidance_hidden_channels: int = 64
    guidance_sampling_step: float = 1.0
    guidance_alpha_init: float = 0.0
    loss: StructureLossConfig = field(default_factory=StructureLossConfig)

    def __post_init__(self) -> None:
        if self.num_bins < 2 or self.num_bins % 2:
            raise ValueError("num_bins must be an even integer >= 2.")
        if self.p3_layer_index < 0:
            raise ValueError("p3_layer_index must be non-negative.")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be positive.")
        if self.guidance_hidden_channels < 1:
            raise ValueError("guidance_hidden_channels must be positive.")
        if self.guidance_sampling_step <= 0:
            raise ValueError("guidance_sampling_step must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "StructureConfig":
        values = dict(values or {})
        loss_values = values.pop("loss", {})
        loss = StructureLossConfig(**_known_kwargs(StructureLossConfig, dict(loss_values)))
        return cls(loss=loss, **_known_kwargs(cls, values))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StructureConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise TypeError("Structure config YAML must contain a mapping at the top level.")
        return cls.from_dict(values)
