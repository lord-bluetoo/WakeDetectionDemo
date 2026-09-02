"""Configuration for landmark-supervised wake geometry learning."""

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
class GeometryLossConfig:
    mil_weight: float = 1.0
    background_weight: float = 0.25
    sparse_weight: float = 0.05
    tip_weight: float = 1.0
    offset_weight: float = 0.25
    arm_weight: float = 0.5
    mil_topk_fraction: float = 0.10
    max_foreground_fraction: float = 0.25
    direction_kappa: float = 8.0
    roi_margin: float = 0.05
    tip_radius: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.mil_topk_fraction <= 1:
            raise ValueError("mil_topk_fraction must be in (0, 1].")
        if not 0 < self.max_foreground_fraction <= 1:
            raise ValueError("max_foreground_fraction must be in (0, 1].")
        if self.tip_radius < 0:
            raise ValueError("tip_radius must be non-negative.")


@dataclass(frozen=True)
class GeometryConfig:
    num_bins: int = 16
    p3_layer_index: int = 4
    hidden_channels: int = 64
    dropout: float = 0.0
    enable_refinement: bool = True
    refinement_hidden_channels: int = 64
    sampling_step: float = 1.0
    denoise_scale_init: float = 0.0
    feature_scale_init: float = 0.0
    loss: GeometryLossConfig = field(default_factory=GeometryLossConfig)

    def __post_init__(self) -> None:
        if self.num_bins < 4:
            raise ValueError("num_bins must be at least 4.")
        if self.p3_layer_index < 0:
            raise ValueError("p3_layer_index must be non-negative.")
        if self.hidden_channels < 1 or self.refinement_hidden_channels < 1:
            raise ValueError("hidden channel counts must be positive.")
        if self.sampling_step <= 0:
            raise ValueError("sampling_step must be positive.")

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "GeometryConfig":
        values = dict(values or {})
        loss_values = values.pop("loss", {})
        loss = GeometryLossConfig(**_known_kwargs(GeometryLossConfig, dict(loss_values)))
        return cls(loss=loss, **_known_kwargs(cls, values))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeometryConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return cls.from_dict(values)
