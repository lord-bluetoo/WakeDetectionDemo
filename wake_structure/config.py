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
    continuity_weight: float = 0.10
    tip_weight: float = 1.0
    offset_weight: float = 0.25
    arm_weight: float = 0.5
    mil_topk_fraction: float = 0.10
    direction_kappa: float = 8.0
    roi_margin: float = 0.05
    structure_band_width: float = 2.0
    structure_ignore_width: float = 4.0
    structure_segments: int = 4
    tip_radius: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.mil_topk_fraction <= 1:
            raise ValueError("mil_topk_fraction must be in (0, 1].")
        if self.structure_band_width <= 0:
            raise ValueError("structure_band_width must be positive.")
        if self.structure_ignore_width < self.structure_band_width:
            raise ValueError("structure_ignore_width must be at least structure_band_width.")
        if self.structure_segments < 1:
            raise ValueError("structure_segments must be positive.")
        if self.tip_radius < 0:
            raise ValueError("tip_radius must be non-negative.")


@dataclass(frozen=True)
class GeometryConfig:
    num_bins: int = 16
    p3_layer_index: int = 4
    hidden_channels: int = 64
    dropout: float = 0.0
    enable_refinement: bool = True
    enable_denoising: bool = True
    enable_directional_extraction: bool = True
    refinement_hidden_channels: int = 64
    sampling_steps: tuple[float, ...] = (1.0, 2.0, 4.0)
    confidence_floor: float = 0.2
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
        if not self.sampling_steps or any(step <= 0 for step in self.sampling_steps):
            raise ValueError("sampling_steps must contain positive values.")
        if not 0 <= self.confidence_floor <= 1:
            raise ValueError("confidence_floor must be in [0, 1].")

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "GeometryConfig":
        values = dict(values or {})
        loss_values = values.pop("loss", {})
        if "sampling_steps" in values:
            values["sampling_steps"] = tuple(float(step) for step in values["sampling_steps"])
        loss = GeometryLossConfig(**_known_kwargs(GeometryLossConfig, dict(loss_values)))
        return cls(loss=loss, **_known_kwargs(cls, values))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeometryConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return cls.from_dict(values)
