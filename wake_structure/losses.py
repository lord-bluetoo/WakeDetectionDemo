"""Losses for landmark-supervised wake geometry."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import GeometryLossConfig
from .head import split_geometry_logits
from .targets import build_geometry_targets


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _heatmap_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    positive = target.eq(1)
    negative = ~positive
    positive_loss = -(1 - probability).square() * probability.log() * positive
    negative_loss = -(probability.square()) * (1 - probability).log() * (1 - target).pow(4) * negative
    count = positive.sum().clamp_min(1)
    return (positive_loss.sum() + negative_loss.sum()) / count


class GeometryCriterion(nn.Module):
    def __init__(self, num_bins: int, config: GeometryLossConfig) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.config = config

    def forward(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        parts = split_geometry_logits(logits, self.num_bins)
        targets = build_geometry_targets(
            batch,
            feature_size=logits.shape[-2:],
            num_bins=self.num_bins,
            direction_kappa=self.config.direction_kappa,
            roi_margin=self.config.roi_margin,
            tip_radius=self.config.tip_radius,
            structure_band_width=self.config.structure_band_width,
            structure_ignore_width=self.config.structure_ignore_width,
            structure_segments=self.config.structure_segments,
        )

        presence = parts.structure.sigmoid()
        segment_scores: list[torch.Tensor] = []
        for mask, image_index in zip(targets.segment_masks, targets.segment_batch_indices):
            values = presence[int(image_index.item()), 0][mask]
            count = max(1, math.ceil(values.numel() * self.config.mil_topk_fraction))
            segment_scores.append(values.topk(count).values.mean())
        if segment_scores:
            scores = torch.stack(segment_scores)
            mil = -torch.log(scores.clamp_min(1e-6)).mean()
            continuity_terms = []
            for group_index in targets.segment_group_indices.unique():
                group_scores = scores[targets.segment_group_indices == group_index]
                if len(group_scores) > 1:
                    continuity_terms.append((group_scores[1:] - group_scores[:-1]).square().mean())
            continuity = torch.stack(continuity_terms).mean() if continuity_terms else _zero(logits)
        else:
            mil = _zero(logits)
            continuity = _zero(logits)

        background_mask = targets.structure_background_mask
        background = (F.softplus(parts.structure) * background_mask).sum() / background_mask.sum().clamp_min(1)
        structure_loss = (
            self.config.mil_weight * mil
            + self.config.background_weight * background
            + self.config.continuity_weight * continuity
        )

        tip_loss = self.config.tip_weight * _heatmap_focal_loss(parts.tip, targets.tip_heatmap)

        offset_loss = _zero(logits)
        if len(targets.tip_indices):
            y, x = targets.tip_indices.unbind(dim=1)
            predicted_offsets = parts.offset.sigmoid()[targets.tip_batch_indices, :, y, x]
            offset_loss = F.smooth_l1_loss(predicted_offsets, targets.tip_offsets)
        offset_loss = self.config.offset_weight * offset_loss

        direction_weight = targets.direction_mask
        arm1_ce = -(targets.arm1_distribution * parts.arm1.log_softmax(dim=1)).sum(dim=1, keepdim=True)
        arm2_ce = -(targets.arm2_distribution * parts.arm2.log_softmax(dim=1)).sum(dim=1, keepdim=True)
        arm_loss = ((arm1_ce + arm2_ce) * direction_weight).sum()
        arm_loss = arm_loss / (2 * direction_weight.sum().clamp_min(1))
        arm_loss = self.config.arm_weight * arm_loss

        return {
            "geometry_structure_loss": structure_loss,
            "geometry_tip_loss": tip_loss,
            "geometry_offset_loss": offset_loss,
            "geometry_arm_loss": arm_loss,
        }
