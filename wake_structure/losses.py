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
        )

        presence = parts.structure.sigmoid()
        mil = _zero(logits)
        for mask, image_index in zip(targets.instance_masks, targets.instance_batch_indices):
            values = presence[int(image_index.item()), 0][mask]
            count = max(1, math.ceil(values.numel() * self.config.mil_topk_fraction))
            mil = mil - torch.log(values.topk(count).values.mean().clamp_min(1e-6))
        if len(targets.instance_masks):
            mil = mil / len(targets.instance_masks)

        outside = 1.0 - targets.roi_mask
        background = (F.softplus(parts.structure) * outside).sum() / outside.sum().clamp_min(1)
        sparse = _zero(logits)
        valid_images = 0
        for image_index in range(logits.shape[0]):
            inside = targets.roi_mask[image_index].bool()
            if inside.any():
                sparse = sparse + F.relu(
                    presence[image_index][inside].mean() - self.config.max_foreground_fraction
                ).square()
                valid_images += 1
        if valid_images:
            sparse = sparse / valid_images
        structure_loss = (
            self.config.mil_weight * mil
            + self.config.background_weight * background
            + self.config.sparse_weight * sparse
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

