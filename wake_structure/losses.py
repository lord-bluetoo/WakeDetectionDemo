"""Weakly supervised objectives for the Structure Head."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import StructureLossConfig
from .head import split_structure_logits
from .targets import build_weak_structure_targets


def _differentiable_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


class StructureCriterion(nn.Module):
    """MIL + background + soft orientation + sparsity + rotation consistency."""

    def __init__(self, num_bins: int, config: StructureLossConfig) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.config = config

    def forward(
        self,
        logits: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        rotated_logits: torch.Tensor | None = None,
        quarter_turns: int = 0,
    ) -> dict[str, torch.Tensor]:
        presence_logits, orientation_logits = split_structure_logits(logits, self.num_bins)
        targets = build_weak_structure_targets(
            batch,
            feature_size=logits.shape[-2:],
            num_bins=self.num_bins,
            orientation_kappa=self.config.orientation_kappa,
            roi_margin=self.config.roi_margin,
        )
        presence = presence_logits.sigmoid()

        mil = _differentiable_zero(logits)
        for mask, image_index in zip(targets.instance_masks, targets.instance_batch_indices):
            values = presence[int(image_index.item()), 0][mask]
            count = max(1, math.ceil(values.numel() * self.config.mil_topk_fraction))
            pooled = values.topk(count).values.mean()
            mil = mil - torch.log(pooled.clamp_min(1e-6))
        if len(targets.instance_masks):
            mil = mil / len(targets.instance_masks)

        outside = 1.0 - targets.roi_mask
        outside_count = outside.sum()
        background = (
            (F.softplus(presence_logits) * outside).sum() / outside_count.clamp_min(1)
            if outside_count > 0
            else _differentiable_zero(logits)
        )
        presence_loss = self.config.mil_weight * mil + self.config.background_weight * background

        log_q = orientation_logits.log_softmax(dim=1)
        direction_ce = -(targets.orientation_distribution * log_q).sum(dim=1, keepdim=True)
        direction_weight = targets.orientation_mask * (
            self.config.orientation_presence_floor
            + (1.0 - self.config.orientation_presence_floor) * presence.detach()
        )
        orientation_loss = (direction_ce * direction_weight).sum() / direction_weight.sum().clamp_min(1)
        orientation_loss = self.config.orientation_weight * orientation_loss

        sparse = _differentiable_zero(logits)
        valid_images = 0
        for image_index in range(logits.shape[0]):
            inside = targets.roi_mask[image_index].bool()
            if inside.any():
                fraction = presence[image_index][inside].mean()
                sparse = sparse + F.relu(fraction - self.config.max_foreground_fraction).square()
                valid_images += 1
        if valid_images:
            sparse = sparse / valid_images
        sparse_loss = self.config.sparse_weight * sparse

        equivariance_loss = _differentiable_zero(logits)
        if rotated_logits is not None and self.config.equivariance_weight:
            equivariance_loss = self._equivariance_loss(logits, rotated_logits, quarter_turns)
            equivariance_loss = self.config.equivariance_weight * equivariance_loss

        return {
            "structure_presence_loss": presence_loss,
            "structure_orientation_loss": orientation_loss,
            "structure_sparse_loss": sparse_loss,
            "structure_equivariance_loss": equivariance_loss,
        }

    def _equivariance_loss(
        self,
        logits: torch.Tensor,
        rotated_logits: torch.Tensor,
        quarter_turns: int,
    ) -> torch.Tensor:
        turns = quarter_turns % 4
        presence_logits, orientation_logits = split_structure_logits(logits, self.num_bins)
        rotated_presence_logits, rotated_orientation_logits = split_structure_logits(rotated_logits, self.num_bins)

        aligned_presence = torch.rot90(rotated_presence_logits.sigmoid(), -turns, dims=(-2, -1))
        aligned_q = torch.rot90(rotated_orientation_logits.softmax(dim=1), -turns, dims=(-2, -1))
        # A 90-degree image turn moves an axial direction by K/2 bins.
        aligned_q = torch.roll(aligned_q, shifts=-(turns * self.num_bins // 2), dims=1)

        presence = presence_logits.sigmoid()
        q = orientation_logits.softmax(dim=1)
        presence_consistency = F.smooth_l1_loss(aligned_presence, presence)
        direction_weight = ((aligned_presence + presence) / 2).detach()
        direction_consistency = ((aligned_q - q).square().sum(dim=1, keepdim=True) * direction_weight).sum()
        direction_consistency = direction_consistency / direction_weight.sum().clamp_min(1)
        return presence_consistency + direction_consistency

