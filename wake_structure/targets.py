"""Rasterize coarse OBB labels into weak structure supervision."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .geometry import orientation_bin_centers


@dataclass
class WeakStructureTargets:
    roi_mask: torch.Tensor
    orientation_distribution: torch.Tensor
    orientation_mask: torch.Tensor
    instance_masks: torch.Tensor
    instance_batch_indices: torch.Tensor


def _long_axis_box(box: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cx, cy, width, height, angle = box
    swap = height > width
    long_side = torch.where(swap, height, width)
    short_side = torch.where(swap, width, height)
    long_angle = torch.remainder(angle + swap.to(angle.dtype) * (math.pi / 2), math.pi)
    return cx, cy, long_side, short_side, long_angle


def build_weak_structure_targets(
    batch: dict[str, torch.Tensor],
    feature_size: tuple[int, int],
    num_bins: int = 8,
    orientation_kappa: float = 4.0,
    roi_margin: float = 0.05,
) -> WeakStructureTargets:
    """Build MIL bags and soft direction targets from normalized ``xywhr`` OBBs.

    Pixels inside an OBB are *not* labeled entirely positive. Each OBB is only a
    positive MIL bag; pixels outside every OBB are background candidates.
    """

    images = batch["img"]
    boxes = batch["bboxes"].reshape(-1, 5).to(device=images.device, dtype=images.dtype)
    batch_indices = batch["batch_idx"].reshape(-1).to(device=images.device, dtype=torch.long)
    batch_size = images.shape[0]
    height, width = feature_size
    dtype, device = images.dtype, images.device

    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    roi = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    orientation_sum = torch.zeros(batch_size, num_bins, height, width, device=device, dtype=dtype)
    orientation_count = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    centers = orientation_bin_centers(num_bins, device=device, dtype=dtype)
    masks: list[torch.Tensor] = []
    valid_batch_indices: list[torch.Tensor] = []

    for box, image_index in zip(boxes, batch_indices):
        cx, cy, long_side, short_side, angle = _long_axis_box(box)
        if long_side <= 0 or short_side <= 0 or not (0 <= image_index < batch_size):
            continue
        dx, dy = grid_x - cx, grid_y - cy
        cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
        along = cos_angle * dx + sin_angle * dy
        across = -sin_angle * dx + cos_angle * dy
        scale = 1.0 + roi_margin
        mask = (along.abs() <= long_side * scale / 2) & (across.abs() <= short_side * scale / 2)
        if not mask.any():
            # Very small boxes still need at least one MIL instance.
            nearest = (dx.square() + dy.square()).reshape(-1).argmin()
            mask = torch.zeros_like(grid_x, dtype=torch.bool)
            mask.reshape(-1)[nearest] = True

        index = int(image_index.item())
        roi[index, 0].masked_fill_(mask, 1.0)
        direction = torch.softmax(orientation_kappa * torch.cos(2 * (centers - angle)), dim=0)
        orientation_sum[index, :, mask] += direction[:, None]
        orientation_count[index, 0, mask] += 1
        masks.append(mask)
        valid_batch_indices.append(image_index)

    orientation_mask = (orientation_count > 0).to(dtype)
    orientation = orientation_sum / orientation_count.clamp_min(1)
    if masks:
        instance_masks = torch.stack(masks)
        instance_batch_indices = torch.stack(valid_batch_indices).long()
    else:
        instance_masks = torch.zeros(0, height, width, device=device, dtype=torch.bool)
        instance_batch_indices = torch.zeros(0, device=device, dtype=torch.long)

    return WeakStructureTargets(
        roi_mask=roi,
        orientation_distribution=orientation,
        orientation_mask=orientation_mask,
        instance_masks=instance_masks,
        instance_batch_indices=instance_batch_indices,
    )

