"""Build dense wake-geometry targets from OBBs and transformed landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .geometry import direction_bin_centers


@dataclass
class GeometryTargets:
    roi_mask: torch.Tensor
    structure_search_mask: torch.Tensor
    structure_background_mask: torch.Tensor
    segment_masks: torch.Tensor
    segment_batch_indices: torch.Tensor
    segment_group_indices: torch.Tensor
    tip_heatmap: torch.Tensor
    tip_indices: torch.Tensor
    tip_batch_indices: torch.Tensor
    tip_offsets: torch.Tensor
    arm1_distribution: torch.Tensor
    arm2_distribution: torch.Tensor
    direction_mask: torch.Tensor


def _long_axis_box(box: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cx, cy, width, height, angle = box
    swap = height > width
    long_side = torch.where(swap, height, width)
    short_side = torch.where(swap, width, height)
    long_angle = torch.remainder(angle + swap.to(angle.dtype) * (math.pi / 2), math.pi)
    return cx, cy, long_side, short_side, long_angle


def _soft_direction(angle: torch.Tensor, centers: torch.Tensor, kappa: float) -> torch.Tensor:
    return torch.softmax(kappa * torch.cos(centers - angle), dim=0)


def _draw_tip(heatmap: torch.Tensor, x: int, y: int, radius: int) -> None:
    height, width = heatmap.shape
    left, right = max(0, x - radius), min(width, x + radius + 1)
    top, bottom = max(0, y - radius), min(height, y + radius + 1)
    ys = torch.arange(top, bottom, device=heatmap.device, dtype=heatmap.dtype)
    xs = torch.arange(left, right, device=heatmap.device, dtype=heatmap.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    sigma = max(radius / 2, 0.5)
    gaussian = torch.exp(-((grid_x - x).square() + (grid_y - y).square()) / (2 * sigma * sigma))
    heatmap[top:bottom, left:right] = torch.maximum(heatmap[top:bottom, left:right], gaussian)


def build_geometry_targets(
    batch: dict[str, torch.Tensor],
    feature_size: tuple[int, int],
    num_bins: int,
    direction_kappa: float,
    roi_margin: float,
    tip_radius: int,
    structure_band_width: float = 2.0,
    structure_ignore_width: float = 4.0,
    structure_segments: int = 4,
) -> GeometryTargets:
    images = batch["img"]
    boxes = batch["bboxes"].reshape(-1, 5).to(device=images.device, dtype=images.dtype)
    keypoints = batch["keypoints"].to(device=images.device, dtype=images.dtype)
    batch_indices = batch["batch_idx"].reshape(-1).to(device=images.device, dtype=torch.long)
    batch_size = images.shape[0]
    height, width = feature_size
    dtype, device = images.dtype, images.device

    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    centers = direction_bin_centers(num_bins, device=device, dtype=dtype)

    roi = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    structure_search = torch.zeros_like(roi)
    structure_ignore = torch.zeros_like(roi)
    tip_heatmap = torch.zeros_like(roi)
    arm1_sum = torch.zeros(batch_size, num_bins, height, width, device=device, dtype=dtype)
    arm2_sum = torch.zeros_like(arm1_sum)
    direction_count = torch.zeros_like(roi)
    segment_masks: list[torch.Tensor] = []
    segment_batch_indices: list[torch.Tensor] = []
    segment_group_indices: list[int] = []
    tip_indices: list[torch.Tensor] = []
    tip_batch_indices: list[torch.Tensor] = []
    tip_offsets: list[torch.Tensor] = []

    next_group = 0
    feature_grid_x = grid_x * width
    feature_grid_y = grid_y * height

    for box, points, image_index in zip(boxes, keypoints, batch_indices):
        index = int(image_index.item())
        if not 0 <= index < batch_size:
            continue
        cx, cy, long_side, short_side, angle = _long_axis_box(box)
        dx, dy = grid_x - cx, grid_y - cy
        along = torch.cos(angle) * dx + torch.sin(angle) * dy
        across = -torch.sin(angle) * dx + torch.cos(angle) * dy
        scale = 1.0 + roi_margin
        mask = (along.abs() <= long_side * scale / 2) & (across.abs() <= short_side * scale / 2)
        if not mask.any():
            continue
        roi[index, 0].masked_fill_(mask, 1.0)

        tip = points[0]
        if tip.shape[0] > 2 and tip[2] <= 0:
            continue
        feature_tip = torch.stack((tip[0] * width, tip[1] * height))
        cell_x = int(torch.floor(feature_tip[0]).clamp(0, width - 1).item())
        cell_y = int(torch.floor(feature_tip[1]).clamp(0, height - 1).item())
        _draw_tip(tip_heatmap[index, 0], cell_x, cell_y, tip_radius)
        tip_indices.append(torch.tensor([cell_y, cell_x], device=device, dtype=torch.long))
        tip_batch_indices.append(image_index)
        tip_offsets.append(feature_tip - torch.tensor([cell_x, cell_y], device=device, dtype=dtype))

        vector1 = torch.stack(((points[1, 0] - tip[0]) * width, (points[1, 1] - tip[1]) * height))
        vector2 = torch.stack(((points[2, 0] - tip[0]) * width, (points[2, 1] - tip[1]) * height))
        if vector1.norm() <= 1e-6 or vector2.norm() <= 1e-6:
            continue
        arm_search_masks: list[torch.Tensor] = []
        for vector in (vector1, vector2):
            norm = vector.norm()
            unit = vector / norm
            rel_x = feature_grid_x - feature_tip[0]
            rel_y = feature_grid_y - feature_tip[1]
            arm_along = rel_x * unit[0] + rel_y * unit[1]
            arm_across = (-rel_x * unit[1] + rel_y * unit[0]).abs()
            forward = mask & (arm_along >= 0)
            if not forward.any():
                continue
            max_length = arm_along[forward].max().clamp_min(1e-6)
            search_mask = forward & (arm_across <= structure_band_width)
            ignore_mask = forward & (arm_across <= structure_ignore_width)
            structure_search[index, 0].masked_fill_(search_mask, 1.0)
            structure_ignore[index, 0].masked_fill_(ignore_mask, 1.0)
            arm_search_masks.append(search_mask)

            for segment_index in range(structure_segments):
                lower = max_length * segment_index / structure_segments
                upper = max_length * (segment_index + 1) / structure_segments
                segment = search_mask & (arm_along >= lower) & (arm_along <= upper)
                if segment.any():
                    segment_masks.append(segment)
                    segment_batch_indices.append(image_index)
                    segment_group_indices.append(next_group)
            next_group += 1

        theta1 = torch.remainder(torch.atan2(vector1[1], vector1[0]), 2 * math.pi)
        theta2 = torch.remainder(torch.atan2(vector2[1], vector2[0]), 2 * math.pi)
        if not arm_search_masks:
            continue
        direction_region = torch.stack(arm_search_masks).any(dim=0)
        arm1_sum[index, :, direction_region] += _soft_direction(theta1, centers, direction_kappa)[:, None]
        arm2_sum[index, :, direction_region] += _soft_direction(theta2, centers, direction_kappa)[:, None]
        direction_count[index, 0, direction_region] += 1

    if segment_masks:
        segment_mask_tensor = torch.stack(segment_masks)
        segment_batch_tensor = torch.stack(segment_batch_indices).long()
        segment_group_tensor = torch.tensor(segment_group_indices, device=device, dtype=torch.long)
    else:
        segment_mask_tensor = torch.zeros(0, height, width, device=device, dtype=torch.bool)
        segment_batch_tensor = torch.zeros(0, device=device, dtype=torch.long)
        segment_group_tensor = torch.zeros(0, device=device, dtype=torch.long)
    if tip_indices:
        tip_index_tensor = torch.stack(tip_indices)
        tip_batch_tensor = torch.stack(tip_batch_indices).long()
        tip_offset_tensor = torch.stack(tip_offsets)
    else:
        tip_index_tensor = torch.zeros(0, 2, device=device, dtype=torch.long)
        tip_batch_tensor = torch.zeros(0, device=device, dtype=torch.long)
        tip_offset_tensor = torch.zeros(0, 2, device=device, dtype=dtype)

    return GeometryTargets(
        roi_mask=roi,
        structure_search_mask=structure_search,
        structure_background_mask=1.0 - structure_ignore,
        segment_masks=segment_mask_tensor,
        segment_batch_indices=segment_batch_tensor,
        segment_group_indices=segment_group_tensor,
        tip_heatmap=tip_heatmap,
        tip_indices=tip_index_tensor,
        tip_batch_indices=tip_batch_tensor,
        tip_offsets=tip_offset_tensor,
        arm1_distribution=arm1_sum / direction_count.clamp_min(1),
        arm2_distribution=arm2_sum / direction_count.clamp_min(1),
        direction_mask=(direction_count > 0).to(dtype),
    )
