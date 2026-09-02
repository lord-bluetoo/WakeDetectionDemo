import torch

from wake_structure.config import GeometryLossConfig
from wake_structure.losses import GeometryCriterion
from wake_structure.targets import build_geometry_targets


def _batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.zeros(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.6, 0.2, 0.0]]),
        "keypoints": torch.tensor([[[0.25, 0.5, 1.0], [0.35, 0.45, 1.0], [0.35, 0.55, 1.0]]]),
    }


def test_targets_include_tip_offset_and_two_arms() -> None:
    targets = build_geometry_targets(_batch(), (8, 8), 16, 8.0, 0.05, 2)
    assert targets.tip_heatmap.max() == 1
    assert targets.tip_offsets.shape == (1, 2)
    assert targets.arm1_distribution.shape == (1, 16, 8, 8)
    assert targets.direction_mask.sum() > 0


def test_geometry_loss_is_finite_and_backpropagates() -> None:
    logits = torch.randn(1, 36, 8, 8, requires_grad=True)
    items = GeometryCriterion(16, GeometryLossConfig())(logits, _batch())
    total = sum(items.values())
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

