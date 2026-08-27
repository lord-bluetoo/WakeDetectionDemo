import torch

from wake_structure.config import StructureLossConfig
from wake_structure.losses import StructureCriterion
from wake_structure.targets import build_weak_structure_targets


def _batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.zeros(2, 3, 64, 64),
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor(
            [
                [0.5, 0.5, 0.6, 0.12, 0.0],
                [0.5, 0.5, 0.15, 0.5, 0.0],
            ]
        ),
    }


def test_targets_are_mil_bags_not_dense_positive_masks() -> None:
    targets = build_weak_structure_targets(_batch(), (8, 8), num_bins=8)
    assert targets.roi_mask.shape == (2, 1, 8, 8)
    assert targets.instance_masks.shape[0] == 2
    assert targets.orientation_distribution.shape == (2, 8, 8, 8)
    inside_sum = (targets.orientation_distribution.sum(1, keepdim=True) * targets.orientation_mask).sum()
    assert torch.allclose(inside_sum, targets.orientation_mask.sum(), atol=1e-5)


def test_structure_loss_is_finite_and_backpropagates() -> None:
    logits = torch.randn(2, 9, 8, 8, requires_grad=True)
    rotated_logits = torch.rot90(logits.detach(), 1, dims=(-2, -1)).clone().requires_grad_(True)
    criterion = StructureCriterion(8, StructureLossConfig())
    items = criterion(logits, _batch(), rotated_logits=rotated_logits, quarter_turns=1)
    total = sum(items.values())
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert rotated_logits.grad is not None and torch.isfinite(rotated_logits.grad).all()


def test_empty_batch_still_penalizes_background_without_nan() -> None:
    batch = {
        "img": torch.zeros(1, 3, 64, 64),
        "batch_idx": torch.zeros(0),
        "cls": torch.zeros(0, 1),
        "bboxes": torch.zeros(0, 5),
    }
    logits = torch.randn(1, 9, 8, 8, requires_grad=True)
    items = StructureCriterion(8, StructureLossConfig())(logits, batch)
    total = sum(items.values())
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None

