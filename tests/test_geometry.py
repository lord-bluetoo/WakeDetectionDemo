import math

import torch

from wake_structure.geometry import axial_distance, decode_orientation, orientation_moments


def test_one_hot_bins_decode_to_centers_with_unit_confidence() -> None:
    probabilities = torch.zeros(1, 8, 1, 1)
    probabilities[:, 3] = 1
    theta, confidence, _ = orientation_moments(probabilities)
    assert torch.allclose(theta, torch.tensor([[[[3 * math.pi / 8]]]]), atol=1e-6)
    assert torch.allclose(confidence, torch.ones_like(confidence), atol=1e-6)


def test_uniform_distribution_has_zero_confidence() -> None:
    logits = torch.zeros(2, 8, 3, 4)
    _, confidence, probabilities = decode_orientation(logits)
    assert torch.allclose(probabilities.sum(1), torch.ones(2, 3, 4))
    assert confidence.max() < 1e-5


def test_axial_distance_wraps_at_180_degrees() -> None:
    first = torch.tensor(math.radians(1.0))
    second = torch.tensor(math.radians(179.0))
    assert math.isclose(math.degrees(axial_distance(first, second).item()), 2.0, abs_tol=1e-4)

