import math

import torch

from wake_structure.geometry import decode_direction


def test_direction_bins_cover_full_circle() -> None:
    logits = torch.full((1, 16, 1, 1), -20.0)
    logits[:, 12] = 20.0
    theta, confidence, probabilities = decode_direction(logits)
    assert torch.allclose(theta, torch.tensor([[[[3 * math.pi / 2]]]]), atol=1e-5)
    assert torch.allclose(confidence, torch.ones_like(confidence), atol=1e-5)
    assert torch.allclose(probabilities.sum(1), torch.ones(1, 1, 1))


def test_uniform_direction_has_zero_confidence() -> None:
    _, confidence, _ = decode_direction(torch.zeros(2, 16, 3, 4))
    assert confidence.max() < 1e-5

