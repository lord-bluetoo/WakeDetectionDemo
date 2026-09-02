import torch

from wake_structure.guidance import GeometryGuidedRefinement


def test_refinement_is_identity_at_zero_scales() -> None:
    module = GeometryGuidedRefinement(16, 8, 16)
    module.eval()
    feature = torch.randn(2, 16, 10, 10)
    logits = torch.randn(2, 36, 10, 10)
    assert torch.equal(module(feature, logits), feature)


def test_refinement_backpropagates() -> None:
    module = GeometryGuidedRefinement(16, 8, 16, denoise_scale_init=0.1, feature_scale_init=0.1)
    feature = torch.randn(2, 16, 8, 8, requires_grad=True)
    logits = torch.randn(2, 36, 8, 8, requires_grad=True)
    module(feature, logits).square().mean().backward()
    assert feature.grad is not None
    assert logits.grad is not None
    assert module.denoise_scale.grad is not None
    assert module.feature_scale.grad is not None
