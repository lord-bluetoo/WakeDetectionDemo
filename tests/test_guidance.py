import torch

from wake_structure.guidance import GeometryGuidedRefinement


def test_refinement_is_identity_at_zero_scales() -> None:
    module = GeometryGuidedRefinement(16, 8, 16)
    module.eval()
    feature = torch.randn(2, 16, 10, 10)
    logits = torch.randn(2, 36, 10, 10)
    assert torch.equal(module(feature, logits), feature)


def test_refinement_backpropagates() -> None:
    module = GeometryGuidedRefinement(
        16, 8, 16, sampling_steps=(1.0, 2.0, 4.0), denoise_scale_init=0.1, feature_scale_init=0.1
    )
    feature = torch.randn(2, 16, 8, 8, requires_grad=True)
    logits = torch.randn(2, 36, 8, 8, requires_grad=True)
    module(feature, logits).square().mean().backward()
    assert feature.grad is not None
    assert logits.grad is not None
    assert module.denoise_scale.grad is not None
    assert module.feature_scale.grad is not None


def test_denoising_uses_presence_while_enhancement_uses_direction_confidence() -> None:
    module = GeometryGuidedRefinement(16, 8, 16, confidence_floor=0.2)
    logits = torch.zeros(1, 36, 4, 4)
    logits[:, 0] = 2.0
    gates = module.decode_gates(logits)
    presence = logits[:, 0:1].sigmoid()
    assert torch.allclose(gates["denoise_gate"], 1 - presence)
    assert torch.allclose(gates["enhancement_gate"], presence * 0.2, atol=1e-6)


def test_refinement_paths_can_be_disabled_independently() -> None:
    module = GeometryGuidedRefinement(
        16,
        8,
        16,
        enable_denoising=False,
        enable_directional_extraction=False,
        denoise_scale_init=1.0,
        feature_scale_init=1.0,
    )
    feature = torch.randn(1, 16, 6, 6)
    logits = torch.randn(1, 36, 6, 6)
    assert torch.equal(module(feature, logits), feature)
