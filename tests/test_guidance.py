import torch

from wake_structure.guidance import StructureGuidedExtractor


def test_guidance_is_exact_identity_at_zero_alpha() -> None:
    module = StructureGuidedExtractor(16, hidden_channels=8, num_bins=8, alpha_init=0.0)
    module.eval()
    feature = torch.randn(2, 16, 12, 10)
    logits = torch.randn(2, 9, 12, 10)
    output = module(feature, logits)
    assert torch.equal(output, feature)


def test_guidance_backpropagates_through_feature_and_structure_logits() -> None:
    module = StructureGuidedExtractor(16, hidden_channels=8, num_bins=8, alpha_init=0.1)
    feature = torch.randn(2, 16, 8, 8, requires_grad=True)
    logits = torch.randn(2, 9, 8, 8, requires_grad=True)
    output = module(feature, logits)
    output.square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert module.output.weight.grad is not None and torch.isfinite(module.output.weight.grad).all()


def test_guidance_rejects_mismatched_spatial_shapes() -> None:
    module = StructureGuidedExtractor(16, hidden_channels=8, num_bins=8)
    feature = torch.randn(1, 16, 8, 8)
    logits = torch.randn(1, 9, 7, 8)
    try:
        module(feature, logits)
    except ValueError as error:
        assert "same spatial size" in str(error)
    else:
        raise AssertionError("Expected mismatched feature and logit shapes to fail.")
