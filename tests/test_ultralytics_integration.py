import pytest
import torch

pytest.importorskip("ultralytics")

from ultralytics.cfg import get_cfg

from wake_structure.config import GeometryConfig
from wake_structure.model import GeometryOBBModel


def test_yolov8_obb_geometry_model_minimal_forward_and_loss() -> None:
    model = GeometryOBBModel(
        "yolov8n-obb.yaml",
        nc=1,
        verbose=False,
        geometry_config=GeometryConfig(enable_refinement=True),
    )
    model.args = get_cfg()
    model.train()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.12, 0.2]]),
        "keypoints": torch.tensor([[[0.35, 0.5, 1.0], [0.45, 0.45, 1.0], [0.45, 0.55, 1.0]]]),
    }
    loss_vector, items = model(batch)
    assert loss_vector.ndim == 1
    assert "geometry_tip_loss" in items
    assert torch.isfinite(loss_vector).all()
    loss_vector.sum().backward()
    assert model.geometry_head.output.weight.grad is not None
    assert model.geometry_refinement is not None
    assert model.geometry_refinement.feature_scale.grad is not None

    maps = model.geometry_maps(batch["img"])
    assert maps["structure"].shape[1] == 1
    assert maps["arm1_distribution"].shape[1] == 16
