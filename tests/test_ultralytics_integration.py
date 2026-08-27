import pytest
import torch

pytest.importorskip("ultralytics")

from wake_structure.config import StructureConfig, StructureLossConfig
from wake_structure.model import StructureOBBModel
from ultralytics.cfg import get_cfg


def test_yolov8_obb_structure_model_minimal_forward_and_loss() -> None:
    config = StructureConfig(
        enable_equivariance=False,
        loss=StructureLossConfig(equivariance_weight=0.0),
    )
    model = StructureOBBModel("yolov8n-obb.yaml", nc=1, verbose=False, structure_config=config)
    # The real trainer assigns its resolved runtime arguments before the first loss call.
    model.args = get_cfg()
    model.train()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.12, 0.2]]),
    }
    loss_vector, items = model(batch)
    assert loss_vector.ndim == 1
    assert "structure_presence_loss" in items
    assert torch.isfinite(loss_vector).all()
    loss_vector.sum().backward()
    assert model.structure_head.output.weight.grad is not None

    maps = model.structure_maps(batch["img"])
    assert maps["presence"].shape[1] == 1
    assert maps["orientation_distribution"].shape[1] == 8
