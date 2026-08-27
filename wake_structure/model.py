"""Ultralytics YOLOv8-OBB integration for the auxiliary Structure Head."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK

from .config import StructureConfig
from .geometry import decode_structure
from .head import StructureHead
from .losses import StructureCriterion


def _module_out_channels(module: torch.nn.Module) -> int:
    """Infer the output width of common Ultralytics blocks without a dummy pass."""

    for candidate in (getattr(module, "cv2", None), getattr(module, "conv", None), module):
        conv = getattr(candidate, "conv", candidate)
        if isinstance(conv, torch.nn.Conv2d):
            return conv.out_channels
    convolutions = [item for item in module.modules() if isinstance(item, torch.nn.Conv2d)]
    if convolutions:
        return convolutions[-1].out_channels
    raise TypeError(f"Cannot infer output channels for layer {module!r}")


class StructureOBBModel(OBBModel):
    """YOLO OBB model with an auxiliary P3 Structure Head.

    Detection predictions are unchanged. During training only, the extra loss
    updates the shared P3/backbone representation and Structure Head parameters.
    """

    def __init__(
        self,
        cfg: str | dict = "yolov8n-obb.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        structure_config: StructureConfig | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.structure_config = structure_config or StructureConfig()
        layer_index = self.structure_config.p3_layer_index
        if layer_index >= len(self.model):
            raise IndexError(f"P3 layer {layer_index} is outside a {len(self.model)}-layer model.")
        in_channels = _module_out_channels(self.model[layer_index])
        self.structure_head = StructureHead(
            in_channels=in_channels,
            hidden_channels=self.structure_config.hidden_channels,
            num_bins=self.structure_config.num_bins,
            dropout=self.structure_config.dropout,
        )
        self.structure_criterion = StructureCriterion(
            num_bins=self.structure_config.num_bins,
            config=self.structure_config.loss,
        )

    def _extract_p3(self, images: torch.Tensor) -> torch.Tensor:
        """Run the graph only up to the configured P3 tap."""

        outputs: list[Any] = []
        value: Any = images
        target_index = self.structure_config.p3_layer_index
        for module in self.model:
            if module.f != -1:
                value = (
                    outputs[module.f]
                    if isinstance(module.f, int)
                    else [value if index == -1 else outputs[index] for index in module.f]
                )
            value = module(value)
            outputs.append(value if module.i in self.save else None)
            if module.i == target_index:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Configured P3 layer {target_index} did not return a tensor.")
                return value
        raise RuntimeError(f"P3 layer {target_index} was not reached.")

    def _predict_and_capture_p3(self, images: torch.Tensor) -> tuple[Any, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}

        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError("The configured P3 layer must produce a tensor.")
            captured["feature"] = output

        handle = self.model[self.structure_config.p3_layer_index].register_forward_hook(capture)
        try:
            predictions = self.forward(images)
        finally:
            handle.remove()
        if "feature" not in captured:
            raise RuntimeError("Failed to capture the configured P3 feature.")
        return predictions, captured["feature"]

    def loss(self, batch: dict[str, torch.Tensor], preds: Any = None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds, p3 = self._predict_and_capture_p3(batch["img"])
        else:
            # Ultralytics compile=True calculates predictions before calling loss.
            p3 = self._extract_p3(batch["img"])

        structure_logits = self.structure_head(p3)
        rotated_logits = None
        quarter_turns = 0
        if self.training and self.structure_config.enable_equivariance:
            height, width = batch["img"].shape[-2:]
            if height == width:
                quarter_turns = int(torch.randint(1, 4, (), device=batch["img"].device).item())
            else:
                quarter_turns = 2
            rotated_images = torch.rot90(batch["img"], quarter_turns, dims=(-2, -1))
            rotated_logits = self.structure_head(self._extract_p3(rotated_images))

        detection_loss, loss_items = self.criterion(preds, batch)
        structure_items = self.structure_criterion(
            structure_logits,
            batch,
            rotated_logits=rotated_logits,
            quarter_turns=quarter_turns,
        )
        batch_size = batch["img"].shape[0]
        auxiliary_vector = torch.stack(tuple(structure_items.values())) * batch_size
        total_vector = torch.cat((detection_loss.reshape(-1), auxiliary_vector))
        logged_items = dict(loss_items)
        logged_items.update({name: value.detach() for name, value in structure_items.items()})
        return total_vector, logged_items

    def structure_maps(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return P, q_theta, theta, C, and P*C without running the OBB head."""

        return decode_structure(self.structure_head(self._extract_p3(images)))


class StructureOBBTrainer(OBBTrainer):
    """Single-process Ultralytics trainer that constructs :class:`StructureOBBModel`."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict | None = None,
        _callbacks: dict | None = None,
        structure_config: StructureConfig | None = None,
    ) -> None:
        self.structure_config = structure_config or StructureConfig()
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> StructureOBBModel:
        model = self.set_model_names_for_load(
            StructureOBBModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                structure_config=self.structure_config,
            )
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        return yolo.obb.OBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

