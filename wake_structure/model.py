"""Ultralytics YOLOv8-OBB integration for the auxiliary Structure Head."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK

from .config import StructureConfig
from .geometry import decode_structure
from .guidance import StructureGuidedExtractor
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
    """YOLO OBB model with a P3 Structure Head and optional feature guidance.

    V1 only uses the structure objective as auxiliary supervision. V2 can also
    feed a zero-initialized, structure-conditioned residual into the layers that
    follow P3, while preserving the original feature as an identity path.
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
        self.feature_guidance = (
            StructureGuidedExtractor(
                in_channels=in_channels,
                hidden_channels=self.structure_config.guidance_hidden_channels,
                num_bins=self.structure_config.num_bins,
                sampling_step=self.structure_config.guidance_sampling_step,
                alpha_init=self.structure_config.guidance_alpha_init,
            )
            if self.structure_config.enable_feature_guidance
            else None
        )
        self.structure_criterion = StructureCriterion(
            num_bins=self.structure_config.num_bins,
            config=self.structure_config.loss,
        )
        self._capture_structure_logits = False
        self._captured_structure_logits: torch.Tensor | None = None

    @staticmethod
    def _predict_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
        """Accept both older and newer Ultralytics ``_predict_once`` call shapes."""

        visualize = kwargs.pop("visualize", False)
        embed = kwargs.pop("embed", None)
        if kwargs:
            raise TypeError(f"Unexpected prediction options: {sorted(kwargs)}")
        if len(args) == 1:
            value = args[0]
            if value is None or isinstance(value, (list, tuple, set, frozenset)):
                embed = value
            else:
                visualize = value
        elif len(args) == 2:
            visualize, embed = args
        elif len(args) > 2:
            raise TypeError("Too many positional options for _predict_once().")
        return visualize, embed

    def _predict_once(self, x: torch.Tensor, profile: bool = False, *args: Any, **kwargs: Any):
        """Run the Ultralytics graph and inject structure guidance after P3."""

        visualize, embed = self._predict_options(args, kwargs)
        outputs: list[Any] = []
        timings: list[float] = []
        embeddings: list[torch.Tensor] = []
        embed_indices = frozenset(embed) if embed else {-1}
        max_embed_index = max(embed_indices)
        target_index = getattr(getattr(self, "structure_config", None), "p3_layer_index", -1)

        for module in self.model:
            if module.f != -1:
                x = (
                    outputs[module.f]
                    if isinstance(module.f, int)
                    else [x if index == -1 else outputs[index] for index in module.f]
                )
            if profile:
                self._profile_one_layer(module, x, timings)
            x = module(x)

            if module.i == target_index and hasattr(self, "structure_head"):
                structure_logits = self.structure_head(x)
                if self._capture_structure_logits:
                    self._captured_structure_logits = structure_logits
                if self.feature_guidance is not None:
                    x = self.feature_guidance(x, structure_logits)

            outputs.append(x if module.i in self.save else None)
            if visualize:
                try:
                    from ultralytics.utils.plotting import feature_visualization

                    feature_visualization(x, module.type, module.i, save_dir=visualize)
                except ImportError:
                    pass
            if module.i in embed_indices:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if module.i == max_embed_index:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

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

    def loss(self, batch: dict[str, torch.Tensor], preds: Any = None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            self._captured_structure_logits = None
            self._capture_structure_logits = True
            try:
                preds = self.forward(batch["img"])
            finally:
                self._capture_structure_logits = False
            structure_logits = self._captured_structure_logits
            self._captured_structure_logits = None
            if structure_logits is None:
                raise RuntimeError("Failed to capture structure logits during detection forward.")
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
