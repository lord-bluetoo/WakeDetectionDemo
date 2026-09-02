"""YOLOv8-OBB integration for landmark-supervised wake geometry."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .config import GeometryConfig
from .dataset import GeometryYOLODataset
from .geometry import decode_geometry
from .guidance import GeometryGuidedRefinement
from .head import GeometryHead
from .losses import GeometryCriterion


def _module_out_channels(module: torch.nn.Module) -> int:
    for candidate in (getattr(module, "cv2", None), getattr(module, "conv", None), module):
        conv = getattr(candidate, "conv", candidate)
        if isinstance(conv, torch.nn.Conv2d):
            return conv.out_channels
    convolutions = [item for item in module.modules() if isinstance(item, torch.nn.Conv2d)]
    return convolutions[-1].out_channels


class GeometryOBBModel(OBBModel):
    """OBB detector with a P3 geometry head and geometry-guided refinement."""

    def __init__(
        self,
        cfg: str | dict = "yolov8n-obb.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        geometry_config: GeometryConfig | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.geometry_config = geometry_config or GeometryConfig()
        layer_index = self.geometry_config.p3_layer_index
        in_channels = _module_out_channels(self.model[layer_index])
        self.geometry_head = GeometryHead(
            in_channels,
            hidden_channels=self.geometry_config.hidden_channels,
            num_bins=self.geometry_config.num_bins,
            dropout=self.geometry_config.dropout,
        )
        self.geometry_refinement = (
            GeometryGuidedRefinement(
                in_channels,
                hidden_channels=self.geometry_config.refinement_hidden_channels,
                num_bins=self.geometry_config.num_bins,
                sampling_step=self.geometry_config.sampling_step,
                denoise_scale_init=self.geometry_config.denoise_scale_init,
                feature_scale_init=self.geometry_config.feature_scale_init,
            )
            if self.geometry_config.enable_refinement
            else None
        )
        self.geometry_criterion = GeometryCriterion(self.geometry_config.num_bins, self.geometry_config.loss)
        self._capture_geometry = False
        self._captured_geometry: torch.Tensor | None = None

    @staticmethod
    def _predict_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
        visualize = kwargs.pop("visualize", False)
        embed = kwargs.pop("embed", None)
        if len(args) == 1:
            value = args[0]
            if value is None or isinstance(value, (list, tuple, set, frozenset)):
                embed = value
            else:
                visualize = value
        elif len(args) == 2:
            visualize, embed = args
        return visualize, embed

    def _predict_once(self, x: torch.Tensor, profile: bool = False, *args: Any, **kwargs: Any):
        visualize, embed = self._predict_options(args, kwargs)
        outputs: list[Any] = []
        timings: list[float] = []
        embeddings: list[torch.Tensor] = []
        embed_indices = frozenset(embed) if embed else {-1}
        max_embed_index = max(embed_indices)
        target_index = getattr(getattr(self, "geometry_config", None), "p3_layer_index", -1)

        for module in self.model:
            if module.f != -1:
                x = outputs[module.f] if isinstance(module.f, int) else [x if i == -1 else outputs[i] for i in module.f]
            if profile:
                self._profile_one_layer(module, x, timings)
            x = module(x)
            if module.i == target_index and hasattr(self, "geometry_head"):
                geometry_logits = self.geometry_head(x)
                if self._capture_geometry:
                    self._captured_geometry = geometry_logits
                if self.geometry_refinement is not None:
                    x = self.geometry_refinement(x, geometry_logits)
            outputs.append(x if module.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization

                feature_visualization(x, module.type, module.i, save_dir=visualize)
            if module.i in embed_indices:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if module.i == max_embed_index:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _extract_p3(self, images: torch.Tensor) -> torch.Tensor:
        outputs: list[Any] = []
        value: Any = images
        for module in self.model:
            if module.f != -1:
                value = (
                    outputs[module.f]
                    if isinstance(module.f, int)
                    else [value if i == -1 else outputs[i] for i in module.f]
                )
            value = module(value)
            outputs.append(value if module.i in self.save else None)
            if module.i == self.geometry_config.p3_layer_index:
                return value
        raise RuntimeError("Configured P3 layer was not reached.")

    def loss(self, batch: dict[str, torch.Tensor], preds: Any = None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            self._captured_geometry = None
            self._capture_geometry = True
            try:
                preds = self.forward(batch["img"])
            finally:
                self._capture_geometry = False
            geometry_logits = self._captured_geometry
        else:
            geometry_logits = self.geometry_head(self._extract_p3(batch["img"]))
        if geometry_logits is None:
            raise RuntimeError("Geometry logits were not captured during the detection forward pass.")

        detection_loss, loss_items = self.criterion(preds, batch)
        geometry_items = self.geometry_criterion(geometry_logits, batch)
        auxiliary = torch.stack(tuple(geometry_items.values())) * batch["img"].shape[0]
        total_vector = torch.cat((detection_loss.reshape(-1), auxiliary))
        logged_items = dict(loss_items)
        logged_items.update({name: value.detach() for name, value in geometry_items.items()})
        return total_vector, logged_items

    def geometry_maps(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.geometry_head(self._extract_p3(images))
        return decode_geometry(logits, self.geometry_config.num_bins)


class GeometryOBBTrainer(OBBTrainer):
    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict | None = None,
        _callbacks: dict | None = None,
        geometry_config: GeometryConfig | None = None,
    ) -> None:
        self.geometry_config = geometry_config or GeometryConfig()
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return GeometryYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=copy(self.args),
            rect=mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> GeometryOBBModel:
        model = self.set_model_names_for_load(
            GeometryOBBModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
                geometry_config=self.geometry_config,
            )
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
