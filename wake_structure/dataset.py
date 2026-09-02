"""Landmark-aware SWIM dataset built on Ultralytics' OBB transforms."""

from __future__ import annotations

import math
from copy import copy
from pathlib import Path

import numpy as np

from ultralytics.data.dataset import YOLODataset


def landmark_path_for_image(image_path: str | Path) -> Path:
    path = Path(image_path)
    parts = list(path.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "landmarks"
    return Path(*parts).with_suffix(".txt")


def read_landmarks(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return np.zeros((0, 4), dtype=np.float32)
    values = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if values.shape[1] != 4:
        raise ValueError(f"Expected px py theta1 theta2 in {path}")
    return values


def landmarks_to_keypoints(landmarks: np.ndarray, shape: tuple[int, int], arm_length: float = 0.08) -> np.ndarray:
    """Represent each tip and its two arm directions as three transformable keypoints."""

    height, width = shape
    radius = arm_length * min(height, width)
    keypoints = np.ones((len(landmarks), 3, 3), dtype=np.float32)
    for index, (px, py, theta1, theta2) in enumerate(landmarks):
        tip_x, tip_y = px * width, py * height
        keypoints[index, 0, :2] = tip_x / width, tip_y / height
        keypoints[index, 1, :2] = (
            (tip_x + radius * math.cos(float(theta1))) / width,
            (tip_y + radius * math.sin(float(theta1))) / height,
        )
        keypoints[index, 2, :2] = (
            (tip_x + radius * math.cos(float(theta2))) / width,
            (tip_y + radius * math.sin(float(theta2))) / height,
        )
    return keypoints


class GeometryYOLODataset(YOLODataset):
    """Load SWIM landmark sidecars and let Ultralytics transform them as keypoints."""

    def __init__(self, *args, data: dict | None = None, **kwargs) -> None:
        data = copy(data or {})
        data["kpt_shape"] = [3, 3]
        data["flip_idx"] = [0, 1, 2]
        super().__init__(*args, data=data, task="obb", **kwargs)

    def get_labels(self) -> list[dict]:
        labels = super().get_labels()
        for label in labels:
            landmarks = read_landmarks(landmark_path_for_image(label["im_file"]))
            if len(landmarks) != len(label["cls"]):
                if self.augment:
                    raise ValueError(
                        f"OBB/landmark count mismatch for {label['im_file']}: "
                        f"{len(label['cls'])} boxes, {len(landmarks)} landmarks"
                    )
                label["keypoints"] = np.zeros((len(label["cls"]), 3, 3), dtype=np.float32)
            else:
                label["keypoints"] = landmarks_to_keypoints(landmarks, tuple(label["shape"]))
        self.use_keypoints = True
        return labels
