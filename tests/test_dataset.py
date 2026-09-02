import math

import cv2
import numpy as np
from ultralytics.cfg import get_cfg

from wake_structure.dataset import GeometryYOLODataset, landmarks_to_keypoints, read_landmarks


def test_landmark_sidecar_becomes_three_keypoints(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("0.25 0.5 0.0 1.5707963268\n", encoding="utf-8")
    landmarks = read_landmarks(path)
    keypoints = landmarks_to_keypoints(landmarks, (100, 200), arm_length=0.1)
    assert keypoints.shape == (1, 3, 3)
    np.testing.assert_allclose(keypoints[0, 0, :2], [0.25, 0.5])
    assert keypoints[0, 1, 0] > keypoints[0, 0, 0]
    assert math.isclose(float(keypoints[0, 2, 0]), 0.25, abs_tol=1e-6)
    assert keypoints[0, 2, 1] > keypoints[0, 0, 1]


def test_dataset_transforms_obb_and_landmarks_together(tmp_path) -> None:
    root = tmp_path / "dataset"
    for directory in ("images/train", "labels/train", "landmarks/train"):
        (root / directory).mkdir(parents=True)
    cv2.imwrite(str(root / "images/train/sample.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
    (root / "labels/train/sample.txt").write_text(
        "0 0.2 0.4 0.8 0.4 0.8 0.6 0.2 0.6\n", encoding="utf-8"
    )
    (root / "landmarks/train/sample.txt").write_text("0.25 0.5 -0.4 0.4\n", encoding="utf-8")
    hyp = get_cfg()
    hyp.mosaic = hyp.mixup = hyp.copy_paste = hyp.cutmix = 0.0
    hyp.degrees = hyp.translate = hyp.scale = hyp.shear = hyp.perspective = 0.0
    hyp.fliplr, hyp.flipud = 1.0, 0.0
    dataset = GeometryYOLODataset(
        img_path=str(root / "images/train"),
        imgsz=64,
        batch_size=1,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
        classes=None,
        data={"names": {0: "wake"}, "nc": 1, "channels": 3},
        fraction=1.0,
    )
    sample = dataset[0]
    assert sample["bboxes"].shape == (1, 5)
    assert sample["keypoints"].shape == (1, 3, 3)
    np.testing.assert_allclose(sample["keypoints"][0, 0, :2], [0.75, 0.5], atol=1e-5)

    (root / "landmarks/train/sample.txt").unlink()
    evaluation = GeometryYOLODataset(
        img_path=str(root / "images/train"),
        imgsz=64,
        batch_size=1,
        augment=False,
        hyp=get_cfg(),
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
        classes=None,
        data={"names": {0: "wake"}, "nc": 1, "channels": 3},
        fraction=1.0,
    )
    assert evaluation[0]["keypoints"][..., 2].sum() == 0
