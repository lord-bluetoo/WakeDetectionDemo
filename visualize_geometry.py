"""Visualize landmark geometry predictions and refinement gates from a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from wake_structure.dataset import landmark_path_for_image, read_landmarks
from wake_structure.model import GeometryOBBModel  # noqa: F401  # Required when torch loads the custom checkpoint.


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Geometry run weights/best.pt")
    parser.add_argument("--data", required=True, help="Converted SWIM YAML")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", default="runs/geometry_visualization")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _dataset_images(data_path: Path, split: str) -> list[Path]:
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    location = Path(data[split])
    if not location.is_absolute():
        location = root / location
    if location.is_file():
        paths = [Path(line.strip()) for line in location.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path if path.is_absolute() else root / path for path in paths]
    return sorted(path for path in location.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _load_model(weights: Path, device: torch.device) -> GeometryOBBModel:
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    model = (checkpoint.get("ema") or checkpoint.get("model")) if isinstance(checkpoint, dict) else checkpoint
    if model is None or not hasattr(model, "geometry_diagnostics"):
        raise ValueError(f"{weights} is not a GeometryOBBModel checkpoint")
    return model.float().to(device).eval()


def _letterbox(image: np.ndarray, size: int) -> tuple[torch.Tensor, float, tuple[int, int]]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized.shape[1]) // 2
    pad_y = (size - resized.shape[0]) // 2
    canvas[pad_y : pad_y + resized.shape[0], pad_x : pad_x + resized.shape[1]] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    return tensor, scale, (pad_x, pad_y)


def _map_to_image(values: torch.Tensor, shape: tuple[int, int], size: int, scale: float, pad: tuple[int, int]) -> np.ndarray:
    array = values.detach().float().cpu().numpy()
    square = cv2.resize(array, (size, size), interpolation=cv2.INTER_LINEAR)
    height, width = shape
    pad_x, pad_y = pad
    crop = square[pad_y : pad_y + round(height * scale), pad_x : pad_x + round(width * scale)]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def _overlay(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(cv2.addWeighted(image, 0.55, colored, 0.45, 0), cv2.COLOR_BGR2RGB)


def _line_endpoint(point: tuple[float, float], angle: float, length: float) -> tuple[int, int]:
    return round(point[0] + length * math.cos(angle)), round(point[1] + length * math.sin(angle))


def _draw_geometry(
    image: np.ndarray,
    geometries: list[tuple[float, float, float, float]],
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    thickness: int = 2,
) -> np.ndarray:
    output = image.copy()
    length = 0.35 * min(image.shape[:2])
    for x, y, theta1, theta2 in geometries:
        point = (round(x), round(y))
        cv2.circle(output, point, max(3, thickness + 1), (255, 255, 255), -1)
        cv2.line(output, point, _line_endpoint((x, y), theta1, length), colors[0], thickness)
        cv2.line(output, point, _line_endpoint((x, y), theta2, length), colors[1], thickness)
    return output


def _angle_error(first: float, second: float) -> float:
    return abs(math.degrees((first - second + math.pi) % (2 * math.pi) - math.pi))


def _opening_angle(first: float, second: float) -> float:
    difference = abs((first - second + math.pi) % (2 * math.pi) - math.pi)
    return math.degrees(difference)


def _predicted_geometry(
    maps: dict[str, torch.Tensor],
    count: int,
    image_shape: tuple[int, int],
    input_size: int,
    scale: float,
    pad: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    heatmap = maps["tip"][0, 0]
    pooled = F.max_pool2d(heatmap[None, None], 3, stride=1, padding=1)[0, 0]
    peaks = heatmap.masked_fill(heatmap < pooled, -1)
    indices = peaks.flatten().topk(min(max(count, 1), peaks.numel())).indices
    width_cells = heatmap.shape[1]
    output = []
    for index in indices:
        y_cell = int(index.item() // width_cells)
        x_cell = int(index.item() % width_cells)
        offset = maps["offset"][0, :, y_cell, x_cell]
        input_x = (x_cell + float(offset[0])) / width_cells * input_size
        input_y = (y_cell + float(offset[1])) / heatmap.shape[0] * input_size
        image_x = (input_x - pad[0]) / scale
        image_y = (input_y - pad[1]) / scale
        if 0 <= image_x < image_shape[1] and 0 <= image_y < image_shape[0]:
            output.append(
                (
                    image_x,
                    image_y,
                    float(maps["theta1"][0, 0, y_cell, x_cell]),
                    float(maps["theta2"][0, 0, y_cell, x_cell]),
                )
            )
    return output


def _match_metrics(
    image_name: str,
    ground_truth: list[tuple[float, float, float, float]],
    predictions: list[tuple[float, float, float, float]],
) -> list[dict[str, float | str]]:
    rows = []
    remaining = list(predictions)
    for gt_x, gt_y, gt1, gt2 in ground_truth:
        if not remaining:
            break
        best = min(range(len(remaining)), key=lambda i: (remaining[i][0] - gt_x) ** 2 + (remaining[i][1] - gt_y) ** 2)
        pred_x, pred_y, pred1, pred2 = remaining.pop(best)
        direct = (_angle_error(pred1, gt1) + _angle_error(pred2, gt2)) / 2
        swapped = (_angle_error(pred1, gt2) + _angle_error(pred2, gt1)) / 2
        rows.append(
            {
                "image": image_name,
                "tip_error_px": math.hypot(pred_x - gt_x, pred_y - gt_y),
                "arm_mae_deg": min(direct, swapped),
                "opening_angle_error_deg": abs(_opening_angle(pred1, pred2) - _opening_angle(gt1, gt2)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device_name = args.device if args.device == "cpu" or args.device.startswith("cuda") else f"cuda:{args.device}"
    device = torch.device(device_name)
    model = _load_model(Path(args.weights), device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, float | str]] = []

    for image_path in _dataset_images(Path(args.data), args.split)[: args.limit]:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        landmarks = read_landmarks(landmark_path_for_image(image_path))
        height, width = image.shape[:2]
        ground_truth = [(px * width, py * height, theta1, theta2) for px, py, theta1, theta2 in landmarks]
        tensor, scale, pad = _letterbox(image, args.imgsz)
        maps = model.geometry_diagnostics(tensor.to(device))
        predictions = _predicted_geometry(maps, len(ground_truth), (height, width), args.imgsz, scale, pad)
        metric_rows.extend(_match_metrics(image_path.name, ground_truth, predictions))

        feature_stride = args.imgsz / maps["structure"].shape[-1]
        configured_band_width = getattr(model.geometry_config.loss, "structure_band_width", 2.0)
        band_width = configured_band_width * feature_stride / scale
        gt_view = _draw_geometry(
            image,
            ground_truth,
            ((0, 100, 0), (0, 100, 100)),
            thickness=max(3, round(2 * band_width)),
        )
        gt_view = _draw_geometry(gt_view, ground_truth, ((0, 255, 0), (0, 255, 255)), thickness=2)
        prediction_view = _draw_geometry(gt_view, predictions, ((255, 0, 255), (255, 255, 0)), thickness=2)
        panels = [
            ("GT search bands / predictions", cv2.cvtColor(prediction_view, cv2.COLOR_BGR2RGB)),
            ("P_structure", _overlay(image, _map_to_image(maps["structure"][0, 0], (height, width), args.imgsz, scale, pad))),
            ("tip heatmap", _overlay(image, _map_to_image(maps["tip"][0, 0], (height, width), args.imgsz, scale, pad))),
            ("denoise gate", _overlay(image, _map_to_image(maps["denoise_gate"][0, 0], (height, width), args.imgsz, scale, pad))),
            ("enhancement gate", _overlay(image, _map_to_image(maps["enhancement_gate"][0, 0], (height, width), args.imgsz, scale, pad))),
            ("feature change", _overlay(image, _map_to_image(maps["feature_change"][0, 0], (height, width), args.imgsz, scale, pad))),
        ]
        figure, axes = plt.subplots(2, 3, figsize=(15, 10))
        for axis, (title, panel) in zip(axes.flat, panels):
            axis.imshow(panel)
            axis.set_title(title)
            axis.axis("off")
        denoise_scale = float(maps.get("denoise_scale", torch.tensor(0)))
        feature_scale = float(maps.get("feature_scale", torch.tensor(0)))
        figure.suptitle(f"{image_path.name} | denoise={denoise_scale:.3f}, enhance={feature_scale:.3f}")
        figure.tight_layout()
        figure.savefig(output / f"{image_path.stem}_geometry.png", dpi=150)
        plt.close(figure)

    if metric_rows:
        with (output / "geometry_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
        for name in ("tip_error_px", "arm_mae_deg", "opening_angle_error_deg"):
            values = [float(row[name]) for row in metric_rows]
            print(f"{name}: {np.mean(values):.3f} ± {np.std(values):.3f}")
    print(f"Saved geometry diagnostics to {output.resolve()}")


if __name__ == "__main__":
    main()
