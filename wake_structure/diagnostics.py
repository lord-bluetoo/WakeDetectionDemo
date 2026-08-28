"""Quantitative and visual diagnostics for a trained wake Structure Head."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from .artifacts import create_archive


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_device(value: str) -> torch.device:
    """Resolve Ultralytics-style device strings such as ``0`` or ``cpu``."""

    if value.lower() == "cpu":
        return torch.device("cpu")
    if value.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {value} was requested, but CUDA is unavailable.")
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def load_structure_model(weights: str | Path, device: torch.device) -> torch.nn.Module:
    """Load an Ultralytics checkpoint that contains ``StructureOBBModel``."""

    # Importing the custom class before torch.load makes its pickle path available.
    from .model import StructureOBBModel  # noqa: F401

    try:
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        checkpoint = torch.load(weights, map_location=device)
    model = checkpoint
    if isinstance(checkpoint, dict):
        model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None or not hasattr(model, "structure_head") or not hasattr(model, "structure_maps"):
        raise TypeError(
            "Checkpoint does not contain a StructureOBBModel with structure_head and structure_maps. "
            "Use the Structure run's best.pt, not the Baseline checkpoint."
        )
    return model.float().to(device).eval()


def _dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(data.get("path") or data_yaml.parent)
    return root if root.is_absolute() else (data_yaml.parent / root).resolve()


def find_split_images(data_yaml: str | Path, split: str) -> list[Path]:
    """Resolve image files from one directory-based Ultralytics split."""

    yaml_path = Path(data_yaml).resolve()
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if split not in data:
        raise KeyError(f"Split {split!r} is absent from {yaml_path}")
    root = _dataset_root(yaml_path, data)
    values = data[split] if isinstance(data[split], list) else [data[split]]
    images: list[Path] = []
    for value in values:
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            images.extend(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                if line.strip():
                    images.append(candidate if candidate.is_absolute() else root / candidate)
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            raise FileNotFoundError(f"Unsupported or missing split path: {path}")
    # Keep the dataset-facing path instead of resolving image symlinks into
    # /kaggle/input; label_path_for_image relies on the images/labels layout.
    return sorted({path.absolute() for path in images})


def label_path_for_image(image_path: str | Path) -> Path:
    """Map ``.../images/<split>/x.jpg`` to ``.../labels/<split>/x.txt``."""

    path = Path(image_path)
    parts = list(path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indices:
        raise ValueError(f"Image path has no 'images' directory component: {path}")
    parts[image_indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_obb_labels(label_path: str | Path) -> list[np.ndarray]:
    """Read normalized Ultralytics OBB quadrilaterals from a label file."""

    path = Path(label_path)
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    quadrilaterals: list[np.ndarray] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) != 9:
            raise ValueError(f"Expected 9 columns at {path}:{line_number}, found {len(values)}")
        quadrilaterals.append(np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(4, 2))
    return quadrilaterals


def letterbox_image(
    image_bgr: np.ndarray,
    quadrilaterals: list[np.ndarray],
    image_size: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Letterbox an image and transform normalized quadrilaterals with it."""

    height, width = image_bgr.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized

    transformed: list[np.ndarray] = []
    for quadrilateral in quadrilaterals:
        points = quadrilateral.copy()
        points[:, 0] = (points[:, 0] * width * scale + left) / image_size
        points[:, 1] = (points[:, 1] * height * scale + top) / image_size
        transformed.append(points)
    return canvas, transformed


def _instance_masks(quadrilaterals: list[np.ndarray], height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for quadrilateral in quadrilaterals:
        pixels = quadrilateral.copy()
        pixels[:, 0] *= width
        pixels[:, 1] *= height
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(pixels).astype(np.int32)], 1)
        masks.append(mask.astype(bool))
    return masks


def _long_axis_angle(quadrilateral: np.ndarray, height: int, width: int) -> float:
    points = quadrilateral * np.asarray([width, height], dtype=np.float32)
    edges = np.roll(points, -1, axis=0) - points
    edge = edges[np.square(edges).sum(axis=1).argmax()]
    return float(np.mod(np.arctan2(edge[1], edge[0]), math.pi))


def _top_fraction_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    if not values.size:
        return float("nan")
    count = max(1, math.ceil(values.size * fraction))
    return float(np.partition(values, values.size - count)[-count:].mean())


def _axial_error_degrees(theta: np.ndarray, weights: np.ndarray, target: float) -> float:
    weight_sum = weights.sum()
    if weight_sum <= 1e-8:
        return float("nan")
    vx = float((weights * np.cos(2 * theta)).sum() / weight_sum)
    vy = float((weights * np.sin(2 * theta)).sum() / weight_sum)
    prediction = 0.5 * math.atan2(vy, vx) % math.pi
    difference = (prediction - target + math.pi / 2) % math.pi - math.pi / 2
    return abs(math.degrees(difference))


def compute_map_diagnostics(
    maps: dict[str, torch.Tensor | np.ndarray],
    quadrilaterals: list[np.ndarray],
    image_name: str,
) -> dict[str, float | int | str]:
    """Summarize response localization, confidence, entropy, and direction error."""

    def array(name: str) -> np.ndarray:
        value = maps[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        result = np.asarray(value)
        while result.ndim > 2 and result.shape[0] == 1:
            result = result[0]
        return result

    presence = array("presence")
    confidence = array("confidence")
    theta = array("theta")
    gate = array("directional_gate")
    probabilities = array("orientation_distribution")
    if probabilities.ndim != 3:
        raise ValueError(f"Expected q_theta with shape [K,H,W], got {probabilities.shape}")
    height, width = presence.shape
    masks = _instance_masks(quadrilaterals, height, width)
    union = np.logical_or.reduce(masks) if masks else np.zeros((height, width), dtype=bool)
    outside = ~union
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=0)
    entropy /= math.log(probabilities.shape[0])

    direction_errors = []
    for quadrilateral, mask in zip(quadrilaterals, masks):
        target = _long_axis_angle(quadrilateral, height, width)
        direction_errors.append(_axial_error_degrees(theta[mask], gate[mask], target))
    finite_errors = [value for value in direction_errors if math.isfinite(value)]

    inside_presence = presence[union]
    outside_presence = presence[outside]
    inside_confidence = confidence[union]
    inside_gate = gate[union]
    outside_gate = gate[outside]
    return {
        "image": image_name,
        "instances": len(masks),
        "presence_global_mean": float(presence.mean()),
        "presence_global_std": float(presence.std()),
        "presence_inside_mean": float(inside_presence.mean()) if inside_presence.size else float("nan"),
        "presence_inside_top10_mean": _top_fraction_mean(inside_presence),
        "presence_outside_mean": float(outside_presence.mean()) if outside_presence.size else float("nan"),
        "presence_top10_minus_outside": (
            _top_fraction_mean(inside_presence) - float(outside_presence.mean())
            if inside_presence.size and outside_presence.size
            else float("nan")
        ),
        "confidence_inside_mean": float(inside_confidence.mean()) if inside_confidence.size else float("nan"),
        "gate_inside_top10_mean": _top_fraction_mean(inside_gate),
        "gate_outside_mean": float(outside_gate.mean()) if outside_gate.size else float("nan"),
        "orientation_entropy_inside": float(entropy[union].mean()) if union.any() else float("nan"),
        "orientation_error_deg": float(np.mean(finite_errors)) if finite_errors else float("nan"),
    }


def _overlay(ax: Any, image_rgb: np.ndarray, values: np.ndarray, title: str, cmap: str = "magma") -> None:
    ax.imshow(image_rgb)
    ax.imshow(values, cmap=cmap, vmin=0, vmax=1, alpha=0.65, extent=(0, image_rgb.shape[1], image_rgb.shape[0], 0))
    ax.set_title(title)
    ax.axis("off")


def save_structure_figure(
    image_bgr: np.ndarray,
    quadrilaterals: list[np.ndarray],
    maps: dict[str, torch.Tensor | np.ndarray],
    output: str | Path,
) -> None:
    """Save a six-panel qualitative diagnostic figure for one image."""

    def array(name: str) -> np.ndarray:
        value = maps[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        result = np.asarray(value)
        while result.ndim > 2 and result.shape[0] == 1:
            result = result[0]
        return result

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    presence = array("presence")
    confidence = array("confidence")
    gate = array("directional_gate")
    theta = array("theta")
    height, width = theta.shape

    figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(image_rgb)
    for quadrilateral in quadrilaterals:
        points = quadrilateral * np.asarray([image_rgb.shape[1], image_rgb.shape[0]])
        closed = np.vstack((points, points[0]))
        axes[0, 0].plot(closed[:, 0], closed[:, 1], color="#00E5FF", linewidth=1.5)
    axes[0, 0].set_title("Input + OBB")
    axes[0, 0].set_xlim(0, image_rgb.shape[1])
    axes[0, 0].set_ylim(image_rgb.shape[0], 0)
    axes[0, 0].axis("off")

    _overlay(axes[0, 1], image_rgb, presence, "P: wake presence")
    _overlay(axes[0, 2], image_rgb, confidence, "C: direction concentration", cmap="viridis")
    _overlay(axes[1, 0], image_rgb, gate, "P × C: directional gate", cmap="inferno")

    axes[1, 1].imshow(image_rgb)
    axes[1, 1].imshow(
        np.degrees(theta),
        cmap="hsv",
        vmin=0,
        vmax=180,
        alpha=np.clip(gate, 0, 0.8),
        extent=(0, image_rgb.shape[1], image_rgb.shape[0], 0),
    )
    axes[1, 1].set_title("θ hue, opacity=P×C")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(image_rgb)
    stride = max(1, min(height, width) // 12)
    ys, xs = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    sampled_gate = gate[ys, xs]
    keep = sampled_gate >= max(0.05, float(np.quantile(gate, 0.80)))
    sampled_theta = theta[ys, xs]
    x_pixels = (xs[keep] + 0.5) / width * image_rgb.shape[1]
    y_pixels = (ys[keep] + 0.5) / height * image_rgb.shape[0]
    half_length = image_rgb.shape[1] / 55
    dx = np.cos(sampled_theta[keep]) * half_length
    dy = np.sin(sampled_theta[keep]) * half_length
    segments = np.stack(
        (
            np.stack((x_pixels - dx, y_pixels - dy), axis=1),
            np.stack((x_pixels + dx, y_pixels + dy), axis=1),
        ),
        axis=1,
    ) if x_pixels.size else np.empty((0, 2, 2))
    axes[1, 2].add_collection(LineCollection(segments, colors="#00FF7F", linewidths=1.2))
    axes[1, 2].set_xlim(0, image_rgb.shape[1])
    axes[1, 2].set_ylim(image_rgb.shape[0], 0)
    axes[1, 2].set_title("High-confidence axial directions")
    axes[1, 2].axis("off")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _finite_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image diagnostics and emit explicitly heuristic collapse flags."""

    numeric_keys = [
        "presence_global_mean",
        "presence_global_std",
        "presence_inside_mean",
        "presence_inside_top10_mean",
        "presence_outside_mean",
        "presence_top10_minus_outside",
        "confidence_inside_mean",
        "gate_inside_top10_mean",
        "gate_outside_mean",
        "orientation_entropy_inside",
        "orientation_error_deg",
    ]
    means = {key: _finite_mean(rows, key) for key in numeric_keys}
    flags = {
        "low_presence_spatial_variation": (
            means["presence_global_std"] is not None and means["presence_global_std"] < 0.01
        ),
        "weak_inside_outside_separation": (
            means["presence_top10_minus_outside"] is not None
            and means["presence_top10_minus_outside"] < 0.10
        ),
        "low_direction_concentration": (
            means["confidence_inside_mean"] is not None and means["confidence_inside_mean"] < 0.10
        ),
    }
    return {
        "images": len(rows),
        "instances": int(sum(int(row["instances"]) for row in rows)),
        "means": means,
        "heuristic_flags": flags,
        "note": "Flags are screening heuristics, not statistical significance tests.",
    }


def save_training_curves(results_csv: str | Path, output: str | Path) -> bool:
    """Plot structure losses and detection mAP from an Ultralytics results.csv."""

    path = Path(results_csv)
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    columns = {key.strip(): key for key in rows[0]}
    epochs = [float(row[columns.get("epoch", "epoch")]) + 1 for row in rows]
    structure_columns = [name for name in columns if "structure_" in name]
    metric_columns = [name for name in columns if "metrics/mAP" in name]
    if not structure_columns and not metric_columns:
        return False

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for name in structure_columns:
        axes[0].plot(epochs, [float(row[columns[name]]) for row in rows], label=name.replace("train/", ""))
    axes[0].set_title("Structure losses")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    if structure_columns:
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "No structure-loss columns", ha="center", va="center")
    for name in metric_columns:
        axes[1].plot(epochs, [float(row[columns[name]]) for row in rows], label=name.replace("metrics/", ""))
    axes[1].set_title("Detection validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    if metric_columns:
        axes[1].legend(fontsize=8)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def run_diagnostics(
    *,
    weights: str | Path,
    data_yaml: str | Path,
    output: str | Path,
    split: str = "val",
    num_images: int = 12,
    image_size: int = 640,
    device: str = "0",
    seed: int = 42,
    results_csv: str | Path | None = None,
    archive: str | Path | None = None,
) -> Path:
    """Run reproducible qualitative and quantitative Structure Head diagnostics."""

    if num_images < 1:
        raise ValueError("num_images must be positive.")
    output_path = Path(output).resolve()
    figure_directory = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    images = find_split_images(data_yaml, split)
    labelled = [image for image in images if read_obb_labels(label_path_for_image(image))]
    if not labelled:
        raise RuntimeError(f"No labelled images found in split {split!r}.")
    random.Random(seed).shuffle(labelled)
    selected = labelled[: min(num_images, len(labelled))]

    torch_device = resolve_device(device)
    model = load_structure_model(weights, torch_device)
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(selected, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        quadrilaterals = read_obb_labels(label_path_for_image(image_path))
        canvas, transformed = letterbox_image(image, quadrilaterals, image_size)
        tensor = torch.from_numpy(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(torch_device, dtype=torch.float32) / 255.0
        with torch.inference_mode():
            maps = model.structure_maps(tensor)
        row = compute_map_diagnostics(maps, transformed, image_path.name)
        rows.append(row)
        save_structure_figure(
            canvas,
            transformed,
            maps,
            figure_directory / f"{index:02d}_{image_path.stem}.png",
        )

    fieldnames = list(rows[0])
    with (output_path / "diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows)
    feature_guidance = getattr(model, "feature_guidance", None)
    guidance_alpha = (
        float(feature_guidance.alpha.detach().float().cpu()) if feature_guidance is not None else None
    )
    summary.update(
        {
            "feature_guidance_enabled": feature_guidance is not None,
            "guidance_alpha": guidance_alpha,
            "guidance_residual_scale": math.tanh(guidance_alpha) if guidance_alpha is not None else None,
            "weights": str(Path(weights).resolve()),
            "data": str(Path(data_yaml).resolve()),
            "split": split,
            "seed": seed,
            "selected_images": [path.name for path in selected],
        }
    )
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_path / "selected_images.txt").write_text(
        "\n".join(str(path) for path in selected) + "\n", encoding="utf-8"
    )

    inferred_results = Path(weights).resolve().parent.parent / "results.csv"
    curve_source = Path(results_csv).resolve() if results_csv else inferred_results
    save_training_curves(curve_source, output_path / "training_curves.png")

    if archive:
        archive_path = create_archive([Path(weights).resolve().parent.parent, output_path], archive)
        print(f"Diagnostic archive: {archive_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Structure run best.pt or last.pt")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--output", default="structure_diagnostics")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", help="Optional results.csv; inferred from the weights run by default")
    parser.add_argument("--archive", help="Optional .zip destination containing the run and diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_diagnostics(
        weights=args.weights,
        data_yaml=args.data,
        output=args.output,
        split=args.split,
        num_images=args.num_images,
        image_size=args.imgsz,
        device=args.device,
        seed=args.seed,
        results_csv=args.results,
        archive=args.archive,
    )
    print(f"Diagnostics saved to: {output}")
