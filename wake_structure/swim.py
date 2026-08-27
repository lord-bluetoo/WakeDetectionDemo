"""Convert the original SWIM XML release to Ultralytics OBB format."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass
class ConversionStats:
    images: int = 0
    boxes: int = 0
    clipped_boxes: int = 0
    skipped_difficult: int = 0
    empty_labels: int = 0

    def add(self, other: "ConversionStats") -> None:
        for field in asdict(self):
            setattr(self, field, getattr(self, field) + getattr(other, field))


def rotated_box_corners(cx: float, cy: float, width: float, height: float, angle: float) -> list[tuple[float, float]]:
    """Return four ordered corners for an XML ``cx, cy, w, h, angle`` box."""

    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    local_corners = (
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    )
    return [
        (
            cx + x * cos_angle - y * sin_angle,
            cy + x * sin_angle + y * cos_angle,
        )
        for x, y in local_corners
    ]


def convert_annotation(xml_path: str | Path, *, clip_boxes: bool = True) -> tuple[list[list[float]], ConversionStats]:
    """Convert one SWIM XML file into normalized four-corner OBB records."""

    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    image_width = float(root.findtext("size/width", "0"))
    image_height = float(root.findtext("size/height", "0"))
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}")

    records: list[list[float]] = []
    stats = ConversionStats()
    for object_node in root.findall("object"):
        if object_node.findtext("difficult", "0").strip() == "1":
            stats.skipped_difficult += 1
            continue
        box_node = object_node.find("robndbox")
        if box_node is None:
            raise ValueError(f"Missing <robndbox> in {xml_path}")
        try:
            cx = float(box_node.findtext("cx", "nan"))
            cy = float(box_node.findtext("cy", "nan"))
            box_width = float(box_node.findtext("w", "nan"))
            box_height = float(box_node.findtext("h", "nan"))
            angle = float(box_node.findtext("angle", "nan"))
        except ValueError as error:
            raise ValueError(f"Invalid rotated-box number in {xml_path}") from error
        if not all(math.isfinite(value) for value in (cx, cy, box_width, box_height, angle)):
            raise ValueError(f"Non-finite rotated box in {xml_path}")
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Non-positive rotated box size in {xml_path}")

        corners = rotated_box_corners(cx, cy, box_width, box_height, angle)
        out_of_bounds = any(
            x < 0 or x > image_width or y < 0 or y > image_height for x, y in corners
        )
        if out_of_bounds:
            stats.clipped_boxes += 1
            if clip_boxes:
                corners = [
                    (min(max(x, 0.0), image_width), min(max(y, 0.0), image_height))
                    for x, y in corners
                ]

        normalized = []
        for x, y in corners:
            normalized.extend((x / image_width, y / image_height))
        records.append(normalized)
        stats.boxes += 1

    if not records:
        stats.empty_labels += 1
    return records, stats


def _find_split_file(source: Path, split: str) -> Path:
    candidates = [path for path in source.rglob(f"{split}.txt") if "imagesets" in path.as_posix().lower()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one ImageSets/{split}.txt under {source}, found {candidates}")
    return candidates[0]


def _index_by_stem(directory: Path, extensions: set[str] | None = None) -> dict[str, Path]:
    files = [path for path in directory.rglob("*") if path.is_file()]
    if extensions is not None:
        files = [path for path in files if path.suffix.lower() in extensions]
    index: dict[str, Path] = {}
    for path in files:
        if path.stem in index:
            raise ValueError(f"Duplicate stem {path.stem!r}: {index[path.stem]} and {path}")
        index[path.stem] = path
    return index


def _materialize_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return mode
    if mode in {"auto", "symlink"}:
        try:
            destination.symlink_to(source.resolve())
            return "symlink"
        except OSError:
            if mode == "symlink":
                raise
    if mode == "hardlink":
        os.link(source, destination)
        return "hardlink"
    shutil.copy2(source, destination)
    return "copy"


def convert_swim_dataset(
    source: str | Path,
    output: str | Path,
    *,
    image_mode: str = "auto",
    clip_boxes: bool = True,
) -> dict[str, ConversionStats]:
    """Convert official SWIM train/val/test splits and write ``swim.yaml``."""

    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"SWIM source does not exist: {source}")
    image_directory = source / "JPEGImages"
    annotation_directory = source / "Annotations"
    if not image_directory.is_dir() or not annotation_directory.is_dir():
        raise FileNotFoundError(f"Expected JPEGImages and Annotations under {source}")
    if image_mode not in {"auto", "copy", "symlink", "hardlink"}:
        raise ValueError(f"Unsupported image mode: {image_mode}")

    images = _index_by_stem(image_directory, IMAGE_EXTENSIONS)
    annotations = _index_by_stem(annotation_directory, {".xml"})
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, ConversionStats] = {}

    for split in SPLITS:
        identifiers = [
            line.strip().rsplit(".", 1)[0]
            for line in _find_split_file(source, split).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        split_stats = ConversionStats()
        for identifier in identifiers:
            if identifier not in images:
                raise FileNotFoundError(f"Missing image for {split} identifier {identifier}")
            if identifier not in annotations:
                raise FileNotFoundError(f"Missing annotation for {split} identifier {identifier}")

            source_image = images[identifier]
            target_image = output / "images" / split / f"{identifier}{source_image.suffix.lower()}"
            _materialize_image(source_image, target_image, image_mode)

            records, annotation_stats = convert_annotation(annotations[identifier], clip_boxes=clip_boxes)
            target_label = output / "labels" / split / f"{identifier}.txt"
            target_label.parent.mkdir(parents=True, exist_ok=True)
            lines = ["0 " + " ".join(f"{coordinate:.8f}" for coordinate in record) for record in records]
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_stats.images += 1
            split_stats.add(annotation_stats)
        results[split] = split_stats

    dataset_yaml = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "wake"},
    }
    (output / "swim.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path containing JPEGImages, Annotations, and ImageSets")
    parser.add_argument("--output", default="/kaggle/working/swim_yolo_obb")
    parser.add_argument(
        "--image-mode",
        choices=("auto", "copy", "symlink", "hardlink"),
        default="auto",
        help="auto uses symlinks on Kaggle and falls back to copying when unavailable",
    )
    parser.add_argument(
        "--no-clip-boxes",
        action="store_true",
        help="Keep out-of-image corners instead of clipping them into [0, 1]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = convert_swim_dataset(
        args.source,
        args.output,
        image_mode=args.image_mode,
        clip_boxes=not args.no_clip_boxes,
    )
    print(f"Converted SWIM to: {Path(args.output).resolve()}")
    total = ConversionStats()
    for split, stats in results.items():
        total.add(stats)
        print(f"{split:>5}: {asdict(stats)}")
    print(f"total: {asdict(total)}")
    if total.clipped_boxes:
        print("WARNING: Some OBB corners crossed the image boundary and were clipped; inspect this count.")
    print(f"Dataset YAML: {Path(args.output).resolve() / 'swim.yaml'}")


if __name__ == "__main__":
    main()

