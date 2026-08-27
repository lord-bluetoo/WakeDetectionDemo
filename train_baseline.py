"""Train the unchanged YOLOv8n-OBB baseline."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Ultralytics OBB dataset YAML")
    parser.add_argument("--model", default="yolov8n-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/wake_ablation")
    parser.add_argument("--name", default="yolov8n_obb_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
        task="obb",
    )


if __name__ == "__main__":
    main()

