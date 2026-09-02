"""Train YOLOv8-OBB with landmark-supervised wake geometry refinement."""

from __future__ import annotations

import argparse

from wake_structure.artifacts import create_archive
from wake_structure.config import GeometryConfig
from wake_structure.model import GeometryOBBTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--geometry-config", default="configs/geometry.yaml")
    parser.add_argument("--model", default="yolov8n-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/wake_geometry")
    parser.add_argument("--name", default="yolov8n_obb_geometry")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--archive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GeometryConfig.from_yaml(args.geometry_config)
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "fraction": args.fraction,
        "seed": args.seed,
        "deterministic": True,
        "project": args.project,
        "name": args.name,
        "amp": args.amp,
        "task": "obb",
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    trainer = GeometryOBBTrainer(overrides=overrides, geometry_config=config)
    trainer.train()
    if args.archive:
        archive = create_archive([trainer.save_dir], args.archive)
        print(f"Run archive: {archive}")


if __name__ == "__main__":
    main()
