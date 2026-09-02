"""One-command Kaggle runner for the landmark geometry experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def run(command: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def ensure_dependencies(project_root: Path) -> None:
    try:
        version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        version = ""
    if (8, 3) <= _version_tuple(version) < (9,):
        return
    run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3,<9", "PyYAML>=6.0"],
        cwd=project_root,
    )


def _is_swim_source(path: Path) -> bool:
    return all((path / name).is_dir() for name in ("JPEGImages", "Annotations", "Landmarks", "ImageSets"))


def find_swim_source(explicit: str | None) -> Path:
    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not _is_swim_source(source):
            raise FileNotFoundError(f"Not a SWIM source directory: {source}")
        return source
    candidates = [
        path.parent
        for root in (KAGGLE_INPUT, KAGGLE_WORKING)
        if root.is_dir()
        for path in root.rglob("JPEGImages")
        if _is_swim_source(path.parent)
    ]
    if not candidates:
        raise FileNotFoundError("Could not find SWIM with JPEGImages, Annotations, Landmarks, and ImageSets.")
    return sorted(set(candidates), key=lambda path: (len(path.parts), path.as_posix()))[0]


def resolve_data(args: argparse.Namespace, project_root: Path) -> Path:
    if args.data:
        return Path(args.data).expanduser().resolve()
    output = KAGGLE_WORKING / "swim_yolo_geometry"
    data = output / "swim.yaml"
    if not data.is_file():
        run(
            [
                sys.executable,
                str(project_root / "prepare_swim.py"),
                "--source",
                str(find_swim_source(args.source)),
                "--output",
                str(output),
            ],
            cwd=project_root,
        )
    return data


def resolve_model(explicit: str | None) -> str:
    if explicit:
        return explicit
    for root in (KAGGLE_WORKING, KAGGLE_INPUT):
        matches = sorted(root.rglob("yolov8n-obb.pt")) if root.is_dir() else []
        if matches:
            return str(matches[0])
    return "yolov8n-obb.pt"


def newest_run_weights(run_root: Path, started_at: float) -> Path:
    candidates = [
        path
        for path in run_root.rglob("best.pt")
        if path.parent.name == "weights" and path.stat().st_mtime >= started_at - 5
    ]
    if not candidates:
        raise FileNotFoundError(f"No new best.pt found under {run_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def unique_output(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate output beside {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data")
    parser.add_argument("--source")
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("baseline", "geometry", "both"), default="both")
    parser.add_argument("--name-prefix", default="pilot20_s42")
    return parser.parse_args()


def train_variant(
    variant: str,
    args: argparse.Namespace,
    project_root: Path,
    data: Path,
    model: str,
    run_root: Path,
) -> tuple[Path, Path]:
    name = f"{args.name_prefix}_{variant}"
    archive = unique_output(KAGGLE_WORKING / f"{name}.zip")
    script = project_root / ("train_baseline.py" if variant == "baseline" else "train_geometry.py")
    command = [
        sys.executable,
        str(script),
        "--data",
        str(data),
        "--model",
        model,
        "--epochs",
        str(args.epochs),
        "--fraction",
        str(args.fraction),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--project",
        str(run_root),
        "--name",
        name,
        "--archive",
        str(archive),
    ]
    if variant == "geometry":
        command[4:4] = ["--geometry-config", str(project_root / "configs" / "geometry.yaml")]
    started_at = time.time()
    run(command, cwd=project_root)
    return newest_run_weights(run_root, started_at), archive


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.environ.setdefault("WANDB_DISABLED", "true")
    ensure_dependencies(project_root)
    data = resolve_data(args, project_root)
    model = resolve_model(args.model)
    run_root = KAGGLE_WORKING / "runs" / "wake_geometry"
    variants = ("baseline", "geometry") if args.mode == "both" else (args.mode,)
    for variant in variants:
        weights, archive = train_variant(variant, args, project_root, data, model, run_root)
        print(f"{variant:>8} best weights: {weights}")
        print(f"{variant:>8} archive:      {archive}")


if __name__ == "__main__":
    main()
