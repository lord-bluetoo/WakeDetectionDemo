"""One-command Kaggle runner for the V2 structure-guided OBB experiment."""

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
    """Print and run a subprocess without hiding its notebook output."""

    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.match(r"(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in numbers.group(1).split(".")) if numbers else ()


def ensure_dependencies(project_root: Path, install: bool) -> None:
    """Install the declared runtime only when Kaggle does not already provide it."""

    try:
        version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        version = ""
    compatible = (8, 3) <= _version_tuple(version) < (9,)
    if compatible:
        print(f"Ultralytics {version} is already available.")
        return
    if not install:
        raise RuntimeError(
            f"Compatible Ultralytics is unavailable (found {version or 'none'}). "
            "Rerun without --no-install-dependencies."
        )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "ultralytics>=8.3,<9",
            "PyYAML>=6.0",
        ],
        cwd=project_root,
    )


def _is_swim_source(path: Path) -> bool:
    return all((path / name).is_dir() for name in ("JPEGImages", "Annotations", "ImageSets"))


def find_swim_source(explicit: str | None) -> Path:
    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not _is_swim_source(source):
            raise FileNotFoundError(f"Not a SWIM source directory: {source}")
        return source

    candidates: list[Path] = []
    for root in (KAGGLE_INPUT, KAGGLE_WORKING):
        if not root.is_dir():
            continue
        candidates.extend(path.parent for path in root.rglob("JPEGImages") if _is_swim_source(path.parent))
    unique = sorted(set(candidates), key=lambda path: (len(path.parts), path.as_posix()))
    if not unique:
        raise FileNotFoundError(
            "Could not find SWIM_Dataset_1.0.0 under /kaggle/input. "
            "Attach the SWIM dataset to this notebook or pass --source."
        )
    if len(unique) > 1:
        print(f"Found multiple SWIM sources; using {unique[0]}")
    return unique[0]


def resolve_data_yaml(args: argparse.Namespace, project_root: Path) -> Path:
    if args.data:
        data = Path(args.data).expanduser().resolve()
        if not data.is_file():
            raise FileNotFoundError(f"Dataset YAML does not exist: {data}")
        return data

    common = (
        KAGGLE_WORKING / "swim_yolo_obb_raw" / "swim.yaml",
        KAGGLE_WORKING / "swim_yolo_obb" / "swim.yaml",
    )
    for candidate in common:
        if candidate.is_file():
            print(f"Using existing converted dataset: {candidate}")
            return candidate

    source = find_swim_source(args.source)
    output = KAGGLE_WORKING / "swim_yolo_obb_raw"
    run(
        [
            sys.executable,
            str(project_root / "prepare_swim.py"),
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        cwd=project_root,
    )
    data = output / "swim.yaml"
    if not data.is_file():
        raise RuntimeError(f"Conversion finished without creating {data}")
    return data


def resolve_model(explicit: str | None) -> str:
    if explicit:
        return explicit
    for root in (KAGGLE_WORKING, KAGGLE_INPUT):
        if root.is_dir():
            matches = sorted(root.rglob("yolov8n-obb.pt"), key=lambda path: (len(path.parts), path.as_posix()))
            if matches:
                print(f"Using attached pretrained weights: {matches[0]}")
                return str(matches[0])
    print("Using yolov8n-obb.pt; Ultralytics will download it if it is not cached.")
    return "yolov8n-obb.pt"


def newest_run_weights(run_root: Path, started_at: float) -> Path:
    candidates = [
        path
        for path in run_root.rglob("best.pt")
        if path.parent.name == "weights" and path.stat().st_mtime >= started_at - 5
    ]
    if not candidates:
        raise FileNotFoundError(f"Training completed but no new best.pt was found under {run_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def unique_output(path: Path) -> Path:
    """Return a non-existing sibling path without deleting a previous result."""

    if not path.exists():
        return path
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem}{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique output beside {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Existing converted swim.yaml; auto-detected when omitted")
    parser.add_argument("--source", help="Original SWIM root; auto-detected under /kaggle/input")
    parser.add_argument("--model", help="Pretrained .pt path; defaults to attached or downloadable yolov8n-obb.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="pilot20_structure_guided_s42")
    parser.add_argument("--diagnostic-images", type=int, default=12)
    parser.add_argument("--install-dependencies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not KAGGLE_WORKING.is_dir():
        raise RuntimeError("This runner is intended for a Kaggle notebook with /kaggle/working available.")
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1].")

    project_root = Path(__file__).resolve().parent
    config = project_root / "configs" / "structure_v2.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"Missing V2 config: {config}")

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    ensure_dependencies(project_root, args.install_dependencies)
    data = resolve_data_yaml(args, project_root)
    model = resolve_model(args.model)

    run_root = KAGGLE_WORKING / "runs" / "wake_ablation"
    train_archive = unique_output(KAGGLE_WORKING / f"{args.name}.zip")
    started_at = time.time()
    train_command = [
        sys.executable,
        str(project_root / "train_structure.py"),
        "--data",
        str(data),
        "--structure-config",
        str(config),
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
        args.name,
        "--archive",
        str(train_archive),
    ]
    run(train_command, cwd=project_root)
    best_weights = newest_run_weights(run_root, started_at)

    diagnostic_archive: Path | None = None
    diagnostic_output: Path | None = None
    if args.diagnostics:
        diagnostic_output = unique_output(KAGGLE_WORKING / f"structure_diagnostics_{args.name}")
        diagnostic_archive = unique_output(diagnostic_output.with_suffix(".zip"))
        run(
            [
                sys.executable,
                str(project_root / "diagnose_structure.py"),
                "--weights",
                str(best_weights),
                "--data",
                str(data),
                "--split",
                "val",
                "--num-images",
                str(args.diagnostic_images),
                "--imgsz",
                str(args.imgsz),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--output",
                str(diagnostic_output),
                "--archive",
                str(diagnostic_archive),
            ],
            cwd=project_root,
        )

    print("\n" + "=" * 72)
    print("V2 experiment completed successfully")
    print(f"Best weights:       {best_weights}")
    print(f"Training archive:   {train_archive}")
    if diagnostic_archive is not None:
        print(f"Diagnostic archive: {diagnostic_archive}")
        print(f"Diagnostic output:  {diagnostic_output}")
    print("Download the ZIP files from the Kaggle Output/Files panel before ending the session.")


if __name__ == "__main__":
    main()
