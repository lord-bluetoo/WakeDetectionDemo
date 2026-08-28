"""Utilities for preserving experiment outputs outside an ephemeral notebook session."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def create_archive(sources: list[str | Path], output: str | Path) -> Path:
    """Archive files or directories under one top-level entry per source."""

    resolved_sources = [Path(source).resolve() for source in sources]
    missing = [source for source in resolved_sources if not source.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot archive missing path(s): {missing}")

    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".zip":
        output_path = output_path.with_suffix(".zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        for source in resolved_sources:
            if source.is_file():
                archive.write(source, arcname=source.name)
                continue
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.resolve() != output_path:
                    archive.write(path, arcname=(Path(source.name) / path.relative_to(source)).as_posix())
    return output_path

