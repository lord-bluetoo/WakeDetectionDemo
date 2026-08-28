import csv
import json
from zipfile import ZipFile

import cv2
import numpy as np

from wake_structure.artifacts import create_archive
from wake_structure.diagnostics import (
    compute_map_diagnostics,
    find_split_images,
    label_path_for_image,
    run_diagnostics,
    save_structure_figure,
    save_training_curves,
    summarize_rows,
)


def _maps() -> dict[str, np.ndarray]:
    presence = np.full((1, 1, 8, 8), 0.1, dtype=np.float32)
    presence[:, :, 3:5, 2:6] = 0.9
    probabilities = np.zeros((1, 8, 8, 8), dtype=np.float32)
    probabilities[:, 0] = 1.0
    confidence = np.ones((1, 1, 8, 8), dtype=np.float32)
    theta = np.zeros((1, 1, 8, 8), dtype=np.float32)
    return {
        "presence": presence,
        "orientation_distribution": probabilities,
        "confidence": confidence,
        "theta": theta,
        "directional_gate": presence * confidence,
    }


def _box() -> np.ndarray:
    return np.asarray([[0.25, 0.375], [0.75, 0.375], [0.75, 0.625], [0.25, 0.625]], dtype=np.float32)


def test_diagnostics_detect_localized_horizontal_structure() -> None:
    row = compute_map_diagnostics(_maps(), [_box()], "sample.jpg")

    assert row["presence_inside_top10_mean"] > 0.8
    assert row["presence_outside_mean"] < 0.2
    assert row["confidence_inside_mean"] == 1.0
    assert row["orientation_error_deg"] == 0.0
    summary = summarize_rows([row])
    assert not summary["heuristic_flags"]["weak_inside_outside_separation"]
    assert not summary["heuristic_flags"]["low_direction_concentration"]


def test_structure_figure_is_written(tmp_path) -> None:
    output = tmp_path / "figure.png"
    save_structure_figure(np.zeros((64, 64, 3), dtype=np.uint8), [_box()], _maps(), output)
    assert output.is_file()
    assert output.stat().st_size > 0


def test_training_curves_and_archive_are_written(tmp_path) -> None:
    results = tmp_path / "results.csv"
    with results.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "structure_presence_loss", "metrics/mAP50-95(B)"])
        writer.writerow([0, 1.0, 0.1])
        writer.writerow([1, 0.5, 0.2])
    curve = tmp_path / "curves.png"
    assert save_training_curves(results, curve)
    assert curve.is_file()

    run = tmp_path / "run"
    run.mkdir()
    (run / "best.pt").write_bytes(b"weights")
    archive = create_archive([run, curve], tmp_path / "bundle")
    with ZipFile(archive) as zipped:
        assert "run/best.pt" in zipped.namelist()
        assert "curves.png" in zipped.namelist()


def test_dataset_discovery_preserves_images_to_labels_layout(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    image = dataset / "images" / "val" / "00001.jpg"
    label = dataset / "labels" / "val" / "00001.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    label.write_text("", encoding="utf-8")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        f"path: {dataset.as_posix()}\nval: images/val\nnames:\n  0: wake\n",
        encoding="utf-8",
    )

    discovered = find_split_images(yaml_path, "val")

    assert discovered == [image.absolute()]
    assert label_path_for_image(discovered[0]) == label


def test_end_to_end_diagnostics_bundle(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    image = dataset / "images" / "val" / "00001.jpg"
    label = dataset / "labels" / "val" / "00001.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    cv2.imwrite(str(image), np.zeros((64, 64, 3), dtype=np.uint8))
    coordinates = " ".join(str(value) for value in _box().reshape(-1))
    label.write_text(f"0 {coordinates}\n", encoding="utf-8")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        f"path: {dataset.as_posix()}\nval: images/val\nnames:\n  0: wake\n",
        encoding="utf-8",
    )
    run = tmp_path / "run"
    weights = run / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")

    class DummyModel:
        def structure_maps(self, _tensor):
            return _maps()

    monkeypatch.setattr("wake_structure.diagnostics.load_structure_model", lambda *_args: DummyModel())
    output = run_diagnostics(
        weights=weights,
        data_yaml=yaml_path,
        output=tmp_path / "diagnostics",
        num_images=1,
        image_size=64,
        device="cpu",
        archive=tmp_path / "diagnostics_bundle.zip",
    )

    assert (output / "diagnostics.csv").is_file()
    assert (output / "summary.json").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["feature_guidance_enabled"] is False
    assert summary["guidance_alpha"] is None
    assert len(list((output / "figures").glob("*.png"))) == 1
    assert (tmp_path / "diagnostics_bundle.zip").is_file()
