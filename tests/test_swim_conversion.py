import math
from pathlib import Path

from wake_structure.swim import convert_annotation, convert_swim_dataset, rotated_box_corners


def _write_xml(path: Path, *, angle: float = 0.0) -> None:
    path.write_text(
        f"""<annotation>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <object>
    <name>wake</name><difficult>0</difficult>
    <robndbox><cx>50</cx><cy>40</cy><w>40</w><h>20</h><angle>{angle}</angle></robndbox>
  </object>
</annotation>""",
        encoding="utf-8",
    )


def test_rotated_box_corners_at_zero_angle() -> None:
    assert rotated_box_corners(50, 40, 40, 20, 0) == [
        (30.0, 30.0),
        (70.0, 30.0),
        (70.0, 50.0),
        (30.0, 50.0),
    ]


def test_annotation_is_normalized_and_ordered(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    _write_xml(xml, angle=math.pi / 2)
    records, stats = convert_annotation(xml)
    assert stats.boxes == 1
    assert stats.clipped_boxes == 0
    assert len(records[0]) == 8
    assert all(0 <= value <= 1 for value in records[0])


def test_complete_split_conversion(tmp_path: Path) -> None:
    source, output = tmp_path / "SWIM", tmp_path / "converted"
    for directory in ("JPEGImages", "Annotations", "ImageSets"):
        (source / directory).mkdir(parents=True)
    for index, split in enumerate(("train", "val", "test"), start=1):
        identifier = f"{index:05d}"
        (source / "JPEGImages" / f"{identifier}.jpg").write_bytes(b"image")
        _write_xml(source / "Annotations" / f"{identifier}.xml")
        (source / "ImageSets" / f"{split}.txt").write_text(identifier + "\n", encoding="utf-8")

    results = convert_swim_dataset(source, output, image_mode="copy")
    assert {split: stats.images for split, stats in results.items()} == {"train": 1, "val": 1, "test": 1}
    assert (output / "swim.yaml").is_file()
    assert (output / "labels" / "train" / "00001.txt").read_text().startswith("0 ")
    assert (output / "images" / "test" / "00003.jpg").is_file()

