import os

from kaggle_run_geometry import _version_tuple, newest_run_weights, unique_output


def test_version_tuple_accepts_package_suffix() -> None:
    assert _version_tuple("8.3.12+cpu") == (8, 3, 12)
    assert _version_tuple("") == ()


def test_unique_output_preserves_existing_result(tmp_path) -> None:
    archive = tmp_path / "result.zip"
    archive.write_bytes(b"old")
    assert unique_output(archive) == tmp_path / "result2.zip"


def test_newest_run_weights_only_selects_current_run(tmp_path) -> None:
    old = tmp_path / "old" / "weights" / "best.pt"
    new = tmp_path / "new" / "weights" / "best.pt"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    old.touch()
    started_at = old.stat().st_mtime + 10
    new.touch()
    # Set the new checkpoint into the current-run time window without sleeping.
    timestamp = started_at + 1
    os.utime(new, (timestamp, timestamp))
    assert newest_run_weights(tmp_path, started_at) == new
