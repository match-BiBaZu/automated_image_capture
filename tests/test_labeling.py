from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from automated_image_capture.labeling import (
    LabelingConfig,
    LabelingError,
    LabelSource,
    generate_obb_dataset,
    match_captures,
    scan_capture,
)


def _write_capture(
    directory: Path,
    pose: int,
    index: int,
    p1: int,
    p2: int,
    ramp_sample_id: int | None = None,
) -> None:
    height, width = 240, 320
    yy, xx = np.indices((height, width))
    background = (100 + ((xx // 12 + yy // 12) % 2) * 35 + (p1 + p2) // 8).astype(np.uint8)
    foreground = background.copy()
    center = (150 + (pose == 200) * 35, 120)
    rect = (center, (82, 48), 28 if pose == 155 else -22)
    cv2.fillConvexPoly(foreground, np.rint(cv2.boxPoints(rect)).astype(np.int32), 25)
    ramp = "" if ramp_sample_id is None else f"ramp-{ramp_sample_id:03d}_"
    name = f"img_{index:06d}_ur{pose}_{ramp}p1-{p1:03d}_p2-{p2:03d}_auto.png"
    target = directory / name
    image = foreground if directory.name.startswith("parts") else background
    assert cv2.imwrite(str(target), image)


def _make_capture_pair(tmp_path: Path) -> tuple[Path, Path]:
    foreground = tmp_path / "parts"
    background = tmp_path / "empty"
    foreground.mkdir()
    background.mkdir()
    index = 1
    for pose in (155, 200):
        for p1, p2 in ((0, 0), (50, 0), (50, 50)):
            _write_capture(foreground, pose, index, p1, p2)
            _write_capture(background, pose, index, p1, p2)
            index += 1
    return foreground, background


def test_capture_pairing_reports_missing_background(tmp_path: Path) -> None:
    foreground, background = _make_capture_pair(tmp_path)
    next(background.glob("*.png")).unlink()

    with pytest.raises(LabelingError, match="Leerbilder fehlen"):
        match_captures(scan_capture(foreground), scan_capture(background))


def test_ramp_pairing_uses_sample_id_even_when_brightness_repeats(tmp_path: Path) -> None:
    foreground = tmp_path / "parts_ramp"
    background = tmp_path / "empty_ramp"
    foreground.mkdir()
    background.mkdir()
    for sample_id in (0, 1):
        _write_capture(foreground, 155, sample_id + 1, 0, 0, sample_id)
        _write_capture(background, 155, sample_id + 1, 0, 0, sample_id)

    pairs = match_captures(scan_capture(foreground), scan_capture(background))

    assert [pair.foreground.key.ramp_sample_id for pair in pairs] == [0, 1]


def test_ramp_and_grid_profiles_report_concrete_pairing_error(tmp_path: Path) -> None:
    foreground = tmp_path / "parts_ramp"
    background = tmp_path / "empty_grid"
    foreground.mkdir()
    background.mkdir()
    _write_capture(foreground, 155, 1, 0, 0, 0)
    _write_capture(background, 155, 1, 0, 0)

    with pytest.raises(LabelingError, match="Raster- und Rampenserie"):
        match_captures(scan_capture(foreground), scan_capture(background))


def test_generate_yolo_obb_dataset_with_negative_images(tmp_path: Path) -> None:
    foreground, background = _make_capture_pair(tmp_path)
    output = tmp_path / "dataset"

    result = generate_obb_dataset(
        LabelingConfig(
            (
                LabelSource("Pose 1", foreground),
                LabelSource("Leere Rutsche", background, is_empty=True),
            ),
            output,
            validation_fraction=0.5,
            minimum_difference=20,
            consensus_fraction=0.5,
        )
    )

    assert result.positive_images == 6
    assert result.negative_images == 6
    assert result.poses == 2
    assert len(list((output / "images").rglob("*.png"))) == 12
    labels = list((output / "labels").rglob("class_*.txt"))
    negative_labels = list((output / "labels").rglob("empty_*.txt"))
    assert len(labels) == 6
    assert len(negative_labels) == 6
    assert all(len(path.read_text(encoding="ascii").split()) == 9 for path in labels)
    assert all(not path.read_text(encoding="ascii") for path in negative_labels)
    assert "train: images/train" in (output / "data.yaml").read_text(encoding="utf-8")
    assert (output / "label_report.csv").is_file()
    assert len(list((output / "review").glob("class_*_ur_*_obb.jpg"))) == 2


def test_generate_multiple_pose_classes_and_deduplicate_empty_images(tmp_path: Path) -> None:
    pose1, background = _make_capture_pair(tmp_path)
    pose2 = tmp_path / "parts_pose2"
    pose2.mkdir()
    for record in scan_capture(pose1).values():
        _write_capture(
            pose2,
            record.key.pose_id,
            record.sequence_index,
            record.key.panel_1,
            record.key.panel_2,
        )
    output = tmp_path / "multi_dataset"

    result = generate_obb_dataset(
        LabelingConfig(
            (
                LabelSource("Pose 1", pose1),
                LabelSource("Pose 2", pose2),
                LabelSource("Leere Rutsche", background, is_empty=True),
            ),
            output,
            validation_fraction=0.5,
            minimum_difference=20,
            consensus_fraction=0.5,
        )
    )

    assert result.classes == 2
    assert result.positive_images == 12
    assert result.negative_images == 6
    assert len(list((output / "images").rglob("*.png"))) == 18
    positive_labels = list((output / "labels").rglob("class_*.txt"))
    assert {path.read_text(encoding="ascii").split()[0] for path in positive_labels} == {
        "0",
        "1",
    }
    yaml_text = (output / "data.yaml").read_text(encoding="utf-8")
    assert '0: "Pose 1"' in yaml_text
    assert '1: "Pose 2"' in yaml_text
