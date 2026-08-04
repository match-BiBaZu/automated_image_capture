from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from automated_image_capture.labeling import (
    LabelingConfig,
    LabelingError,
    generate_obb_dataset,
    match_captures,
    scan_capture,
)


def _write_capture(directory: Path, pose: int, index: int, p1: int, p2: int) -> None:
    height, width = 240, 320
    yy, xx = np.indices((height, width))
    background = (100 + ((xx // 12 + yy // 12) % 2) * 35 + (p1 + p2) // 8).astype(
        np.uint8
    )
    foreground = background.copy()
    center = (150 + (pose == 200) * 35, 120)
    rect = (center, (82, 48), 28 if pose == 155 else -22)
    cv2.fillConvexPoly(foreground, np.rint(cv2.boxPoints(rect)).astype(np.int32), 25)
    name = f"img_{index:06d}_ur{pose}_p1-{p1:03d}_p2-{p2:03d}_auto.png"
    target = directory / name
    image = foreground if directory.name == "parts" else background
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


def test_generate_yolo_obb_dataset_with_negative_images(tmp_path: Path) -> None:
    foreground, background = _make_capture_pair(tmp_path)
    output = tmp_path / "dataset"

    result = generate_obb_dataset(
        LabelingConfig(
            foreground,
            background,
            output,
            class_name="Kk1",
            validation_fraction=0.5,
            minimum_difference=20,
            consensus_fraction=0.5,
        )
    )

    assert result.positive_images == 6
    assert result.negative_images == 6
    assert result.poses == 2
    assert len(list((output / "images").rglob("*.png"))) == 12
    labels = list((output / "labels").rglob("Kk1_*.txt"))
    negative_labels = list((output / "labels").rglob("background_*.txt"))
    assert len(labels) == 6
    assert len(negative_labels) == 6
    assert all(len(path.read_text(encoding="ascii").split()) == 9 for path in labels)
    assert all(not path.read_text(encoding="ascii") for path in negative_labels)
    assert "train: images/train" in (output / "data.yaml").read_text(encoding="utf-8")
    assert (output / "label_report.csv").is_file()
    assert len(list((output / "review").glob("pose_*_obb.jpg"))) == 2
