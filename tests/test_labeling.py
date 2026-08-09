from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from automated_image_capture.labeling import (
    CaptureKey,
    CaptureRecord,
    LabelingConfig,
    LabelingError,
    LabelSource,
    MatchedPair,
    _trim_thin_mask_protrusions,
    assess_obb_visibility,
    generate_obb_dataset,
    match_captures,
    scan_capture,
    stabilize_boxes_by_conveyor_position,
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


def test_continuous_angle_and_belt_station_are_part_of_capture_key(tmp_path: Path) -> None:
    foreground = tmp_path / "continuous"
    foreground.mkdir()
    image = np.zeros((20, 30), dtype=np.uint8)
    names = (
        "img_000001_ura-0155_belt-001_pos-0100_out_p1-000_p2-000_auto.png",
        "img_000002_ura-0155_belt-009_pos-0100_back_p1-000_p2-000_auto.png",
    )
    for name in names:
        cv2.imwrite(str(foreground / name), image)

    records = scan_capture(foreground)
    keys = sorted(records, key=lambda key: key.conveyor_station_id)

    assert [key.robot_mode for key in keys] == ["angle", "angle"]
    assert [key.pose_id for key in keys] == [155, 155]
    assert [key.conveyor_station_id for key in keys] == [1, 9]
    assert [key.conveyor_direction for key in keys] == ["out", "back"]
    assert keys[0].view_id != keys[1].view_id


def test_scan_capture_reads_measured_conveyor_position_from_yaml(tmp_path: Path) -> None:
    image_path = (
        tmp_path / "img_000001_ura-0155_belt-001_pos-0100_out_ramp-001_"
        "p1-020_p2-030_auto.png"
    )
    assert cv2.imwrite(str(image_path), np.zeros((20, 30), dtype=np.uint8))
    image_path.with_suffix(".yaml").write_text(
        "conveyor:\n"
        "  nominal_offset_mm: 10.0\n"
        "  measured_logical_offset_mm: 10.73\n"
        '  position_sampled_at: "2026-08-08T16:00:00+02:00"\n',
        encoding="utf-8",
    )

    record = next(iter(scan_capture(tmp_path).values()))

    assert record.nominal_conveyor_position_mm == 10.0
    assert record.measured_conveyor_position_mm == 10.73
    assert record.position_sampled_at == "2026-08-08T16:00:00+02:00"


def test_measured_conveyor_track_stabilizes_bad_single_box(tmp_path: Path) -> None:
    entries: list[tuple[dict[str, object], MatchedPair, np.ndarray]] = []
    bad_index = 5
    for index in range(10):
        path = tmp_path / f"sample_{index}.png"
        assert cv2.imwrite(str(path), np.zeros((240, 400), dtype=np.uint8))
        key = CaptureKey(
            pose_id=155,
            panel_2=0,
            panel_1=0,
            exposure="auto",
            robot_mode="angle",
            conveyor_station_id=index,
            conveyor_position_tenths_mm=index * 100,
            conveyor_direction="out",
        )
        record = CaptureRecord(path, key, index, float(index * 10), float(index * 10))
        pair = MatchedPair(record, record)
        expected_x = 70.0 + index * 12.0
        raw_x = 330.0 if index == bad_index else expected_x
        raw_box = cv2.boxPoints(((raw_x, 120.0), (70.0, 35.0), 12.0)).astype(np.float32)
        entries.append(({"quality": "PASS", "quality_reason": ""}, pair, raw_box))

    boxes, summary = stabilize_boxes_by_conveyor_position(entries)

    corrected_center = boxes[entries[bad_index][1].foreground.path].mean(axis=0)
    assert summary.active
    assert summary.tracked_images == 10
    assert summary.corrected_images >= 1
    assert corrected_center[0] == pytest.approx(70.0 + bad_index * 12.0, abs=5.0)
    assert entries[bad_index][0]["quality"] == "REVIEW"
    assert entries[bad_index][0]["conveyor_track_used"] is True


def test_measured_conveyor_track_interpolates_missing_segmentations(tmp_path: Path) -> None:
    entries: list[tuple[dict[str, object], MatchedPair, np.ndarray | None]] = []
    missing = {0, 4, 9}
    for index in range(10):
        path = tmp_path / f"sample_{index}.png"
        assert cv2.imwrite(str(path), np.zeros((240, 400), dtype=np.uint8))
        key = CaptureKey(
            pose_id=155,
            panel_2=0,
            panel_1=0,
            exposure="auto",
            robot_mode="angle",
            conveyor_station_id=index,
            conveyor_position_tenths_mm=index * 100,
            conveyor_direction="out",
        )
        record = CaptureRecord(path, key, index, float(index * 10), float(index * 10))
        pair = MatchedPair(record, record)
        raw_box = None
        if index not in missing:
            raw_box = cv2.boxPoints(
                ((70.0 + index * 12.0, 120.0), (70.0, 35.0), 12.0)
            ).astype(np.float32)
        entries.append(({"quality": "PASS", "quality_reason": ""}, pair, raw_box))

    boxes, summary = stabilize_boxes_by_conveyor_position(entries)

    assert summary.active
    assert summary.tracked_images == 10
    assert len(boxes) == 10
    for index in missing:
        row, pair, _ = entries[index]
        assert boxes[pair.foreground.path].mean(axis=0)[0] == pytest.approx(
            70.0 + index * 12.0, abs=5.0
        )
        assert row["quality"] == "REVIEW"
        assert "interpoliert" in str(row["quality_reason"])


def test_measured_conveyor_track_uses_straight_anchor_line_among_blob_outliers(
    tmp_path: Path,
) -> None:
    entries: list[tuple[dict[str, object], MatchedPair, np.ndarray]] = []
    true_indices = {0, 2, 4, 7, 10, 12, 14}
    outlier_centers = (
        (315.0, 30.0),
        (35.0, 205.0),
        (280.0, 190.0),
        (60.0, 35.0),
        (350.0, 150.0),
        (180.0, 30.0),
        (40.0, 150.0),
        (330.0, 220.0),
    )
    outlier_index = 0
    for index in range(15):
        path = tmp_path / f"robust_{index}.png"
        assert cv2.imwrite(str(path), np.zeros((240, 400), dtype=np.uint8))
        key = CaptureKey(
            pose_id=155,
            panel_2=0,
            panel_1=0,
            exposure="auto",
            robot_mode="angle",
            conveyor_station_id=index,
            conveyor_position_tenths_mm=index * 100,
            conveyor_direction="out",
        )
        record = CaptureRecord(path, key, index, float(index * 10), float(index * 10))
        if index in true_indices:
            center = (65.0 + index * 13.0, 105.0 + index * 1.5)
            size = (72.0, 38.0)
        else:
            center = outlier_centers[outlier_index]
            outlier_index += 1
            size = (145.0, 18.0)
        raw_box = cv2.boxPoints((center, size, 14.0)).astype(np.float32)
        entries.append(
            ({"quality": "PASS", "quality_reason": ""}, MatchedPair(record, record), raw_box)
        )

    boxes, summary = stabilize_boxes_by_conveyor_position(entries)

    centers = np.asarray([boxes[pair.foreground.path].mean(axis=0) for _, pair, _ in entries])
    expected_x = 65.0 + np.arange(15) * 13.0
    expected_y = 105.0 + np.arange(15) * 1.5
    assert summary.active
    assert np.max(np.abs(centers[:, 0] - expected_x)) < 3.0
    assert np.max(np.abs(centers[:, 1] - expected_y)) < 3.0
    assert sum(bool(row["track_anchor"]) for row, _, _ in entries) == len(true_indices)


def test_measured_conveyor_track_allows_nonlinear_progress_on_straight_image_line(
    tmp_path: Path,
) -> None:
    entries: list[tuple[dict[str, object], MatchedPair, np.ndarray]] = []
    expected: list[tuple[float, float]] = []
    for index in range(12):
        path = tmp_path / f"nonlinear_{index}.png"
        assert cv2.imwrite(str(path), np.zeros((400, 900), dtype=np.uint8))
        position = float(index * 10)
        center_x = 70.0 + 0.04 * position**2
        center_y = 110.0 + 0.15 * center_x
        expected.append((center_x, center_y))
        key = CaptureKey(
            pose_id=210,
            panel_2=0,
            panel_1=0,
            exposure="auto",
            robot_mode="angle",
            conveyor_station_id=index,
            conveyor_position_tenths_mm=index * 100,
            conveyor_direction="out",
        )
        record = CaptureRecord(path, key, index, position, position)
        raw_box = cv2.boxPoints(((center_x, center_y), (72.0, 38.0), 14.0)).astype(
            np.float32
        )
        entries.append(
            ({"quality": "PASS", "quality_reason": ""}, MatchedPair(record, record), raw_box)
        )

    boxes, summary = stabilize_boxes_by_conveyor_position(entries)

    predicted = np.asarray(
        [boxes[pair.foreground.path].mean(axis=0) for _, pair, _ in entries]
    )
    assert summary.active
    assert np.max(np.linalg.norm(predicted - np.asarray(expected), axis=1)) < 3.0
    direction = predicted[-1] - predicted[0]
    offsets = predicted - predicted[0]
    perpendicular = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
    assert np.max(perpendicular / np.linalg.norm(direction)) < 0.5


def test_ramp_and_grid_profiles_report_concrete_pairing_error(tmp_path: Path) -> None:
    foreground = tmp_path / "parts_ramp"
    background = tmp_path / "empty_grid"
    foreground.mkdir()
    background.mkdir()
    _write_capture(foreground, 155, 1, 0, 0, 0)
    _write_capture(background, 155, 1, 0, 0)

    with pytest.raises(LabelingError, match="Raster- und Rampenserie"):
        match_captures(scan_capture(foreground), scan_capture(background))


def test_visibility_assessment_distinguishes_black_borderline_and_visible_images() -> None:
    box = cv2.boxPoints(((160.0, 120.0), (82.0, 48.0), 20.0)).astype(np.float32)
    black = np.zeros((240, 320), dtype=np.uint8)
    borderline = np.full((240, 320), 14, dtype=np.uint8)
    cv2.fillConvexPoly(borderline, np.rint(box).astype(np.int32), 2)
    visible = np.full((240, 320), 180, dtype=np.uint8)
    cv2.fillConvexPoly(visible, np.rint(box).astype(np.int32), 30)

    black_result = assess_obb_visibility(black, box)
    borderline_result = assess_obb_visibility(borderline, box)
    visible_result = assess_obb_visibility(visible, box)

    assert black_result.suspicious and black_result.recommended_exclude
    assert borderline_result.suspicious and not borderline_result.recommended_exclude
    assert not visible_result.suspicious
    assert visible_result.score > borderline_result.score > black_result.score


def test_thin_mask_protrusion_is_trimmed_but_compact_component_is_unchanged() -> None:
    mask = np.zeros((240, 400), dtype=np.uint8)
    cv2.rectangle(mask, (130, 80), (250, 170), 255, -1)
    cv2.line(mask, (20, 125), (380, 125), 255, 1)
    compact = np.zeros_like(mask)
    cv2.rectangle(compact, (130, 80), (250, 170), 255, -1)

    trimmed = _trim_thin_mask_protrusions(mask)
    untouched = _trim_thin_mask_protrusions(compact)

    original_width = max(cv2.minAreaRect(cv2.findNonZero(mask))[1])
    trimmed_size = cv2.minAreaRect(cv2.findNonZero(trimmed))[1]
    assert max(trimmed_size) < original_width * 0.5
    assert np.count_nonzero(trimmed) > np.count_nonzero(compact) * 0.95
    assert np.array_equal(untouched, compact)


def test_visibility_review_can_exclude_positive_but_keeps_audit_row(tmp_path: Path) -> None:
    foreground, background = _make_capture_pair(tmp_path)
    black_path = sorted(foreground.glob("*.png"))[0]
    assert cv2.imwrite(str(black_path), np.zeros((240, 320), dtype=np.uint8))
    output = tmp_path / "filtered_dataset"
    reviewed: list[Path] = []

    def exclude_recommended(items):
        reviewed.extend(item.source_path for item in items)
        return frozenset(item.source_path for item in items if item.recommended_exclude)

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
        ),
        visibility_review=exclude_recommended,
    )

    report = (output / "label_report.csv").read_text(encoding="utf-8-sig")
    assert black_path in reviewed
    assert result.positive_images == 5
    assert result.excluded_images == 1
    assert "True" in report
    assert black_path.name in report
    assert not any(
        path.name.startswith("class_") and black_path.name in path.name
        for path in (output / "images").rglob("*.png")
    )


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
