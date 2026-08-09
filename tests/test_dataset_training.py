from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from automated_image_capture.dataset import (
    ALL_POSES,
    DatasetBuildConfig,
    DatasetError,
    build_curated_dataset,
    collect_dataset_records,
    prepare_resized_training_dataset,
    render_record_preview,
    save_curation,
    verify_curated_dataset,
)
from automated_image_capture.training import TrainingConfig, TrainingError


def _write_labeled_source(root: Path) -> None:
    image_dir = root / "images" / "old"
    label_dir = root / "labels" / "old"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for class_id, class_name in ((0, "Pose 1"), (1, "Pose 2")):
        index = 1
        for pose in sorted(ALL_POSES):
            for panel_1 in (0, 50):
                raw_name = f"img_{index:06d}_ur{pose}_p1-{panel_1:03d}_p2-000_auto.png"
                positive_name = f"class_{class_id:03d}_{raw_name}"
                image = np.full((160, 240), 30 + panel_1, dtype=np.uint8)
                cv2.rectangle(image, (60, 40), (170, 120), 180, -1)
                assert cv2.imwrite(str(image_dir / positive_name), image)
                positive_label = f"{Path(positive_name).stem}.txt"
                (label_dir / positive_label).write_text(
                    f"{class_id} 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
                    encoding="ascii",
                )
                rows.append(
                    {
                        "pose_id": pose,
                        "panel_1": panel_1,
                        "panel_2": 0,
                        "exposure": "auto",
                        "foreground_file": raw_name,
                        "background_file": raw_name,
                        "consensus_iou": 0.2 if panel_1 == 0 else 0.9,
                        "quality": "REVIEW" if panel_1 == 0 else "PASS",
                        "conveyor_station_id": 7,
                        "conveyor_direction": "out",
                        "conveyor_position_mm": 20.0,
                        "conveyor_nominal_metadata_position_mm": 20.0,
                        "conveyor_measured_position_mm": 20.7,
                        "ramp_sample_id": 7,
                        "conveyor_track_used": True,
                        "track_correction_applied": panel_1 == 0,
                        "class_id": class_id,
                        "class_name": class_name,
                        "dataset_image": positive_name,
                        "label_file": positive_label,
                    }
                )
                index += 1
    for pose_index, pose in enumerate(sorted(ALL_POSES), 1):
        for panel_1 in (0, 50):
            raw_index = (pose_index - 1) * 2 + (1 if panel_1 == 0 else 2)
            raw_name = f"img_{raw_index:06d}_ur{pose}_p1-{panel_1:03d}_p2-000_auto.png"
            background_name = f"empty_{raw_name}"
            image = np.full((160, 240), 30 + panel_1, dtype=np.uint8)
            assert cv2.imwrite(str(image_dir / background_name), np.full_like(image, 50))
            background_label = f"{Path(background_name).stem}.txt"
            (label_dir / background_label).write_text("", encoding="ascii")
    with (root / "label_report.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "label_summary.json").write_text(
        json.dumps({"classes": {"0": {"name": "Pose 1"}, "1": {"name": "Pose 2"}}}),
        encoding="utf-8",
    )
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/old\nval: images/old\nnames:\n  0: Pose 1\n  1: Pose 2\n",
        encoding="utf-8",
    )


@pytest.fixture
def source_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "obb"
    _write_labeled_source(source)
    return source


def _config(tmp_path: Path, source: Path) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        source_dataset=source,
        output_root=tmp_path / "combined",
        curation_path=tmp_path / "combined" / "curation.json",
        version_name="test_dataset",
    )


def test_collects_two_classes_and_deduplicates_empty_images(
    tmp_path: Path, source_dataset: Path
) -> None:
    records = collect_dataset_records(_config(tmp_path, source_dataset))

    assert len(records) == 90
    assert sum(record.class_id == 0 for record in records) == 30
    assert sum(record.class_id == 1 for record in records) == 30
    assert sum(record.kind == "empty" for record in records) == 30
    assert len({record.target_name for record in records}) == 90
    assert sum(record.split == "train" for record in records) == 54
    assert sum(record.split == "val" for record in records) == 18
    assert sum(record.split == "test" for record in records) == 18
    positive = next(record for record in records if record.kind == "positive")
    assert positive.conveyor_station_id == 7
    assert positive.conveyor_direction == "out"
    assert positive.conveyor_nominal_position_mm == 20.0
    assert positive.conveyor_measured_position_mm == 20.7
    assert positive.ramp_sample_id == 7
    assert positive.conveyor_track_used


def test_collect_skips_audited_exclusions_and_their_missing_empty_image(
    tmp_path: Path, source_dataset: Path
) -> None:
    report_path = source_dataset / "label_report.csv"
    with report_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    background_file = rows[0]["background_file"]
    for row in rows:
        row["excluded_from_dataset"] = str(row["background_file"] == background_file)
        if row["background_file"] == background_file:
            (source_dataset / "images" / "old" / row["dataset_image"]).unlink()
            (source_dataset / "labels" / "old" / row["label_file"]).unlink()
            row["dataset_image"] = ""
            row["label_file"] = ""
    empty_name = f"empty_{background_file}"
    (source_dataset / "images" / "old" / empty_name).unlink()
    (source_dataset / "labels" / "old" / f"{Path(empty_name).stem}.txt").unlink()
    with report_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    records = collect_dataset_records(_config(tmp_path, source_dataset))

    assert len(records) == 87
    assert all(record.source_image.name != empty_name for record in records)


def test_build_keeps_classes_and_pose_splits_disjoint(
    tmp_path: Path, source_dataset: Path
) -> None:
    result = build_curated_dataset(_config(tmp_path, source_dataset))
    integrity = verify_curated_dataset(result.dataset_directory)

    assert integrity.valid
    assert integrity.image_count == integrity.label_count == 90
    assert integrity.class_counts == {"Pose 1": 30, "Pose 2": 30, "Leere Rutsche": 30}
    pose1_labels = list((result.dataset_directory / "labels").rglob("class_000_*.txt"))
    pose2_labels = list((result.dataset_directory / "labels").rglob("class_001_*.txt"))
    assert all(path.read_text(encoding="ascii").startswith("0 ") for path in pose1_labels)
    assert all(path.read_text(encoding="ascii").startswith("1 ") for path in pose2_labels)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    by_pose: dict[int, set[str]] = {}
    for record in manifest["records"]:
        by_pose.setdefault(record["pose_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in by_pose.values())


def test_curation_excludes_only_selected_record(
    tmp_path: Path, source_dataset: Path
) -> None:
    config = _config(tmp_path, source_dataset)
    records = collect_dataset_records(config)
    excluded = records[0]
    save_curation(
        config.curation_path,
        [excluded.record_id],
        config.source_dataset,
    )  # type: ignore[arg-type]

    result = build_curated_dataset(config)

    assert result.included_images == 89
    assert result.excluded_images == 1
    assert not list((result.dataset_directory / "images").rglob(excluded.target_name))


def test_hardlink_failure_falls_back_to_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_dataset: Path,
) -> None:
    config = _config(tmp_path, source_dataset)

    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError("different volume")

    monkeypatch.setattr(os, "link", fail_link)
    result = build_curated_dataset(config)
    assert result.included_images == 90


def test_preview_contains_rgb_overlay(tmp_path: Path, source_dataset: Path) -> None:
    record = collect_dataset_records(_config(tmp_path, source_dataset))[0]
    preview = render_record_preview(record)
    assert preview.shape == (160, 240, 3)
    assert np.any(preview[:, :, 1] > preview[:, :, 0])


def test_unknown_pose_is_rejected(tmp_path: Path, source_dataset: Path) -> None:
    report = source_dataset / "label_report.csv"
    text = report.read_text(encoding="utf-8").replace("155,", "999,", 1)
    report.write_text(text, encoding="utf-8")

    with pytest.raises(DatasetError, match="keinem Split"):
        collect_dataset_records(_config(tmp_path, source_dataset))


def test_explicit_source_split_accepts_new_continuous_angles(
    tmp_path: Path, source_dataset: Path
) -> None:
    (source_dataset / "images" / "old").rename(source_dataset / "images" / "train")
    (source_dataset / "labels" / "old").rename(source_dataset / "labels" / "train")
    report = source_dataset / "label_report.csv"
    text = report.read_text(encoding="utf-8").replace("155,", "999,", 1)
    report.write_text(text, encoding="utf-8")

    records = collect_dataset_records(_config(tmp_path, source_dataset))
    new_angle = next(record for record in records if record.pose_id == 999)

    assert new_angle.split == "train"


def test_missing_test_split_holds_out_a_complete_pose(
    tmp_path: Path, source_dataset: Path
) -> None:
    image_train = source_dataset / "images" / "train"
    label_train = source_dataset / "labels" / "train"
    (source_dataset / "images" / "old").rename(image_train)
    (source_dataset / "labels" / "old").rename(label_train)
    image_val = source_dataset / "images" / "val"
    label_val = source_dataset / "labels" / "val"
    image_val.mkdir()
    label_val.mkdir()
    for image in image_train.glob("*ur170_*.png"):
        image.rename(image_val / image.name)
    for label in label_train.glob("*ur170_*.txt"):
        label.rename(label_val / label.name)

    records = collect_dataset_records(_config(tmp_path, source_dataset))
    pose_splits: dict[int, set[str]] = {}
    for record in records:
        pose_splits.setdefault(record.pose_id, set()).add(record.split)

    assert pose_splits[170] == {"val"}
    assert pose_splits[2200] == {"test"}
    assert all(len(splits) == 1 for splits in pose_splits.values())


def test_training_config_requires_verified_dataset(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="data.yaml"):
        TrainingConfig(tmp_path / "missing", tmp_path / "runs").validated()


def test_resized_training_dataset_preserves_labels_and_is_reused(
    tmp_path: Path, source_dataset: Path
) -> None:
    dataset = build_curated_dataset(_config(tmp_path, source_dataset)).dataset_directory

    first = prepare_resized_training_dataset(dataset, 128)
    second = prepare_resized_training_dataset(dataset, 128)
    integrity = verify_curated_dataset(first.dataset_directory)
    sample = cv2.imread(
        str(next((first.dataset_directory / "images" / "train").glob("*.png"))),
        cv2.IMREAD_UNCHANGED,
    )

    assert integrity.valid
    assert integrity.image_count == 90
    assert sample is not None and max(sample.shape[:2]) == 128
    assert not first.reused
    assert second.reused
