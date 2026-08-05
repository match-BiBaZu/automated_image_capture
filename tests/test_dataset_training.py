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


def _write_labeled_source(root: Path, old_class: int) -> None:
    image_dir = root / "images" / "old"
    label_dir = root / "labels" / "old"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    index = 1
    for pose in sorted(ALL_POSES):
        for panel_1 in (0, 50):
            raw_name = f"img_{index:06d}_ur{pose}_p1-{panel_1:03d}_p2-000_auto.png"
            positive_name = f"Kk1_{raw_name}"
            background_name = f"background_{raw_name}"
            image = np.full((160, 240), 30 + panel_1, dtype=np.uint8)
            cv2.rectangle(image, (60, 40), (170, 120), 180, -1)
            assert cv2.imwrite(str(image_dir / positive_name), image)
            assert cv2.imwrite(str(image_dir / background_name), np.full_like(image, 50))
            positive_label = f"{Path(positive_name).stem}.txt"
            background_label = f"{Path(background_name).stem}.txt"
            (label_dir / positive_label).write_text(
                f"{old_class} 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
                encoding="ascii",
            )
            (label_dir / background_label).write_text("", encoding="ascii")
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
                    "dataset_image": positive_name,
                    "label_file": positive_label,
                }
            )
            index += 1
    with (root / "label_report.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def source_datasets(tmp_path: Path) -> tuple[Path, Path]:
    pose1 = tmp_path / "pose1"
    pose2 = tmp_path / "pose2"
    _write_labeled_source(pose1, 7)
    _write_labeled_source(pose2, 9)
    return pose1, pose2


def _config(tmp_path: Path, sources: tuple[Path, Path]) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        pose1_dataset=sources[0],
        pose2_dataset=sources[1],
        output_root=tmp_path / "combined",
        curation_path=tmp_path / "combined" / "curation.json",
        version_name="test_dataset",
    )


def test_collects_two_classes_and_deduplicates_empty_images(
    tmp_path: Path, source_datasets: tuple[Path, Path]
) -> None:
    records = collect_dataset_records(_config(tmp_path, source_datasets))

    assert len(records) == 90
    assert sum(record.class_id == 0 for record in records) == 30
    assert sum(record.class_id == 1 for record in records) == 30
    assert sum(record.kind == "empty" for record in records) == 30
    assert len({record.target_name for record in records}) == 90
    assert sum(record.split == "train" for record in records) == 54
    assert sum(record.split == "val" for record in records) == 18
    assert sum(record.split == "test" for record in records) == 18


def test_build_rewrites_classes_and_keeps_pose_splits_disjoint(
    tmp_path: Path, source_datasets: tuple[Path, Path]
) -> None:
    result = build_curated_dataset(_config(tmp_path, source_datasets))
    integrity = verify_curated_dataset(result.dataset_directory)

    assert integrity.valid
    assert integrity.image_count == integrity.label_count == 90
    assert integrity.class_counts == {"Pose 1": 30, "Pose 2": 30, "Leere Rutsche": 30}
    pose1_labels = list((result.dataset_directory / "labels").rglob("pose1_*.txt"))
    pose2_labels = list((result.dataset_directory / "labels").rglob("pose2_*.txt"))
    assert all(path.read_text(encoding="ascii").startswith("0 ") for path in pose1_labels)
    assert all(path.read_text(encoding="ascii").startswith("1 ") for path in pose2_labels)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    by_pose: dict[int, set[str]] = {}
    for record in manifest["records"]:
        by_pose.setdefault(record["pose_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in by_pose.values())


def test_curation_excludes_only_selected_record(
    tmp_path: Path, source_datasets: tuple[Path, Path]
) -> None:
    config = _config(tmp_path, source_datasets)
    records = collect_dataset_records(config)
    excluded = records[0]
    save_curation(config.curation_path, [excluded.record_id])  # type: ignore[arg-type]

    result = build_curated_dataset(config)

    assert result.included_images == 89
    assert result.excluded_images == 1
    assert not list((result.dataset_directory / "images").rglob(excluded.target_name))


def test_hardlink_failure_falls_back_to_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_datasets: tuple[Path, Path],
) -> None:
    config = _config(tmp_path, source_datasets)

    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError("different volume")

    monkeypatch.setattr(os, "link", fail_link)
    result = build_curated_dataset(config)
    assert result.included_images == 90


def test_preview_contains_rgb_overlay(tmp_path: Path, source_datasets: tuple[Path, Path]) -> None:
    record = collect_dataset_records(_config(tmp_path, source_datasets))[0]
    preview = render_record_preview(record)
    assert preview.shape == (160, 240, 3)
    assert np.any(preview[:, :, 1] > preview[:, :, 0])


def test_unknown_pose_is_rejected(tmp_path: Path, source_datasets: tuple[Path, Path]) -> None:
    report = source_datasets[0] / "label_report.csv"
    text = report.read_text(encoding="utf-8").replace("155,", "999,", 1)
    report.write_text(text, encoding="utf-8")

    with pytest.raises(DatasetError, match="keinem Split"):
        collect_dataset_records(_config(tmp_path, source_datasets))


def test_training_config_requires_verified_dataset(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="data.yaml"):
        TrainingConfig(tmp_path / "missing", tmp_path / "runs").validated()


def test_resized_training_dataset_preserves_labels_and_is_reused(
    tmp_path: Path, source_datasets: tuple[Path, Path]
) -> None:
    dataset = build_curated_dataset(_config(tmp_path, source_datasets)).dataset_directory

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
