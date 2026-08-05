from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, Qt

from automated_image_capture.dataset import DatasetRecord
from automated_image_capture.settings import SettingsStore
from automated_image_capture.ui.training_dialog import TrainingDialog


def _source_root(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "label_report.csv").write_text("unused\n", encoding="utf-8")


def _record(tmp_path: Path, record_id: str, quality: str) -> DatasetRecord:
    image = tmp_path / f"{record_id}.png"
    label = tmp_path / f"{record_id}.txt"
    assert cv2.imwrite(str(image), np.full((60, 90), 80, dtype=np.uint8))
    label.write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n", encoding="ascii")
    return DatasetRecord(
        record_id=record_id,
        kind="positive",
        class_id=0,
        class_name="Pose 1",
        pose_id=155,
        panel_1=0,
        panel_2=0,
        exposure="auto",
        quality=quality,
        consensus_iou=0.2 if quality == "REVIEW" else 0.9,
        source_image=image,
        source_label=label,
        target_name=f"pose1_{record_id}.png",
        split="train",
    )


def test_review_can_exclude_a_single_image(qtbot, monkeypatch, tmp_path: Path) -> None:
    pose1 = tmp_path / "pose1"
    pose2 = tmp_path / "pose2"
    _source_root(pose1)
    _source_root(pose2)
    backend = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    dialog = TrainingDialog(SettingsStore(backend))
    qtbot.addWidget(dialog)
    dialog.pose1_path.setText(str(pose1))
    dialog.pose2_path.setText(str(pose2))
    dialog.output_path.setText(str(tmp_path / "out"))
    records = [_record(tmp_path, "pass", "PASS"), _record(tmp_path, "review", "REVIEW")]
    monkeypatch.setattr(
        "automated_image_capture.ui.training_dialog.collect_dataset_records",
        lambda _config: records,
    )

    dialog.load_records()

    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 1).text() == "REVIEW"
    dialog.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert "review" in dialog._excluded_ids
    assert "1 ausgeschlossen" in dialog.review_summary.text()


def test_training_events_update_progress_and_result_link(qtbot, tmp_path: Path) -> None:
    backend = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    dialog = TrainingDialog(SettingsStore(backend))
    qtbot.addWidget(dialog)

    dialog._handle_training_event({"event": "epoch", "epoch": 7, "total": 20})
    assert dialog.progress.value() == 7
    assert "7/20" in dialog.status.text()

    result = tmp_path / "run"
    dialog._handle_training_event(
        {
            "event": "completed",
            "run_directory": str(result),
            "empty_false_positive_rate": 0.125,
        }
    )
    assert dialog._result_directory == result
    assert dialog.open_result_button.isEnabled()
    assert "12.5 %" in dialog.status.text()
