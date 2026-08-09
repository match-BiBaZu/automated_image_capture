from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, Qt

from automated_image_capture.labeling import (
    LabelingConfig,
    LabelSource,
    VisibilityReviewItem,
)
from automated_image_capture.settings import SettingsStore
from automated_image_capture.ui.labeling_dialog import LabelingDialog, VisibilityReviewDialog


def _dialog(qtbot, tmp_path: Path) -> LabelingDialog:
    directories = [tmp_path / name for name in ("pose1", "pose2", "empty")]
    for directory in directories:
        directory.mkdir()
    backend = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    store = SettingsStore(backend)
    store.save_labeling(
        LabelingConfig(
            (
                LabelSource("Pose 1", directories[0]),
                LabelSource("Pose 2", directories[1]),
                LabelSource("Leere Rutsche", directories[2], is_empty=True),
            ),
            tmp_path / "dataset",
        )
    )
    dialog = LabelingDialog(store)
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def test_labeling_dialog_starts_with_two_poses_and_empty_source(qtbot, tmp_path) -> None:
    dialog = _dialog(qtbot, tmp_path)

    assert [row.name.text() for row in dialog.source_rows] == [
        "Pose 1",
        "Pose 2",
        "Leere Rutsche",
    ]
    assert [row.kind.text() for row in dialog.source_rows] == [
        "Klasse 0",
        "Klasse 1",
        "Negativ",
    ]
    assert dialog.source_rows[-1].is_empty
    assert not dialog.source_rows[-1].remove_button.isVisible()


def test_labeling_dialog_can_add_and_remove_pose_rows(qtbot, tmp_path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    pose3 = tmp_path / "pose3"
    pose3.mkdir()

    dialog.add_pose_button.click()
    assert [row.name.text() for row in dialog.source_rows] == [
        "Pose 1",
        "Pose 2",
        "Pose 3",
        "Leere Rutsche",
    ]
    dialog.source_rows[2].directory.setText(str(pose3))
    config = dialog._config()
    assert [source.name for source in config.pose_sources] == ["Pose 1", "Pose 2", "Pose 3"]

    dialog._remove_source_row(dialog.source_rows[1])
    assert [row.kind.text() for row in dialog.source_rows] == [
        "Klasse 0",
        "Klasse 1",
        "Negativ",
    ]


def test_visibility_review_preselects_only_clear_recommendations(qtbot, tmp_path) -> None:
    preview = tmp_path / "preview.png"
    source_a = tmp_path / "black.png"
    source_b = tmp_path / "borderline.png"
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    assert cv2.imwrite(str(preview), image)
    items = (
        VisibilityReviewItem(0, "Pose 1", 155, source_a, preview, 0.0, "schwarz", True),
        VisibilityReviewItem(
            0, "Pose 1", 155, source_b, preview, 0.4, "sehr dunkel", False
        ),
    )
    dialog = VisibilityReviewDialog(items)
    qtbot.addWidget(dialog)

    assert dialog.excluded_paths() == frozenset({source_a})
    dialog._set_all(Qt.CheckState.Checked)
    assert dialog.excluded_paths() == frozenset({source_a, source_b})
