from __future__ import annotations

import pytest
from PyQt6.QtCore import QSettings

from automated_image_capture.acquisition import AcquisitionSettings
from automated_image_capture.labeling import LabelingConfig
from automated_image_capture.settings import AppSettings, SettingsStore


def test_settings_round_trip(tmp_path) -> None:
    backend = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    store = SettingsStore(backend)
    expected = AppSettings(
        camera_ip="169.254.117.70",
        robot_ip="10.10.10.10",
        camera_cti_path=str(tmp_path / "producer.cti"),
        camera_serial="1234",
        light_address="AA:BB:CC:DD:EE:FF",
        light_name="RGB660",
        light_2_address="11:22:33:44:55:66",
        light_2_name="RGB660-2",
        preview_max_fps=12,
        auto_reconnect=False,
    )

    store.save(expected)

    assert store.load() == expected


@pytest.mark.parametrize("field", ["camera_ip", "robot_ip"])
def test_invalid_ip_is_rejected(field: str) -> None:
    values = {field: "not-an-ip"}
    with pytest.raises(ValueError):
        AppSettings(**values).validated()


def test_invalid_preview_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="Bildrate"):
        AppSettings(preview_max_fps=0).validated()


def test_acquisition_settings_round_trip(tmp_path) -> None:
    backend = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    store = SettingsStore(backend)
    expected = AcquisitionSettings(
        output_directory=tmp_path / "dataset",
        pose_start=160,
        pose_end=200,
        light_1_start=10,
        light_1_end=80,
        light_1_step=5,
        light_2_start=20,
        light_2_end=90,
        light_2_step=10,
        exposure_enabled=True,
        exposure_start_us=2000,
        exposure_end_us=8000,
        exposure_step_us=2000,
    )

    store.save_acquisition(expected)

    assert store.load_acquisition() == expected


def test_labeling_settings_round_trip(tmp_path) -> None:
    backend = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    store = SettingsStore(backend)
    foreground = tmp_path / "parts"
    background = tmp_path / "empty"
    foreground.mkdir()
    background.mkdir()
    expected = LabelingConfig(
        foreground,
        background,
        tmp_path / "yolo_obb",
        class_name="Kk1",
        class_id=2,
        validation_fraction=0.25,
        minimum_difference=70,
        consensus_fraction=0.6,
        include_background_negatives=False,
        prefer_hardlinks=False,
    )

    store.save_labeling(expected)

    assert store.load_labeling() == expected
