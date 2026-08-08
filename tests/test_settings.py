from __future__ import annotations

import pytest
from PyQt6.QtCore import QSettings

from automated_image_capture.acquisition import AcquisitionSettings
from automated_image_capture.dataset import default_build_config
from automated_image_capture.labeling import LabelingConfig, LabelSource
from automated_image_capture.settings import AppSettings, SettingsStore


def test_settings_round_trip(tmp_path) -> None:
    backend = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    store = SettingsStore(backend)
    expected = AppSettings(
        camera_ip="169.254.117.70",
        robot_ip="10.10.10.10",
        plc_ip="192.168.10.23",
        plc_ams_net_id="10.145.4.14.1.1",
        plc_port=851,
        conveyor_forward_direction="right",
        camera_cti_path=str(tmp_path / "producer.cti"),
        camera_serial="1234",
        light_address="AA:BB:CC:DD:EE:FF",
        light_name="RGB660",
        light_2_address="11:22:33:44:55:66",
        light_2_name="RGB660-2",
        preview_max_fps=12,
        maximize_camera_frame_rate=False,
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
        capture_mode="ramp",
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
        ramp_duration_s=12.5,
        ramp_image_rate_fps=8,
        ramp_light_1_period_s=1.6,
        ramp_light_2_period_s=12.5,
        robot_control_mode="angle",
        angle_start_deg=15.5,
        angle_end_deg=20.5,
        angle_step_deg=0.5,
        conveyor_enabled=True,
        conveyor_motion_mode="synchronized",
        conveyor_max_offset_mm=50.0,
        conveyor_step_mm=10.0,
        conveyor_speed_mm_per_s=10.0,
        conveyor_settle_ms=300,
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
        (
            LabelSource("Pose 1", foreground),
            LabelSource("Pose 2", tmp_path / "parts_2"),
            LabelSource("Leere Rutsche", background, is_empty=True),
        ),
        tmp_path / "yolo_obb",
        validation_fraction=0.25,
        minimum_difference=70,
        consensus_fraction=0.6,
        include_background_negatives=False,
        prefer_hardlinks=False,
    )

    store.save_labeling(expected)

    assert store.load_labeling() == expected


def test_legacy_labeling_settings_are_migrated_to_source_list(tmp_path) -> None:
    backend = QSettings(str(tmp_path / "legacy.ini"), QSettings.Format.IniFormat)
    backend.setValue("labeling/foreground_directory", str(tmp_path / "old_parts"))
    backend.setValue("labeling/background_directory", str(tmp_path / "old_empty"))
    store = SettingsStore(backend)

    loaded = store.load_labeling()

    assert loaded.sources[0] == LabelSource("Pose 1", tmp_path / "old_parts")
    assert loaded.sources[-1] == LabelSource("Leere Rutsche", tmp_path / "old_empty", is_empty=True)


def test_training_paths_round_trip(tmp_path) -> None:
    backend = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    store = SettingsStore(backend)
    source = tmp_path / "obb"
    output = tmp_path / "combined"
    dataset = output / "dataset_1"

    store.save_training_paths(source, output, dataset)
    loaded = store.load_training_paths(default_build_config())

    assert loaded == {
        "source_dataset": source,
        "output_root": output,
        "dataset_directory": dataset,
    }
