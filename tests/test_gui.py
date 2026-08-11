from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QPushButton

from automated_image_capture.acquisition import AcquisitionSettings, PreflightCheck
from automated_image_capture.inference import InferenceDetection, InferenceFrame
from automated_image_capture.models import (
    CameraStatus,
    ConnectionState,
    ConveyorStatus,
    LightStatus,
    RobotStatus,
)
from automated_image_capture.settings import SettingsStore
from automated_image_capture.ui.main_window import MainWindow
from automated_image_capture.ui.widgets import AcquisitionDialog


def make_window(qtbot, tmp_path) -> MainWindow:
    backend = QSettings(str(tmp_path / "gui.ini"), QSettings.Format.IniFormat)
    window = MainWindow(SettingsStore(backend))
    qtbot.addWidget(window)
    window.show()
    return window


def test_dashboard_renders_status_without_hardware(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)

    window._camera_status(
        CameraStatus(
            model="TestCam",
            serial_number="42",
            ip_address="169.254.117.70",
            width=1920,
            height=1200,
            pixel_format="Mono8",
            camera_fps=30.0,
            stream_fps=25.0,
            preview_fps=15.0,
        )
    )
    window._robot_status(
        RobotStatus(
            rtde_connected=True,
            dashboard_connected=True,
            robot_mode="RUNNING",
            safety_mode="NORMAL",
            remote_control="true",
            speed_scaling=0.5,
        )
    )
    window._light_status(
        LightStatus(
            name="RGB660",
            address="AA:BB",
            connected=True,
            brightness=35,
            values_are_confirmed_commands=True,
        )
    )
    window._light_2_status(
        LightStatus(
            name="RGB660-2",
            address="CC:DD",
            connected=True,
            brightness=70,
            values_are_confirmed_commands=True,
        )
    )

    assert "TestCam" in window.camera_card.details.text()
    assert "30.0 / 25.0 / 15.0 FPS" in window.camera_card.details.text()
    assert "RUNNING" in window.robot_card.details.text()
    assert "35 %" in window.light_card.details.text()
    assert "70 %" in window.light_2_card.details.text()
    assert window.config.camera_serial == "42"
    assert window.config.light_address == "AA:BB"
    assert window.config.light_2_address == "CC:DD"


def test_dashboard_exposes_storage_cleanup_dialog(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)

    labels = {button.text() for button in window.findChildren(QPushButton)}

    assert "Speicher bereinigen …" in labels


def test_light_controls_follow_connection_state(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)

    window._light_state(ConnectionState.DISCONNECTED)
    assert not window.light_brightness.isEnabled()

    window._light_state(ConnectionState.CONNECTED)
    assert window.light_brightness.isEnabled()

    window._light_2_state(ConnectionState.DISCONNECTED)
    assert not window.light_2_card.brightness.isEnabled()
    window._light_2_state(ConnectionState.CONNECTED)
    assert window.light_2_card.brightness.isEnabled()


def test_camera_exposure_control_follows_status_and_emits_request(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)
    spy = QSignalSpy(window.camera_card.exposure_requested)

    window.camera_card.set_state(ConnectionState.CONNECTED)
    window._camera_status(
        CameraStatus(
            exposure_time_us=4_000,
            exposure_min_us=100,
            exposure_max_us=100_000,
            exposure_writable=True,
            exposure_auto="Off",
        )
    )

    assert window.camera_card.exposure_time.isEnabled()
    assert window.camera_card.exposure_time.value() == 4_000
    window.camera_card.exposure_time.setValue(7_500)
    window.camera_card.apply_exposure_button.click()
    assert len(spy) == 1
    assert spy[0][0] == 7_500.0

    window._camera_status(CameraStatus(exposure_writable=True, exposure_auto="Continuous"))
    assert window.camera_card.exposure_time.isEnabled()
    assert "deaktiviert ExposureAuto" in window.camera_card.exposure_hint.text()

    window._camera_status(CameraStatus(exposure_writable=False, exposure_auto="Continuous"))
    assert not window.camera_card.exposure_time.isEnabled()
    assert "ExposureAuto" in window.camera_card.exposure_hint.text()


def test_device_failures_do_not_change_other_cards(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)

    window.camera_card.set_state(ConnectionState.ERROR)

    assert window.camera_card.state is ConnectionState.ERROR
    assert window.robot_card.state is ConnectionState.DISCONNECTED
    assert window.light_card.state is ConnectionState.DISCONNECTED
    assert window.light_2_card.state is ConnectionState.DISCONNECTED


def test_live_inference_result_updates_preview_and_status(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)
    frame = InferenceFrame(
        image=np.zeros((80, 120, 3), dtype=np.uint8),
        detections=(
            InferenceDetection(
                0,
                "Pose 1",
                0.95,
                ((10, 10), (30, 10), (30, 30), (10, 30)),
            ),
        ),
        inference_ms=24.6,
        timestamp=0.0,
    )

    window._inference_frame(frame)

    assert window._last_image is not None
    assert "1 Objekt" in window.inference_status.text()
    assert "25 ms" in window.inference_status.text()


def test_resume_button_is_only_enabled_for_interrupted_sequence(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)
    window.acquisition_card.set_preflight(
        (PreflightCheck("ready", "Test", True, "bereit"),)
    )

    assert not window.acquisition_card.resume_button.isEnabled()
    window.acquisition_card.set_resume_available(True)
    assert window.acquisition_card.resume_button.isEnabled()
    window.acquisition_card.set_running(True)
    assert not window.acquisition_card.resume_button.isEnabled()
    window.acquisition_card.set_running(False)
    assert window.acquisition_card.resume_button.isEnabled()
    window.acquisition_card.set_resume_available(False)
    assert not window.acquisition_card.resume_button.isEnabled()


def test_acquisition_card_lists_failed_preflight_checks(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)

    window.acquisition_card.set_preflight(
        (
            PreflightCheck("camera", "Kamera", True, "verbunden"),
            PreflightCheck(
                "conveyor_origin",
                "Förderband-Nullpunkt",
                False,
                "Aktuelle Position als 0 mm übernehmen",
            ),
        )
    )

    assert not window.acquisition_card.start_button.isEnabled()
    assert "Start blockiert" in window.acquisition_card.preflight.text()
    assert "Förderband-Nullpunkt" in window.acquisition_card.preflight.text()
    assert "0 mm übernehmen" in window.acquisition_card.preflight.text()


def test_acquisition_dialog_switches_to_persistable_ramp_settings(qtbot, tmp_path) -> None:
    dialog = AcquisitionDialog(
        AcquisitionSettings(output_directory=Path(tmp_path)),
        CameraStatus(exposure_writable=True),
    )
    qtbot.addWidget(dialog)

    dialog.capture_mode.setCurrentIndex(dialog.capture_mode.findData("ramp"))
    dialog.ramp_duration.setValue(12.5)
    dialog.ramp_rate.setValue(8)
    dialog.ramp_light_1_period.setValue(1.6)
    dialog.ramp_light_2_period.setValue(12.5)
    config = dialog._current_config().validated()

    assert dialog.ramp_group.isVisibleTo(dialog)
    assert not dialog.light_1_row.isVisibleTo(dialog)
    assert config.capture_mode == "ramp"
    assert config.ramp_duration_s == 12.5
    assert config.ramp_image_rate_fps == 8
    assert "100" in dialog.estimate.text()


def test_acquisition_dialog_configures_continuous_angles_and_conveyor(qtbot, tmp_path) -> None:
    dialog = AcquisitionDialog(AcquisitionSettings(output_directory=Path(tmp_path)))
    qtbot.addWidget(dialog)

    dialog.robot_control_mode.setCurrentIndex(dialog.robot_control_mode.findData("angle"))
    dialog.conveyor_enabled.setChecked(True)
    dialog.conveyor_motion_mode.setCurrentIndex(
        dialog.conveyor_motion_mode.findData("synchronized")
    )
    dialog.conveyor_max_offset.setValue(50.0)
    dialog.conveyor_step.setValue(10.0)
    config = dialog._current_config().validated()

    assert config.robot_control_mode == "angle"
    assert config.angle_step_deg == 0.5
    assert config.conveyor_enabled
    assert config.conveyor_motion_mode == "synchronized"
    assert config.conveyor_max_offset_mm == 50.0
    assert config.conveyor_step_mm == 10.0
    assert dialog.angle_row.isVisibleTo(dialog)
    assert not dialog.pose_row.isVisibleTo(dialog)
    assert dialog.ramp_group.isVisibleTo(dialog)
    assert not dialog.conveyor_step.isEnabled()
    assert "pro Bandlauf" in dialog.estimate.text()


def test_conveyor_card_shows_calibration_and_enables_manual_controls(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)
    jog_spy = QSignalSpy(window.conveyor_card.jog_requested)
    window.conveyor_card.set_state(ConnectionState.CONNECTED)
    window._conveyor_status(
        ConveyorStatus(
            connected=True,
            calibration_valid=True,
            mm_per_full_step=0.32960026,
            ready_to_execute=True,
            internal_position=1234,
            logical_offset_mm=0.0,
        )
    )

    assert "gültig" in window.conveyor_card.details.text()
    assert window.conveyor_card.left_button.isEnabled()
    assert window.conveyor_card.right_button.isEnabled()
    assert window.conveyor_card.jog_distance.isEnabled()
    assert window.conveyor_card.origin_button.isEnabled()

    window.conveyor_card.jog_distance.setValue(12.5)
    window.conveyor_card.right_button.click()

    assert len(jog_spy) == 1
    assert jog_spy[0][0] == "right"
    assert jog_spy[0][1] == 12.5


def test_manual_conveyor_jog_starts_without_confirmation(qtbot, tmp_path, monkeypatch) -> None:
    window = make_window(qtbot, tmp_path)
    requests: list[tuple[str, float, float]] = []
    monkeypatch.setattr(
        window.conveyor,
        "jog",
        lambda direction, distance, speed: requests.append((direction, distance, speed)),
    )

    window._jog_conveyor("left", 37.5)

    assert requests == [("left", 37.5, 10.0)]


def test_robot_pose_requires_running_program_and_one_time_consent(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)
    spy = QSignalSpy(window.robot_card.pose_requested)

    window._robot_status(
        RobotStatus(
            rtde_connected=True,
            dashboard_connected=True,
            command_channel_connected=True,
            robot_mode="RUNNING",
            safety_mode="NORMAL",
            program_state="PLAYING BiBaZu.urp",
            loaded_program="/programs/BiBaZu.urp",
            command_state_code=1,
            command_state="Bereit",
        )
    )

    assert not window.robot_card.move_button.isEnabled()
    window.robot_card.motion_consent.setChecked(True)
    assert window.robot_card.move_button.isEnabled()

    window.robot_card.pose_selector.setCurrentIndex(3)
    window.robot_card.move_button.click()

    assert len(spy) == 1
    assert spy[0][0] == 180
    assert not window.robot_card.motion_consent.isChecked()
    assert not window.robot_card.move_button.isEnabled()


def test_robot_pose_is_disabled_while_command_is_pending(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)
    window._robot_status(
        RobotStatus(
            rtde_connected=True,
            dashboard_connected=True,
            command_channel_connected=True,
            robot_mode="RUNNING",
            safety_mode="NORMAL",
            program_state="PLAYING BiBaZu.urp",
            loaded_program="/programs/BiBaZu.urp",
            command_state_code=2,
            command_state="Fährt",
            command_pending=True,
        )
    )

    window.robot_card.motion_consent.setChecked(True)
    assert not window.robot_card.move_button.isEnabled()


def test_fixed_pose_button_is_disabled_for_continuous_ur_program(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)
    window._robot_status(
        RobotStatus(
            rtde_connected=True,
            dashboard_connected=True,
            command_channel_connected=True,
            robot_mode="RUNNING",
            safety_mode="NORMAL",
            program_state="PLAYING",
            loaded_program="/programs/BiBaZu_Continuous.urp",
            command_state_code=1,
        )
    )
    window.robot_card.motion_consent.setChecked(True)

    assert not window.robot_card.move_button.isEnabled()
