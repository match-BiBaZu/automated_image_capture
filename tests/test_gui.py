from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtTest import QSignalSpy

from automated_image_capture.acquisition import AcquisitionSettings
from automated_image_capture.inference import InferenceDetection, InferenceFrame
from automated_image_capture.models import (
    CameraStatus,
    ConnectionState,
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
    assert "RUNNING" in window.robot_card.details.text()
    assert "35 %" in window.light_card.details.text()
    assert "70 %" in window.light_2_card.details.text()
    assert window.config.camera_serial == "42"
    assert window.config.light_address == "AA:BB"
    assert window.config.light_2_address == "CC:DD"


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

    assert not window.acquisition_card.resume_button.isEnabled()
    window.acquisition_card.set_resume_available(True)
    assert window.acquisition_card.resume_button.isEnabled()
    window.acquisition_card.set_running(True)
    assert not window.acquisition_card.resume_button.isEnabled()
    window.acquisition_card.set_running(False)
    assert window.acquisition_card.resume_button.isEnabled()
    window.acquisition_card.set_resume_available(False)
    assert not window.acquisition_card.resume_button.isEnabled()


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
