from __future__ import annotations

from PyQt6.QtCore import QSettings

from automated_image_capture.models import (
    CameraStatus,
    ConnectionState,
    LightStatus,
    RobotStatus,
)
from automated_image_capture.settings import SettingsStore
from automated_image_capture.ui.main_window import MainWindow


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


def test_device_failures_do_not_change_other_cards(qtbot, tmp_path) -> None:
    window = make_window(qtbot, tmp_path)

    window.camera_card.set_state(ConnectionState.ERROR)

    assert window.camera_card.state is ConnectionState.ERROR
    assert window.robot_card.state is ConnectionState.DISCONNECTED
    assert window.light_card.state is ConnectionState.DISCONNECTED
    assert window.light_2_card.state is ConnectionState.DISCONNECTED
