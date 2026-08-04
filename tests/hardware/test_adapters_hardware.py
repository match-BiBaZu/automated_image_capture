from __future__ import annotations

import os

import pytest

from automated_image_capture.hardware.camera import CameraAdapter
from automated_image_capture.hardware.robot import RobotAdapter
from automated_image_capture.models import ConnectionState
from automated_image_capture.settings import AppSettings

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        os.environ.get("RUN_HARDWARE_TESTS") != "1",
        reason="Set RUN_HARDWARE_TESTS=1 for physical hardware tests",
    ),
]


def _config() -> AppSettings:
    return AppSettings(
        camera_ip=os.environ.get("BAUMER_IP", "169.254.117.70"),
        robot_ip=os.environ.get("UR_IP", "10.10.10.10"),
        camera_cti_path=os.environ.get(
            "BAUMER_CTI", r"C:\Program Files\Baumer Camera Explorer\bgapi2_gige.cti"
        ),
        preview_max_fps=15,
    )


def test_camera_qt_adapter(qtbot) -> None:
    camera = CameraAdapter(_config())
    frames: list[object] = []
    errors: list[str] = []
    camera.frame_ready.connect(frames.append)
    camera.error.connect(errors.append)

    try:
        camera.connect()
        qtbot.waitUntil(lambda: len(frames) >= 3, timeout=20_000)
        assert camera.state is ConnectionState.CONNECTED
        assert not errors
    finally:
        camera.disconnect()

    assert camera.state is ConnectionState.DISCONNECTED


def test_robot_qt_adapter(qtbot) -> None:
    robot = RobotAdapter(_config())
    updates: list[object] = []
    errors: list[str] = []
    robot.status_changed.connect(updates.append)
    robot.error.connect(errors.append)

    try:
        robot.connect()
        qtbot.waitUntil(lambda: len(updates) >= 3, timeout=20_000)
        assert robot.state in {ConnectionState.CONNECTED, ConnectionState.DEGRADED}
        assert updates[-1].rtde_connected
    finally:
        robot.disconnect()

    assert robot.state is ConnectionState.DISCONNECTED


def test_camera_and_robot_qt_adapters_together(qtbot) -> None:
    config = _config()
    camera = CameraAdapter(config)
    robot = RobotAdapter(config)
    frames: list[object] = []
    robot_updates: list[object] = []
    camera_errors: list[str] = []
    robot_errors: list[str] = []
    camera.frame_ready.connect(frames.append)
    robot.status_changed.connect(robot_updates.append)
    camera.error.connect(camera_errors.append)
    robot.error.connect(robot_errors.append)

    try:
        camera.connect()
        robot.connect()
        qtbot.waitUntil(
            lambda: len(frames) >= 3 and len(robot_updates) >= 3,
            timeout=20_000,
        )
        assert camera.state is ConnectionState.CONNECTED
        assert robot.state in {ConnectionState.CONNECTED, ConnectionState.DEGRADED}
        assert not camera_errors
        assert robot_updates[-1].rtde_connected
    finally:
        camera.disconnect()
        robot.disconnect()

    assert camera.state is ConnectionState.DISCONNECTED
    assert robot.state is ConnectionState.DISCONNECTED
