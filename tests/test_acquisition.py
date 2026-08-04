from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from automated_image_capture.acquisition import (
    AcquisitionController,
    AcquisitionSettings,
    build_capture_points,
    dump_yaml,
)
from automated_image_capture.models import (
    CameraFrame,
    CameraStatus,
    ConnectionState,
    LightStatus,
    RobotStatus,
)


def test_variation_order_exhausts_lights_before_next_pose(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        pose_start=155,
        pose_end=160,
        light_1_start=0,
        light_1_end=20,
        light_1_step=10,
        light_2_start=0,
        light_2_end=10,
        light_2_step=10,
    )

    points = build_capture_points(settings)

    first_pose = [(point.light_1_brightness, point.light_2_brightness) for point in points[:6]]
    assert first_pose == [(0, 0), (10, 0), (20, 0), (0, 10), (10, 10), (20, 10)]
    assert [point.pose for point in points[:6]] == [155] * 6
    assert [point.pose for point in points[6:]] == [160] * 6


def test_exposure_is_innermost_variation(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        pose_start=180,
        pose_end=180,
        light_1_start=10,
        light_1_end=20,
        light_1_step=10,
        light_2_start=30,
        light_2_end=30,
        exposure_enabled=True,
        exposure_start_us=1000,
        exposure_end_us=2000,
        exposure_step_us=1000,
    )

    points = build_capture_points(settings)

    assert [point.exposure_time_us for point in points] == [1000, 2000, 1000, 2000]
    assert [point.light_1_brightness for point in points] == [10, 10, 20, 20]


def test_new_named_poses_follow_configured_order(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        pose_start=1155,
        pose_end=2200,
        light_1_start=0,
        light_1_end=0,
        light_2_start=0,
        light_2_end=0,
    )

    assert [point.pose for point in build_capture_points(settings)] == [
        1155,
        1170,
        1185,
        1200,
        2155,
        2170,
        2185,
        2200,
    ]


def test_yaml_serializer_handles_nested_metadata() -> None:
    text = dump_yaml(
        {
            "camera": {"model": "Baumer VCX", "gain": 1.5},
            "values": [1, 2],
            "empty": [],
        }
    )

    assert 'model: "Baumer VCX"' in text
    assert "gain: 1.5" in text
    assert "  - 2" in text
    assert "empty: []" in text


class FakeCamera(QObject):
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.state = ConnectionState.CONNECTED

    def set_exposure_time(self, value: float) -> bool:
        self.status_changed.emit(
            CameraStatus(exposure_time_us=value, exposure_writable=True)
        )
        return True

    def restore_exposure(self) -> bool:
        return True


class FakeRobot(QObject):
    status_changed = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[int] = []

    def request_pose(self, pose: int) -> bool:
        self.requests.append(pose)
        return True


class FakeLight(QObject):
    status_changed = pyqtSignal(object)

    def __init__(self, address: str) -> None:
        super().__init__()
        self.state = ConnectionState.CONNECTED
        self.status = LightStatus(address=address, connected=True, power=False)

    def set_power(self, enabled: bool) -> None:
        self.status.power = enabled
        self.status.values_are_confirmed_commands = True
        self.status_changed.emit(self.status)

    def set_brightness(self, brightness: int) -> None:
        self.status.brightness = brightness
        self.status.values_are_confirmed_commands = True
        self.status_changed.emit(self.status)


def test_single_point_sequence_saves_png_and_yaml(qtbot, tmp_path: Path) -> None:
    camera = FakeCamera()
    robot = FakeRobot()
    light_1 = FakeLight("PANEL-1")
    light_2 = FakeLight("PANEL-2")
    controller = AcquisitionController(camera, robot, light_1, light_2)
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        pose_start=180,
        pose_end=180,
        light_1_start=20,
        light_1_end=20,
        light_2_start=30,
        light_2_end=30,
        light_settle_ms=0,
        robot_settle_ms=0,
    )
    camera.status_changed.emit(CameraStatus(model="TestCam"))
    robot.status_changed.emit(
        RobotStatus(
            rtde_connected=True,
            command_channel_connected=True,
            robot_mode="RUNNING",
            safety_mode="NORMAL",
            program_state="PLAYING",
            loaded_program="/programs/BiBaZu_GUI.urp",
            command_state_code=1,
        )
    )
    light_1.status_changed.emit(light_1.status)
    light_2.status_changed.emit(light_2.status)

    assert controller.start(settings)
    qtbot.waitUntil(lambda: robot.requests == [180], timeout=2000)
    robot.status_changed.emit(
        RobotStatus(
            # Register 41 is redundant and may remain zero on some UR programs.
            acknowledged_pose=None,
            acknowledged_sequence=7,
            requested_sequence=7,
            command_state_code=3,
            command_pending=False,
        )
    )
    qtbot.waitUntil(
        lambda: light_1.status.brightness == 20 and light_2.status.brightness == 30,
        timeout=2000,
    )
    qtbot.waitUntil(lambda: controller._phase == "frame", timeout=2000)
    camera.frame_ready.emit(
        CameraFrame(
            np.full((8, 10, 3), 127, dtype=np.uint8),
            "RGB8",
            time.time() + 0.01,
        )
    )
    qtbot.waitUntil(lambda: not controller.running, timeout=5000)
    controller.close()

    images = list(tmp_path.rglob("*.png"))
    metadata = list(tmp_path.rglob("*.yaml"))
    assert len(images) == 1
    assert len(metadata) == 1
    assert images[0].stem == metadata[0].stem
    yaml_text = metadata[0].read_text(encoding="utf-8")
    assert "requested_pose_id: 180" in yaml_text
    assert "requested_brightness_percent: 20" in yaml_text
