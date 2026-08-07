from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtTest import QSignalSpy

from automated_image_capture.acquisition import (
    AcquisitionController,
    AcquisitionSettings,
    CapturePoint,
    DatasetWriter,
    build_capture_points,
    dump_yaml,
    triangle_brightness,
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


def test_default_ramp_has_60_unique_samples_and_triangle_extremes(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        capture_mode="ramp",
        pose_start=155,
        pose_end=155,
    )

    points = build_capture_points(settings)

    assert len(points) == 60
    assert [point.ramp_sample_id for point in points] == list(range(60))
    assert len({(point.light_1_brightness, point.light_2_brightness) for point in points}) == 60
    assert points[0].planned_offset_s == 0.0
    assert min(point.light_1_brightness for point in points) == 0
    assert max(point.light_1_brightness for point in points) == 100
    assert min(point.light_2_brightness for point in points) == 0
    assert max(point.light_2_brightness for point in points) == 100
    assert triangle_brightness(0.0, 2.4) == 0
    assert triangle_brightness(1.2, 2.4) == 100
    assert triangle_brightness(2.4, 2.4) == 0


def test_ramp_order_is_pose_then_exposure_then_sample(tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        capture_mode="ramp",
        pose_start=155,
        pose_end=160,
        ramp_duration_s=2,
        ramp_image_rate_fps=1,
        exposure_enabled=True,
        exposure_start_us=1000,
        exposure_end_us=2000,
        exposure_step_us=1000,
    )
    points = build_capture_points(settings)

    assert [(p.pose, p.exposure_time_us, p.ramp_sample_id) for p in points] == [
        (155, 1000, 0),
        (155, 1000, 1),
        (155, 2000, 0),
        (155, 2000, 1),
        (160, 1000, 0),
        (160, 1000, 1),
        (160, 2000, 0),
        (160, 2000, 1),
    ]


def test_ramp_writer_uses_unique_name_and_lossless_grayscale_png(tmp_path: Path) -> None:
    writer = DatasetWriter()
    session = writer.start_session(tmp_path)
    point = CapturePoint(155, 0, 0, None, ramp_sample_id=0, planned_offset_s=0.0)
    rgb_gray = np.repeat(np.arange(80, dtype=np.uint8).reshape(8, 10, 1), 3, axis=2)
    frame = CameraFrame(rgb_gray, "Mono8", time.time())

    writer.submit(0, point, frame, {"image": {}})
    writer.flush()
    writer.close()

    images = list(session.glob("*.png"))
    assert images[0].name == "img_000001_ur155_ramp-000_p1-000_p2-000_auto.png"
    decoded = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
    assert decoded.ndim == 2
    assert np.array_equal(decoded, rgb_gray[..., 0])


def test_old_checkpoint_settings_receive_ramp_defaults(tmp_path: Path) -> None:
    # Slotted dataclasses have no __dict__; this mirrors the original schema explicitly.
    old = {
        "output_directory": str(tmp_path),
        "pose_start": 155,
        "pose_end": 155,
        "light_1_start": 0,
        "light_1_end": 0,
        "light_1_step": 10,
        "light_2_start": 0,
        "light_2_end": 0,
        "light_2_step": 10,
        "exposure_enabled": False,
        "exposure_start_us": 5000,
        "exposure_end_us": 5000,
        "exposure_step_us": 1000,
        "light_settle_ms": 350,
        "robot_settle_ms": 500,
        "camera_settle_ms": 150,
    }

    restored = AcquisitionController._settings_from_manifest(old)

    assert restored.capture_mode == "grid"
    assert restored.ramp_duration_s == 10.0


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
        self.config = SimpleNamespace(preview_max_fps=15)

    def set_exposure_time(self, value: float) -> bool:
        self.status_changed.emit(CameraStatus(exposure_time_us=value, exposure_writable=True))
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
        self.status.last_command_confirmed_at = time.time()
        self.status_changed.emit(self.status)

    def try_set_ramp_brightness(self, brightness: int) -> bool:
        self.set_brightness(brightness)
        return True


def _ready_controller() -> tuple[
    AcquisitionController,
    FakeCamera,
    FakeRobot,
    FakeLight,
    FakeLight,
]:
    camera = FakeCamera()
    robot = FakeRobot()
    light_1 = FakeLight("PANEL-1")
    light_2 = FakeLight("PANEL-2")
    controller = AcquisitionController(camera, robot, light_1, light_2)
    camera.status_changed.emit(CameraStatus(model="TestCam"))
    robot.status_changed.emit(_robot_status(command_state_code=1))
    light_1.status_changed.emit(light_1.status)
    light_2.status_changed.emit(light_2.status)
    return controller, camera, robot, light_1, light_2


def _robot_status(
    *,
    command_state_code: int,
    sequence: int | None = None,
    acknowledged_pose: int | None = None,
) -> RobotStatus:
    return RobotStatus(
        rtde_connected=True,
        command_channel_connected=True,
        robot_mode="RUNNING",
        safety_mode="NORMAL",
        program_state="PLAYING",
        loaded_program="/programs/BiBaZu_GUI.urp",
        command_state_code=command_state_code,
        requested_sequence=sequence,
        acknowledged_sequence=sequence,
        acknowledged_pose=acknowledged_pose,
        command_pending=False,
    )


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


def test_two_sample_ramp_captures_on_timeline_and_finishes_at_zero(qtbot, tmp_path: Path) -> None:
    controller, camera, robot, light_1, light_2 = _ready_controller()
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        capture_mode="ramp",
        pose_start=180,
        pose_end=180,
        ramp_duration_s=2.0,
        ramp_image_rate_fps=1,
        ramp_light_1_period_s=2.0,
        ramp_light_2_period_s=2.0,
        robot_settle_ms=0,
    )

    assert controller.start(settings)
    qtbot.waitUntil(lambda: robot.requests == [180], timeout=2000)
    robot.status_changed.emit(
        _robot_status(command_state_code=3, sequence=9, acknowledged_pose=180)
    )
    qtbot.waitUntil(lambda: controller._phase == "ramp_frame", timeout=2000)
    first_origin = controller._ramp_origin_wall
    camera.frame_ready.emit(
        CameraFrame(np.full((8, 10, 3), 80, np.uint8), "Mono8", first_origin + 0.01)
    )
    qtbot.waitUntil(
        lambda: controller._index == 1 and controller._phase == "ramp_frame",
        timeout=3000,
    )
    camera.frame_ready.emit(
        CameraFrame(np.full((8, 10, 3), 120, np.uint8), "Mono8", first_origin + 1.01)
    )
    qtbot.waitUntil(lambda: not controller.running, timeout=3000)

    session = controller.session_directory
    assert session is not None
    assert len(list(session.glob("*.png"))) == 2
    assert light_1.status.brightness == light_2.status.brightness == 0
    yaml_text = sorted(session.glob("*.yaml"))[1].read_text(encoding="utf-8")
    assert "sample_id: 1" in yaml_text
    assert "planned_offset_s: 1.0" in yaml_text
    assert "timing_error_ms:" in yaml_text
    assert "last_confirmed_command_percent:" in yaml_text
    controller.close()


def test_ramp_capture_does_not_wait_for_slow_png_writer(qtbot, tmp_path: Path) -> None:
    controller, camera, robot, _light_1, _light_2 = _ready_controller()
    original_write = controller.writer._write_pair

    def slow_write(*args, **kwargs) -> None:
        time.sleep(0.4)
        original_write(*args, **kwargs)

    controller.writer._write_pair = slow_write  # type: ignore[method-assign]
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        capture_mode="ramp",
        pose_start=180,
        pose_end=180,
        ramp_duration_s=2.0,
        ramp_image_rate_fps=1,
        robot_settle_ms=0,
    )

    assert controller.start(settings)
    qtbot.waitUntil(lambda: robot.requests == [180], timeout=2000)
    robot.status_changed.emit(
        _robot_status(command_state_code=3, sequence=10, acknowledged_pose=180)
    )
    qtbot.waitUntil(lambda: controller._phase == "ramp_frame", timeout=2000)
    origin = controller._ramp_origin_wall
    camera.frame_ready.emit(CameraFrame(np.full((8, 10, 3), 80, np.uint8), "Mono8", origin + 0.01))

    assert controller._index == 1
    assert controller._phase == "ramp_frame"
    assert controller.writer.pending_count == 1
    controller.stop()
    controller.close()


def test_ramp_checkpoint_advances_only_across_contiguous_completed_writes(
    qtbot, tmp_path: Path
) -> None:
    controller, *_ = _ready_controller()
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        capture_mode="ramp",
        pose_start=180,
        pose_end=180,
        ramp_duration_s=2.0,
        ramp_image_rate_fps=1,
    )
    controller._settings = settings
    controller._points = build_capture_points(settings)
    controller._session_directory = controller.writer.start_session(tmp_path)
    controller._writer_token = controller.writer.session_token
    controller._running = True
    controller._index = 2
    controller._checkpoint_index = 0
    controller._ramp_pending_writes = {0, 1}

    controller._on_saved(controller._writer_token, 1, "second.png")
    assert controller._checkpoint_index == 0
    controller._on_saved(controller._writer_token, 0, "first.png")
    assert controller._checkpoint_index == 2

    controller._running = False
    controller.close()


def test_interrupted_sequence_resumes_in_same_directory_without_duplicates(
    qtbot, tmp_path: Path
) -> None:
    controller, camera, robot, light_1, light_2 = _ready_controller()
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        pose_start=180,
        pose_end=180,
        light_1_start=20,
        light_1_end=30,
        light_1_step=10,
        light_2_start=40,
        light_2_end=40,
        light_settle_ms=0,
        robot_settle_ms=0,
    )

    assert controller.start(settings)
    qtbot.waitUntil(lambda: robot.requests == [180], timeout=2000)
    robot.status_changed.emit(
        _robot_status(command_state_code=3, sequence=7, acknowledged_pose=180)
    )
    qtbot.waitUntil(lambda: controller._phase == "frame", timeout=2000)
    camera.frame_ready.emit(
        CameraFrame(np.full((8, 10, 3), 90, dtype=np.uint8), "RGB8", time.time() + 0.01)
    )
    qtbot.waitUntil(lambda: controller._index == 1, timeout=3000)
    qtbot.waitUntil(lambda: controller._phase == "frame", timeout=2000)
    session = controller.session_directory
    controller._fail("simulierter Kameraausfall")

    assert not controller.running
    assert controller.resume_available
    assert controller.remaining_count == 1
    assert session is not None
    assert (
        json.loads((session / "capture_session.json").read_text(encoding="utf-8"))["status"]
        == "interrupted"
    )

    camera.state = ConnectionState.ERROR
    assert not controller.resume()
    assert controller.resume_available
    camera.state = ConnectionState.CONNECTED
    assert controller.resume()
    qtbot.waitUntil(lambda: robot.requests == [180, 180], timeout=2000)
    robot.status_changed.emit(
        _robot_status(command_state_code=3, sequence=8, acknowledged_pose=180)
    )
    qtbot.waitUntil(lambda: controller._phase == "frame", timeout=2000)
    camera.frame_ready.emit(
        CameraFrame(np.full((8, 10, 3), 120, dtype=np.uint8), "RGB8", time.time() + 0.01)
    )
    qtbot.waitUntil(lambda: not controller.running, timeout=5000)

    assert not controller.resume_available
    assert len(list(session.glob("*.png"))) == 2
    assert len(list(session.glob("*.yaml"))) == 2
    manifest = json.loads((session / "capture_session.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["next_index"] == 2
    controller.close()


def test_interrupted_session_is_restored_after_controller_restart(qtbot, tmp_path: Path) -> None:
    first, _camera, robot, _light_1, _light_2 = _ready_controller()
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        pose_start=180,
        pose_end=180,
        light_1_start=20,
        light_1_end=30,
        light_1_step=10,
        light_2_start=40,
        light_2_end=40,
    )
    assert first.start(settings)
    qtbot.waitUntil(lambda: robot.requests == [180], timeout=2000)
    session = first.session_directory
    first._fail("simulierter Neustart")
    first.close()

    second, _camera2, _robot2, _light3, _light4 = _ready_controller()
    assert second.restore_interrupted(settings)
    assert second.resume_available
    assert second.remaining_count == 2
    assert second.session_directory == session
    second.close()


def _create_checkpoint(
    controller: AcquisitionController,
    settings: AcquisitionSettings,
    *,
    complete_first_pair: bool,
    orphan_first_image: bool = False,
) -> Path:
    points = build_capture_points(settings)
    session = controller.writer.start_session(settings.output_directory)
    controller._settings = settings
    controller._points = points
    controller._index = 0
    controller._session_directory = session
    controller._writer_token = controller.writer.session_token
    image_path, yaml_path = controller.writer.expected_paths(0, points[0])
    if complete_first_pair or orphan_first_image:
        image_path.write_bytes(b"test")
    if complete_first_pair:
        yaml_path.write_text("test: true\n", encoding="utf-8")
    controller._write_manifest("interrupted", "Test-Checkpoint")
    return session


def test_restore_reconciles_complete_file_pairs_without_recapturing(qtbot, tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        pose_start=180,
        pose_end=180,
        light_1_start=20,
        light_1_end=30,
        light_1_step=10,
        light_2_start=40,
        light_2_end=40,
    )
    first, *_ = _ready_controller()
    session = _create_checkpoint(first, settings, complete_first_pair=True)
    first.close()

    second, *_ = _ready_controller()
    assert second.restore_interrupted(settings)
    assert second.session_directory == session
    assert second.remaining_count == 1
    manifest = json.loads((session / "capture_session.json").read_text(encoding="utf-8"))
    assert manifest["next_index"] == 1
    second.close()


def test_restore_rejects_incomplete_png_yaml_pair(qtbot, tmp_path: Path) -> None:
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        pose_start=180,
        pose_end=180,
        light_1_start=20,
        light_1_end=30,
        light_1_step=10,
        light_2_start=40,
        light_2_end=40,
    )
    first, *_ = _ready_controller()
    _create_checkpoint(
        first,
        settings,
        complete_first_pair=False,
        orphan_first_image=True,
    )
    first.close()

    second, *_ = _ready_controller()
    errors = QSignalSpy(second.error)
    assert not second.restore_interrupted(settings)
    assert not second.resume_available
    assert len(errors) == 1
    assert "Unvollständiges Dateipaar" in errors[0][0]
    second.close()


def test_legacy_interrupted_folder_is_reconstructed_from_complete_prefix(
    qtbot, tmp_path: Path
) -> None:
    settings = AcquisitionSettings(
        output_directory=tmp_path,
        pose_start=180,
        pose_end=180,
        light_1_start=20,
        light_1_end=30,
        light_1_step=10,
        light_2_start=40,
        light_2_end=40,
    )
    first, *_ = _ready_controller()
    points = build_capture_points(settings)
    session = first.writer.start_session(tmp_path)
    image_path, yaml_path = first.writer.expected_paths(0, points[0])
    image_path.write_bytes(b"legacy")
    yaml_path.write_text("legacy: true\n", encoding="utf-8")
    first.close()

    second, *_ = _ready_controller()
    assert second.restore_interrupted(settings)
    assert second.resume_available
    assert second.remaining_count == 1
    assert second.session_directory == session
    manifest = json.loads((session / "capture_session.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["next_index"] == 1
    second.close()
