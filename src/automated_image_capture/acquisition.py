from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from automated_image_capture.hardware.camera import CameraAdapter
from automated_image_capture.hardware.conveyor import ConveyorAdapter
from automated_image_capture.hardware.light import LIGHT_COMMAND_TIMEOUT_SECONDS, LightAdapter
from automated_image_capture.hardware.robot import ALLOWED_POSES, RobotAdapter
from automated_image_capture.models import (
    CameraFrame,
    CameraStatus,
    ConnectionState,
    ConveyorStatus,
    LightStatus,
    RobotCommandMode,
    RobotStatus,
)

RAMP_BLE_COMMAND_TIMEOUT_SECONDS = LIGHT_COMMAND_TIMEOUT_SECONDS
SYNCHRONIZED_FRAME_LATE_TOLERANCE_SECONDS = 1.5


def ramp_command_timed_out(adapter: object) -> bool:
    return (
        bool(getattr(adapter, "command_busy", False))
        and float(getattr(adapter, "command_age_seconds", 0.0)) > RAMP_BLE_COMMAND_TIMEOUT_SECONDS
    )


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def dump_yaml(value: Any, indent: int = 0) -> str:
    """Serialize the metadata subset used here as readable YAML 1.2."""
    prefix = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if item == {}:
                lines.append(f"{prefix}{key}: {{}}")
            elif item == []:
                lines.append(f"{prefix}{key}: []")
            elif isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(dump_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(dump_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def inclusive_values(start: int, end: int, step: int) -> tuple[int, ...]:
    if step <= 0:
        raise ValueError("Die Schrittweite muss größer als null sein.")
    direction = 1 if end >= start else -1
    values = list(range(start, end + direction, direction * step))
    if values[-1] != end:
        values.append(end)
    return tuple(values)


def poses_between(start: int, end: int) -> tuple[int, ...]:
    if start not in ALLOWED_POSES or end not in ALLOWED_POSES:
        raise ValueError("Start und Ende müssen freigegebene UR-Posen sein.")
    start_index = ALLOWED_POSES.index(start)
    end_index = ALLOWED_POSES.index(end)
    if start_index <= end_index:
        return ALLOWED_POSES[start_index : end_index + 1]
    return tuple(reversed(ALLOWED_POSES[end_index : start_index + 1]))


def _tenths(value: float, label: str) -> int:
    converted = round(float(value) * 10.0)
    if abs(float(value) * 10.0 - converted) > 1e-6:
        raise ValueError(f"{label} darf höchstens eine Nachkommastelle haben.")
    return converted


def angle_values(start_deg: float, end_deg: float, step_deg: float) -> tuple[int, ...]:
    start = _tenths(start_deg, "Der UR-Startwinkel")
    end = _tenths(end_deg, "Der UR-Endwinkel")
    step = _tenths(step_deg, "Die UR-Winkelschrittweite")
    if not 155 <= start <= 210 or not 155 <= end <= 210:
        raise ValueError("UR-Winkel müssen zwischen 15,5 und 21,0 Grad liegen.")
    return inclusive_values(start, end, step)


def conveyor_positions(max_offset_mm: float, step_mm: float) -> tuple[tuple[int, float, str], ...]:
    maximum = _tenths(max_offset_mm, "Der maximale Förderbandoffset")
    step = _tenths(step_mm, "Die Förderbandschrittweite")
    if maximum < 0 or maximum > 50000:
        raise ValueError("Der Förderbandoffset muss zwischen 0 und 5000 mm liegen.")
    if step <= 0:
        raise ValueError("Die Förderbandschrittweite muss positiv sein.")
    outward = list(range(0, maximum + 1, step))
    if outward[-1] != maximum:
        outward.append(maximum)
    values = [(index, value / 10.0, "out") for index, value in enumerate(outward)]
    for value in reversed(outward[:-1]):
        values.append((len(values), value / 10.0, "back"))
    return tuple(values)


@dataclass(slots=True, frozen=True)
class AcquisitionSettings:
    output_directory: Path
    capture_mode: str = "grid"
    pose_start: int = 155
    pose_end: int = 210
    light_1_start: int = 0
    light_1_end: int = 100
    light_1_step: int = 10
    light_2_start: int = 0
    light_2_end: int = 100
    light_2_step: int = 10
    exposure_enabled: bool = False
    exposure_start_us: int = 5000
    exposure_end_us: int = 5000
    exposure_step_us: int = 1000
    light_settle_ms: int = 350
    robot_settle_ms: int = 500
    camera_settle_ms: int = 150
    ramp_duration_s: float = 10.0
    ramp_image_rate_fps: int = 6
    ramp_light_1_period_s: float = 2.4
    ramp_light_2_period_s: float = 10.0
    robot_control_mode: str = RobotCommandMode.POSE_ID.value
    angle_start_deg: float = 15.5
    angle_end_deg: float = 21.0
    angle_step_deg: float = 0.5
    conveyor_enabled: bool = False
    conveyor_motion_mode: str = "stations"
    conveyor_max_offset_mm: float = 50.0
    conveyor_step_mm: float = 10.0
    conveyor_speed_mm_per_s: float = 10.0
    conveyor_settle_ms: int = 300

    def validated(self) -> AcquisitionSettings:
        if not str(self.output_directory):
            raise ValueError("Bitte einen Speicherort auswählen.")
        if self.capture_mode not in {"grid", "ramp"}:
            raise ValueError("Der Aufnahmemodus ist ungültig.")
        for value in (
            self.light_1_start,
            self.light_1_end,
            self.light_2_start,
            self.light_2_end,
        ):
            if not 0 <= value <= 100:
                raise ValueError("Panelhelligkeiten müssen zwischen 0 und 100 % liegen.")
        inclusive_values(self.light_1_start, self.light_1_end, self.light_1_step)
        inclusive_values(self.light_2_start, self.light_2_end, self.light_2_step)
        poses_between(self.pose_start, self.pose_end)
        if self.robot_control_mode not in {mode.value for mode in RobotCommandMode}:
            raise ValueError("Der UR-Steuerungsmodus ist ungültig.")
        angle_values(self.angle_start_deg, self.angle_end_deg, self.angle_step_deg)
        conveyor_positions(self.conveyor_max_offset_mm, self.conveyor_step_mm)
        if self.conveyor_motion_mode not in {"stations", "synchronized"}:
            raise ValueError("Der Förderband-Aufnahmemodus ist ungültig.")
        if (
            self.conveyor_enabled
            and self.conveyor_motion_mode == "synchronized"
            and self.conveyor_max_offset_mm <= 0.0
        ):
            raise ValueError(
                "Für die synchronisierte Fahrt muss der maximale Offset größer als 0 mm sein."
            )
        if not 0.1 <= self.conveyor_speed_mm_per_s <= 5000.0:
            raise ValueError(
                "Die Förderbandgeschwindigkeit muss zwischen 0,1 und 5000 mm/s liegen."
            )
        if not 0 <= self.conveyor_settle_ms <= 10000:
            raise ValueError("Die Förderband-Beruhigungszeit muss zwischen 0 und 10000 ms liegen.")
        if self.exposure_enabled:
            if min(self.exposure_start_us, self.exposure_end_us, self.exposure_step_us) <= 0:
                raise ValueError("Belichtungszeiten und Schrittweite müssen positiv sein.")
            inclusive_values(
                self.exposure_start_us,
                self.exposure_end_us,
                self.exposure_step_us,
            )
        if min(self.light_settle_ms, self.robot_settle_ms, self.camera_settle_ms) < 0:
            raise ValueError("Stabilisierungszeiten dürfen nicht negativ sein.")
        if not 2.0 <= self.ramp_duration_s <= 120.0:
            raise ValueError("Die Rampendauer muss zwischen 2 und 120 Sekunden liegen.")
        if not 1 <= self.ramp_image_rate_fps <= 10:
            raise ValueError("Die Rampen-Bildrate muss zwischen 1 und 10 Bildern/s liegen.")
        if not all(
            0.8 <= period <= 120.0
            for period in (self.ramp_light_1_period_s, self.ramp_light_2_period_s)
        ):
            raise ValueError("Panelperioden müssen zwischen 0,8 und 120 Sekunden liegen.")
        return self


@dataclass(slots=True, frozen=True)
class CapturePoint:
    pose: int
    light_1_brightness: int
    light_2_brightness: int
    exposure_time_us: int | None
    ramp_sample_id: int | None = None
    planned_offset_s: float | None = None
    robot_control_mode: str = RobotCommandMode.POSE_ID.value
    angle_tenths: int | None = None
    conveyor_station_id: int = 0
    conveyor_offset_mm: float = 0.0
    conveyor_actual_offset_mm: float | None = None
    conveyor_direction: str = "fixed"
    conveyor_motion_mode: str = "stations"

    @property
    def robot_raw_value(self) -> int:
        return self.angle_tenths if self.angle_tenths is not None else self.pose

    @property
    def robot_key(self) -> tuple[str, int]:
        return (self.robot_control_mode, self.robot_raw_value)

    @property
    def conveyor_key(self) -> tuple[int, float, str]:
        return (self.conveyor_station_id, self.conveyor_offset_mm, self.conveyor_direction)


@dataclass(slots=True, frozen=True)
class PreflightCheck:
    key: str
    label: str
    ready: bool
    detail: str


def triangle_brightness(elapsed_s: float, period_s: float) -> int:
    """Deterministic triangular waveform from zero to 100 and back."""
    phase = (float(elapsed_s) % float(period_s)) / float(period_s)
    normalized = 2.0 * phase if phase <= 0.5 else 2.0 * (1.0 - phase)
    return max(0, min(100, round(100.0 * normalized)))


def synchronized_sweep_profile(
    settings: AcquisitionSettings,
) -> tuple[float, float, int]:
    """Return active-motion duration, effective speed and samples per round trip."""
    natural_duration = 2.0 * settings.conveyor_max_offset_mm / settings.conveyor_speed_mm_per_s
    if settings.capture_mode == "ramp":
        sample_count = max(2, round(natural_duration * settings.ramp_image_rate_fps))
        duration = natural_duration
    else:
        light_count = len(
            inclusive_values(settings.light_1_start, settings.light_1_end, settings.light_1_step)
        ) * len(
            inclusive_values(settings.light_2_start, settings.light_2_end, settings.light_2_step)
        )
        # The selected speed is an upper bound. Slow the belt when necessary so
        # discrete BLE targets are never scheduled faster than the chosen sample rate.
        duration = max(natural_duration, light_count / settings.ramp_image_rate_fps)
        sample_count = max(2, round(duration * settings.ramp_image_rate_fps))
    effective_speed = 2.0 * settings.conveyor_max_offset_mm / duration
    return duration, effective_speed, sample_count


def _sweep_sample(
    sample_id: int,
    sample_count: int,
    duration_s: float,
    maximum_mm: float,
) -> tuple[float, float, str]:
    # Samples sit in the middle of equal time slots. This avoids taking a nominally
    # moving image before the first motion or after the return has already completed.
    planned_s = (sample_id + 0.5) * duration_s / sample_count
    phase = planned_s / duration_s
    if phase < 0.5:
        return planned_s, maximum_mm * phase * 2.0, "out"
    return planned_s, maximum_mm * (2.0 - phase * 2.0), "back"


def build_capture_points(settings: AcquisitionSettings) -> tuple[CapturePoint, ...]:
    settings = settings.validated()
    exposures: tuple[int | None, ...]
    if settings.exposure_enabled:
        exposures = inclusive_values(
            settings.exposure_start_us,
            settings.exposure_end_us,
            settings.exposure_step_us,
        )
    else:
        exposures = (None,)
    if settings.robot_control_mode == RobotCommandMode.ANGLE.value:
        robot_targets = tuple(
            (raw, raw) for raw in angle_values(
                settings.angle_start_deg, settings.angle_end_deg, settings.angle_step_deg
            )
        )
    else:
        robot_targets = tuple(
            (pose, None) for pose in poses_between(settings.pose_start, settings.pose_end)
        )
    synchronized = settings.conveyor_enabled and settings.conveyor_motion_mode == "synchronized"
    if synchronized:
        duration, _effective_speed, sample_count = synchronized_sweep_profile(settings)
        light_pairs = tuple(
            (light_1, light_2)
            for light_2 in inclusive_values(
                settings.light_2_start, settings.light_2_end, settings.light_2_step
            )
            for light_1 in inclusive_values(
                settings.light_1_start, settings.light_1_end, settings.light_1_step
            )
        )
        scale = duration / settings.ramp_duration_s
        panel_periods = (
            settings.ramp_light_1_period_s * scale,
            settings.ramp_light_2_period_s * scale,
        )
        points: list[CapturePoint] = []
        for raw, angle in robot_targets:
            for exposure in exposures:
                for sample_id in range(sample_count):
                    planned_s, offset_mm, direction = _sweep_sample(
                        sample_id,
                        sample_count,
                        duration,
                        settings.conveyor_max_offset_mm,
                    )
                    if settings.capture_mode == "ramp":
                        light_1 = triangle_brightness(planned_s, panel_periods[0])
                        light_2 = triangle_brightness(planned_s, panel_periods[1])
                        ramp_sample_id: int | None = sample_id
                    else:
                        pair_index = min(
                            len(light_pairs) - 1,
                            sample_id * len(light_pairs) // sample_count,
                        )
                        light_1, light_2 = light_pairs[pair_index]
                        ramp_sample_id = None
                    points.append(
                        CapturePoint(
                            pose=raw,
                            light_1_brightness=light_1,
                            light_2_brightness=light_2,
                            exposure_time_us=exposure,
                            ramp_sample_id=ramp_sample_id,
                            planned_offset_s=planned_s,
                            robot_control_mode=settings.robot_control_mode,
                            angle_tenths=angle,
                            conveyor_station_id=sample_id,
                            conveyor_offset_mm=offset_mm,
                            conveyor_direction=direction,
                            conveyor_motion_mode="synchronized",
                        )
                    )
        return tuple(points)

    belt_positions = (
        conveyor_positions(settings.conveyor_max_offset_mm, settings.conveyor_step_mm)
        if settings.conveyor_enabled
        else ((0, 0.0, "fixed"),)
    )
    if settings.capture_mode == "ramp":
        sample_count = round(settings.ramp_duration_s * settings.ramp_image_rate_fps)
        return tuple(
            CapturePoint(
                pose=raw,
                light_1_brightness=triangle_brightness(
                    sample_id / settings.ramp_image_rate_fps,
                    settings.ramp_light_1_period_s,
                ),
                light_2_brightness=triangle_brightness(
                    sample_id / settings.ramp_image_rate_fps,
                    settings.ramp_light_2_period_s,
                ),
                exposure_time_us=exposure,
                ramp_sample_id=sample_id,
                planned_offset_s=sample_id / settings.ramp_image_rate_fps,
                robot_control_mode=settings.robot_control_mode,
                angle_tenths=angle,
                conveyor_station_id=station_id,
                conveyor_offset_mm=offset_mm,
                conveyor_direction=direction,
                conveyor_motion_mode=settings.conveyor_motion_mode,
            )
            for raw, angle in robot_targets
            for station_id, offset_mm, direction in belt_positions
            for exposure in exposures
            for sample_id in range(sample_count)
        )
    return tuple(
        CapturePoint(
            raw,
            light_1,
            light_2,
            exposure,
            robot_control_mode=settings.robot_control_mode,
            angle_tenths=angle,
            conveyor_station_id=station_id,
            conveyor_offset_mm=offset_mm,
            conveyor_direction=direction,
            conveyor_motion_mode=settings.conveyor_motion_mode,
        )
        for raw, angle in robot_targets
        for station_id, offset_mm, direction in belt_positions
        for exposure in exposures
        for light_2 in inclusive_values(
            settings.light_2_start,
            settings.light_2_end,
            settings.light_2_step,
        )
        for light_1 in inclusive_values(
            settings.light_1_start,
            settings.light_1_end,
            settings.light_1_step,
        )
    )


class DatasetWriter(QObject):
    saved = pyqtSignal(int, int, str)
    failed = pyqtSignal(int, int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="dataset-writer")
        self._session_directory: Path | None = None
        self._session_token = 0
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._pending_condition = threading.Condition(self._pending_lock)

    @property
    def session_token(self) -> int:
        return self._session_token

    def start_session(self, output_directory: Path) -> Path:
        root = output_directory.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("capture_%Y%m%d_%H%M%S")
        session = root / stamp
        suffix = 1
        while session.exists():
            session = root / f"{stamp}_{suffix:02d}"
            suffix += 1
        session.mkdir()
        self._session_directory = session
        self._session_token += 1
        return session

    def attach_session(self, session_directory: Path) -> None:
        session = session_directory.expanduser().resolve()
        if not session.is_dir():
            raise RuntimeError(f"Aufnahmesitzung nicht gefunden: {session}")
        self._session_directory = session
        self._session_token += 1

    def expected_paths(self, index: int, point: CapturePoint) -> tuple[Path, Path]:
        if self._session_directory is None:
            raise RuntimeError("Keine Aufnahmesitzung gestartet.")
        exposure = "auto" if point.exposure_time_us is None else f"e{point.exposure_time_us}us"
        ramp = "" if point.ramp_sample_id is None else f"ramp-{point.ramp_sample_id:03d}_"
        robot = (
            f"ura-{point.robot_raw_value:04d}"
            if point.robot_control_mode == RobotCommandMode.ANGLE.value
            else f"ur{point.pose}"
        )
        belt = ""
        if point.conveyor_direction != "fixed":
            position_tenths = round(point.conveyor_offset_mm * 10.0)
            belt = (
                f"belt-{point.conveyor_station_id:03d}_pos-{position_tenths:04d}_"
                f"{point.conveyor_direction}_"
            )
        stem = (
            f"img_{index + 1:06d}_{robot}_{belt}{ramp}"
            f"p1-{point.light_1_brightness:03d}_p2-{point.light_2_brightness:03d}_{exposure}"
        )
        return (
            self._session_directory / f"{stem}.png",
            self._session_directory / f"{stem}.yaml",
        )

    def flush(self, timeout_seconds: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        with self._pending_condition:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Ausstehende Bilddateien wurden nicht rechtzeitig gespeichert."
                    )
                self._pending_condition.wait(remaining)

    @property
    def pending_count(self) -> int:
        with self._pending_lock:
            return self._pending

    def can_accept(self, maximum_pending: int = 8) -> bool:
        return self.pending_count < maximum_pending

    def submit(
        self,
        index: int,
        point: CapturePoint,
        frame: CameraFrame,
        metadata: dict[str, Any],
    ) -> None:
        if self._session_directory is None:
            raise RuntimeError("Keine Aufnahmesitzung gestartet.")
        image_path, yaml_path = self.expected_paths(index, point)
        if image_path.exists() or yaml_path.exists():
            raise RuntimeError(
                f"Zieldatei für Aufnahme {index + 1} existiert bereits: {image_path.name}"
            )
        token = self._session_token
        metadata = {
            **metadata,
            "image": {**metadata.get("image", {}), "file": image_path.name},
        }
        timed_capture = point.planned_offset_s is not None
        ramp_mono = timed_capture and frame.pixel_format.lower().startswith(
            "mono"
        )
        image = np.ascontiguousarray(
            frame.image[..., 0].copy()
            if ramp_mono and frame.image.ndim == 3
            else frame.image.copy()
        )
        with self._pending_lock:
            self._pending += 1
        try:
            future = self._executor.submit(
                self._write_pair,
                image_path,
                yaml_path,
                image,
                metadata,
                timed_capture,
            )
        except Exception:
            with self._pending_condition:
                self._pending -= 1
                self._pending_condition.notify_all()
            raise

        def done(result: Any) -> None:
            with self._pending_condition:
                self._pending -= 1
                self._pending_condition.notify_all()
            try:
                result.result()
                self.saved.emit(token, index, str(image_path))
            except Exception as exc:
                self.failed.emit(token, index, str(exc) or type(exc).__name__)

        future.add_done_callback(done)

    @staticmethod
    def _write_pair(
        image_path: Path,
        yaml_path: Path,
        image: np.ndarray,
        metadata: dict[str, Any],
        fast_lossless: bool = False,
    ) -> None:
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        compression = 1 if fast_lossless else 3
        success, encoded = cv2.imencode(
            ".png", encoded_image, [cv2.IMWRITE_PNG_COMPRESSION, compression]
        )
        if not success:
            raise OSError("PNG-Kodierung ist fehlgeschlagen.")
        image_tmp = image_path.with_suffix(".png.part")
        yaml_tmp = yaml_path.with_suffix(".yaml.part")
        try:
            image_tmp.write_bytes(encoded.tobytes())
            yaml_tmp.write_text(
                dump_yaml(metadata),
                encoding="utf-8",
            )
            image_tmp.replace(image_path)
            yaml_tmp.replace(yaml_path)
        finally:
            image_tmp.unlink(missing_ok=True)
            yaml_tmp.unlink(missing_ok=True)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


class AcquisitionController(QObject):
    running_changed = pyqtSignal(bool)
    resume_available_changed = pyqtSignal(bool)
    progress_changed = pyqtSignal(int, int)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(str)
    completed = pyqtSignal(str)
    alignment_required_changed = pyqtSignal(bool, float, float)

    def __init__(
        self,
        camera: CameraAdapter,
        robot: RobotAdapter,
        light_1: LightAdapter,
        light_2: LightAdapter,
        parent: QObject | None = None,
        conveyor: ConveyorAdapter | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera = camera
        self.robot = robot
        self.light_1 = light_1
        self.light_2 = light_2
        self.conveyor = conveyor
        self.writer = DatasetWriter(self)
        self._settings: AcquisitionSettings | None = None
        self._points: tuple[CapturePoint, ...] = ()
        self._index = 0
        self._checkpoint_index = 0
        self._running = False
        self._phase = "idle"
        self._deadline = 0.0
        self._current_robot_key: tuple[str, int] | None = None
        self._current_conveyor_key: tuple[int, float, str] | None = None
        self._pending_conveyor_sequence: int | None = None
        self._saved_conveyor_origin: int | None = None
        self._alignment_required = False
        self._applied_lights: tuple[int, int] | None = None
        self._applied_exposure: int | None = None
        self._frame_after = 0.0
        self._session_directory: Path | None = None
        self._writer_token = 0
        self._resume_available = False
        self._last_finish_message = ""
        self._camera_status = CameraStatus()
        self._robot_status = RobotStatus()
        self._light_statuses = [LightStatus(), LightStatus()]
        self._ramp_pass_key: (
            tuple[tuple[str, int], tuple[int, float, str], int | None] | None
        ) = None
        self._ramp_origin_monotonic = 0.0
        self._ramp_origin_wall = 0.0
        self._ramp_confirmation_marker = 0.0
        self._ramp_targets = (0, 0)
        self._ramp_sent = [False, False]
        self._ramp_actual_offset_s: float | None = None
        self._ramp_timing_error_s: float | None = None
        self._ramp_pending_writes: set[int] = set()
        self._ramp_completed_writes: set[int] = set()
        self._sweep_pass_key: tuple[tuple[str, int], int | None] | None = None
        self._sweep_duration_s = 0.0
        self._sweep_effective_speed_mm_s = 0.0
        self._sweep_origin_monotonic = 0.0
        self._sweep_origin_wall = 0.0
        self._sweep_out_completed = False
        self._sweep_return_completed = False
        self._sweep_samples_finished = False
        self._sweep_resuming_midpass = False

        self._phase_timer = QTimer(self)
        self._phase_timer.setSingleShot(True)
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(100)
        self._watchdog.timeout.connect(self._check_timeout)
        self._ramp_timer = QTimer(self)
        self._ramp_timer.setInterval(50)
        self._ramp_timer.timeout.connect(self._ramp_tick)

        camera.frame_ready.connect(self._on_frame)
        camera.status_changed.connect(self._on_camera_status)
        robot.status_changed.connect(self._on_robot_status)
        if conveyor is not None:
            conveyor.status_changed.connect(self._on_conveyor_status)
            conveyor.move_failed.connect(self._on_conveyor_move_failed)
        light_1.status_changed.connect(lambda status: self._on_light_status(0, status))
        light_2.status_changed.connect(lambda status: self._on_light_status(1, status))
        self.writer.saved.connect(self._on_saved)
        self.writer.failed.connect(self._on_write_failed)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def resume_available(self) -> bool:
        return self._resume_available and not self._running

    @property
    def remaining_count(self) -> int:
        return max(0, len(self._points) - self._checkpoint_index)

    @property
    def session_directory(self) -> Path | None:
        return self._session_directory

    @property
    def session_settings(self) -> AcquisitionSettings | None:
        return self._settings

    def start(self, settings: AcquisitionSettings) -> bool:
        if self._running:
            return False
        try:
            settings = settings.validated()
            points = build_capture_points(settings)
            self._validate_hardware(settings)
            points = self._with_conveyor_quantization(points, settings)
            session = self.writer.start_session(settings.output_directory)
        except Exception as exc:
            self.error.emit(str(exc) or type(exc).__name__)
            return False

        if self._resume_available:
            self._write_manifest("abandoned", "Durch eine neue Sitzung ersetzt.")
        self._settings = settings
        self._points = points
        self._index = 0
        self._checkpoint_index = 0
        self._running = True
        self._current_robot_key = None
        self._current_conveyor_key = None
        self._pending_conveyor_sequence = None
        self._saved_conveyor_origin = (
            self.conveyor.origin_position if settings.conveyor_enabled and self.conveyor else None
        )
        self._set_alignment_required(False)
        self._applied_lights = None
        self._applied_exposure = None
        self._reset_ramp_state()
        self._session_directory = session
        self._writer_token = self.writer.session_token
        self._set_resume_available(False)
        self._write_manifest("running", "Neue Sitzung gestartet.")
        self._watchdog.start()
        self.running_changed.emit(True)
        self.progress_changed.emit(0, len(points))
        self.status_changed.emit(f"Sitzung gestartet: {session}")
        self._set_phase("power", 8.0)
        self.light_1.set_power(True)
        self.light_2.set_power(True)
        self._maybe_power_ready()
        return True

    def stop(self) -> None:
        if not self._running:
            return
        if self.conveyor is not None:
            self.conveyor.stop_motion()
        self._finish(
            "Aufnahme gestoppt. Das Förderband wurde gestoppt; "
            "eine UR-Fahrt wird nicht abgebrochen.",
            resumable=True,
        )

    def resume(self) -> bool:
        if self._running or not self._resume_available:
            return False
        if self._settings is None or self._session_directory is None or not self._points:
            self._set_resume_available(False)
            self.error.emit("Es ist keine fortsetzbare Aufnahmesitzung geladen.")
            return False
        try:
            self.writer.flush()
            self._reconcile_saved_points()
            self._checkpoint_index = self._index
            self.progress_changed.emit(self._index, len(self._points))
            if self._index >= len(self._points):
                message = f"Aufnahme bereits vollständig: {self._session_directory}"
                self._write_manifest("completed", message)
                self._set_resume_available(False)
                self.progress_changed.emit(self._index, len(self._points))
                self.status_changed.emit(message)
                self.completed.emit(message)
                return True
            self._write_manifest(
                "interrupted",
                "Vorhandene Dateipaare vor dem Fortsetzen abgeglichen.",
            )
            self._validate_hardware(self._settings)
            self._points = self._with_conveyor_quantization(self._points, self._settings)
            if not self._validate_resume_conveyor_position():
                return False
        except Exception as exc:
            self.error.emit(str(exc) or type(exc).__name__)
            return False

        self._running = True
        self._current_robot_key = None
        self._current_conveyor_key = None
        self._pending_conveyor_sequence = None
        self._set_alignment_required(False)
        self._applied_lights = None
        self._applied_exposure = None
        self._reset_ramp_state()
        self._writer_token = self.writer.session_token
        self._set_resume_available(False)
        self._write_manifest("running", "Unterbrochene Sitzung wird fortgesetzt.")
        self._watchdog.start()
        self.running_changed.emit(True)
        self.progress_changed.emit(self._index, len(self._points))
        self.status_changed.emit(
            f"Sitzung wird bei Bild {self._index + 1}/{len(self._points)} fortgesetzt: "
            f"{self._session_directory}"
        )
        self._set_phase("power", 8.0)
        self.light_1.set_power(True)
        self.light_2.set_power(True)
        self._maybe_power_ready()
        return True

    def restore_interrupted(self, configured_settings: AcquisitionSettings) -> bool:
        if self._running:
            return False
        root = configured_settings.output_directory.expanduser().resolve()
        if not root.is_dir():
            return False
        manifests = sorted(
            root.glob("capture_*/capture_session.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for manifest in manifests:
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if payload.get("status") not in {"running", "interrupted"}:
                    continue
                settings = self._settings_from_manifest(payload["settings"])
                points = build_capture_points(settings)
                index = int(payload.get("next_index", 0))
                if not 0 <= index <= len(points):
                    raise ValueError("Ungültiger Aufnahmeindex im Sitzungs-Checkpoint.")
                self.writer.attach_session(manifest.parent)
                self._settings = settings
                self._points = points
                self._index = index
                self._checkpoint_index = index
                self._session_directory = manifest.parent
                self._writer_token = self.writer.session_token
                conveyor_checkpoint = payload.get("conveyor")
                if isinstance(conveyor_checkpoint, dict):
                    origin = conveyor_checkpoint.get("origin_internal_position")
                    self._saved_conveyor_origin = None if origin is None else int(origin)
                self._reconcile_saved_points()
                self._checkpoint_index = self._index
                if self._index >= len(self._points):
                    self._write_manifest("completed", "Alle Dateipaare sind vollständig.")
                    continue
                self._write_manifest(
                    "interrupted",
                    "Checkpoint und vorhandene Dateipaare abgeglichen.",
                )
                self._set_resume_available(True)
                self.progress_changed.emit(self._index, len(self._points))
                self.status_changed.emit(
                    f"Unterbrochene Sitzung gefunden: {self._index}/{len(self._points)} Bilder · "
                    f"{manifest.parent}"
                )
                return True
            except Exception as exc:
                self.error.emit(
                    f"Unterbrochene Sitzung konnte nicht geladen werden ({manifest}): "
                    f"{str(exc) or type(exc).__name__}"
                )
                return False
        if manifests:
            return False
        return self._restore_legacy_session(configured_settings, root)

    def close(self) -> None:
        self.stop()
        self.writer.close()

    @property
    def alignment_required(self) -> bool:
        return self._alignment_required

    @property
    def expected_resume_offset_mm(self) -> float | None:
        if not self._points or self._index >= len(self._points):
            return None
        return self._points[self._index].conveyor_offset_mm

    def _set_alignment_required(self, required: bool) -> None:
        required = bool(required)
        self._alignment_required = required
        expected = float(self.expected_resume_offset_mm or 0.0)
        current = (
            0.0
            if self.conveyor is None or self.conveyor.status.logical_offset_mm is None
            else float(self.conveyor.status.logical_offset_mm)
        )
        self.alignment_required_changed.emit(required, expected, current)

    def align_for_resume(self) -> bool:
        if not self._alignment_required or self._settings is None or self.conveyor is None:
            return False
        expected = self.expected_resume_offset_mm
        if expected is None:
            return False
        sequence = self.conveyor.request_offset(expected, self._settings.conveyor_speed_mm_per_s)
        if sequence is None:
            return False
        if sequence == 0:
            self._set_alignment_required(False)
            self.status_changed.emit("Förderband steht wieder an der erwarteten Position.")
        else:
            self._pending_conveyor_sequence = sequence
            self.status_changed.emit(
                f"Korrigiere Förderbandposition auf {expected:g} mm (Fahrt #{sequence}) …"
            )
        return True

    def _validate_resume_conveyor_position(self) -> bool:
        if self._settings is None or not self._settings.conveyor_enabled:
            return True
        if self.conveyor is None or self._saved_conveyor_origin is None:
            raise RuntimeError("Der Förderband-Nullpunkt fehlt im Sitzungs-Checkpoint.")
        self.conveyor.restore_origin(self._saved_conveyor_origin)
        expected = self.expected_resume_offset_mm
        if expected is None or self.conveyor.position_matches(expected):
            self._set_alignment_required(False)
            return True
        self._set_alignment_required(True)
        actual = self.conveyor.status.logical_offset_mm
        self.status_changed.emit(
            f"Fortsetzen blockiert: Förderband Soll {expected:g} mm, "
            f"Ist {'unbekannt' if actual is None else f'{actual:.3f} mm'}."
        )
        return False

    def _set_resume_available(self, available: bool) -> None:
        available = bool(available)
        if available == self._resume_available:
            return
        self._resume_available = available
        self.resume_available_changed.emit(available)

    @staticmethod
    def _settings_from_manifest(payload: object) -> AcquisitionSettings:
        if not isinstance(payload, dict):
            raise ValueError("Aufnahmeeinstellungen im Checkpoint sind ungültig.")
        defaults = AcquisitionSettings(output_directory=Path("."))
        names = set(AcquisitionSettings.__dataclass_fields__)
        values = {name: payload.get(name, getattr(defaults, name)) for name in names}
        # Manifests written before milestone 3 always used fixed poses and no conveyor.
        if "robot_control_mode" not in payload:
            values["robot_control_mode"] = RobotCommandMode.POSE_ID.value
        if "conveyor_enabled" not in payload:
            values["conveyor_enabled"] = False
        values["output_directory"] = Path(str(values["output_directory"]))
        return AcquisitionSettings(**values).validated()

    def _write_manifest(self, status: str, message: str) -> None:
        if self._settings is None or self._session_directory is None:
            return
        settings = asdict(self._settings)
        settings["output_directory"] = str(self._settings.output_directory)
        payload = {
            "schema_version": 2,
            "status": status,
            "next_index": self._checkpoint_index,
            "total": len(self._points),
            "settings": settings,
            "message": message,
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        if self._settings.conveyor_enabled:
            expected = (
                None
                if not self._points or self._checkpoint_index >= len(self._points)
                else self._points[self._checkpoint_index].conveyor_offset_mm
            )
            payload["conveyor"] = {
                "origin_internal_position": self._saved_conveyor_origin,
                "expected_next_offset_mm": expected,
                "last_confirmed_station_id": (
                    None
                    if self._current_conveyor_key is None
                    else self._current_conveyor_key[0]
                ),
            }
        destination = self._session_directory / "capture_session.json"
        temporary = self._session_directory / "capture_session.json.part"
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            self.error.emit(f"Sitzungs-Checkpoint konnte nicht gespeichert werden: {exc}")

    def _reconcile_saved_points(self) -> None:
        while self._index < len(self._points):
            image_path, yaml_path = self.writer.expected_paths(
                self._index,
                self._points[self._index],
            )
            image_exists = image_path.is_file()
            yaml_exists = yaml_path.is_file()
            if image_exists and yaml_exists:
                self._index += 1
                continue
            if image_exists != yaml_exists:
                raise RuntimeError(
                    "Unvollständiges Dateipaar verhindert sicheres Fortsetzen: "
                    f"{image_path.name} / {yaml_path.name}"
                )
            break

    def _restore_legacy_session(
        self,
        configured_settings: AcquisitionSettings,
        root: Path,
    ) -> bool:
        sessions = sorted(
            (path for path in root.glob("capture_*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not sessions:
            return False
        session = sessions[0]
        if (session / "capture_session.json").exists() or list(session.glob("*.part")):
            return False
        try:
            settings = configured_settings.validated()
            points = build_capture_points(settings)
            self.writer.attach_session(session)
            self._settings = settings
            self._points = points
            self._index = 0
            self._session_directory = session
            self._writer_token = self.writer.session_token
            self._reconcile_saved_points()
            self._checkpoint_index = self._index
            expected_images: set[Path] = set()
            expected_metadata: set[Path] = set()
            for index in range(self._index):
                image_path, yaml_path = self.writer.expected_paths(index, points[index])
                expected_images.add(image_path)
                expected_metadata.add(yaml_path)
            actual_images = set(session.glob("img_*.png"))
            actual_metadata = set(session.glob("img_*.yaml"))
            if actual_images != expected_images or actual_metadata != expected_metadata:
                return False
            if self._index >= len(points):
                return False
            self._write_manifest(
                "interrupted",
                "Fortschritt einer älteren Sitzung aus vorhandenen Dateipaaren rekonstruiert.",
            )
            self._set_resume_available(True)
            self.progress_changed.emit(self._index, len(points))
            self.status_changed.emit(
                f"Unterbrochene ältere Sitzung rekonstruiert: "
                f"{self._index}/{len(points)} Bilder · {session}"
            )
            return True
        except Exception:
            return False

    def _validate_hardware(self, settings: AcquisitionSettings) -> None:
        failed = [check for check in self.preflight_checks(settings) if not check.ready]
        if failed:
            details = "\n".join(f"• {check.label}: {check.detail}" for check in failed)
            raise RuntimeError(f"Startfreigabe fehlt:\n{details}")

    def preflight_checks(self, settings: AcquisitionSettings) -> tuple[PreflightCheck, ...]:
        checks: list[PreflightCheck] = []
        try:
            settings = settings.validated()
            image_count = len(build_capture_points(settings))
            checks.append(
                PreflightCheck(
                    "configuration",
                    "Aufnahmekonfiguration",
                    image_count > 0,
                    f"gültig, {image_count} Bilder geplant",
                )
            )
        except Exception as exc:
            checks.append(
                PreflightCheck(
                    "configuration",
                    "Aufnahmekonfiguration",
                    False,
                    str(exc) or type(exc).__name__,
                )
            )
            return tuple(checks)

        output = settings.output_directory.expanduser()
        probe = output
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        output_ready = (not output.exists() or output.is_dir()) and probe.is_dir()
        output_ready = output_ready and os.access(probe, os.W_OK)
        checks.append(
            PreflightCheck(
                "output",
                "Speicherort",
                output_ready,
                str(output) if output_ready else f"nicht beschreibbar: {output}",
            )
        )

        camera_ready = self.camera.state is ConnectionState.CONNECTED
        checks.append(
            PreflightCheck(
                "camera",
                "Kamera",
                camera_ready,
                (
                    f"verbunden ({self._camera_status.model or 'Modell unbekannt'})"
                    if camera_ready
                    else f"Status {self.camera.state.value}"
                ),
            )
        )

        if settings.capture_mode == "ramp" or (
            settings.conveyor_enabled
            and settings.conveyor_motion_mode == "synchronized"
        ):
            configured_fps = float(
                getattr(getattr(self.camera, "config", None), "preview_max_fps", 0)
            )
            observed_fps = float(self._camera_status.preview_fps or 0.0)
            target_fps = float(settings.ramp_image_rate_fps)
            fps_ready = configured_fps >= target_fps and (
                observed_fps <= 0 or observed_fps + 0.25 >= target_fps
            )
            checks.append(
                PreflightCheck(
                    "camera_fps",
                    "Kamera-Bildrate",
                    fps_ready,
                    (
                        f"Soll {target_fps:g}, Vorschau {observed_fps:.1f}, "
                        f"Limit {configured_fps:g} FPS"
                    ),
                )
            )
        if settings.exposure_enabled:
            minimum = self._camera_status.exposure_min_us
            maximum = self._camera_status.exposure_max_us
            low = min(settings.exposure_start_us, settings.exposure_end_us)
            high = max(settings.exposure_start_us, settings.exposure_end_us)
            exposure_ready = self._camera_status.exposure_writable
            exposure_ready = exposure_ready and (minimum is None or low >= minimum)
            exposure_ready = exposure_ready and (maximum is None or high <= maximum)
            limits = (
                "unbekannt"
                if minimum is None or maximum is None
                else f"{minimum:.0f}–{maximum:.0f} µs"
            )
            checks.append(
                PreflightCheck(
                    "exposure",
                    "Kamera-Belichtung",
                    exposure_ready,
                    f"gewählt {low}–{high} µs, Kamerabereich {limits}",
                )
            )

        for key, label, adapter in (
            ("light_1", "Licht 1", self.light_1),
            ("light_2", "Licht 2", self.light_2),
        ):
            ready = adapter.state is ConnectionState.CONNECTED
            checks.append(
                PreflightCheck(
                    key,
                    label,
                    ready,
                    "verbunden" if ready else f"Status {adapter.state.value}",
                )
            )

        robot = self._robot_status
        rtde_ready = robot.rtde_connected and robot.command_channel_connected
        checks.append(
            PreflightCheck(
                "robot_rtde",
                "UR RTDE-Handshake",
                rtde_ready,
                (
                    "Receive- und Registerkanal verbunden"
                    if rtde_ready
                    else "Receive- oder Registerkanal fehlt"
                ),
            )
        )
        mode_ready = robot.robot_mode.upper() == "RUNNING" and robot.safety_mode.upper() in {
            "NORMAL",
            "REDUCED",
        }
        checks.append(
            PreflightCheck(
                "robot_state",
                "UR Betriebszustand",
                mode_ready,
                f"Robot {robot.robot_mode}, Safety {robot.safety_mode}",
            )
        )
        loaded = robot.loaded_program.upper()
        expected_program = (
            "BIBAZU_CONTINUOUS"
            if settings.robot_control_mode == RobotCommandMode.ANGLE.value
            else "BIBAZU_GUI"
        )
        program_ready = (
            "PLAYING" in robot.program_state.upper()
            and expected_program in loaded
            and robot.command_state_code in {1, 3, -1}
        )
        checks.append(
            PreflightCheck(
                "robot_program",
                "UR-Programm",
                program_ready,
                (
                    f"erwartet {expected_program}, geladen {robot.loaded_program}, "
                    f"Status {robot.program_state}, Handshake {robot.command_state_code}"
                ),
            )
        )

        if settings.conveyor_enabled:
            conveyor_connected = (
                self.conveyor is not None
                and self.conveyor.state is ConnectionState.CONNECTED
            )
            checks.append(
                PreflightCheck(
                    "conveyor_ads",
                    "Förderband ADS",
                    conveyor_connected,
                    "verbunden" if conveyor_connected else "nicht verbunden",
                )
            )
            conveyor = self.conveyor.status if self.conveyor is not None else ConveyorStatus()
            if settings.conveyor_motion_mode == "synchronized":
                position_ready = (
                    conveyor_connected
                    and conveyor.internal_position is not None
                    and conveyor.position_feedback_verified
                )
                checks.append(
                    PreflightCheck(
                        "conveyor_position_feedback",
                        "Förderband-Positionsrückmeldung",
                        position_ready,
                        (
                            "SPS-Positionswert durch eine Positionsänderung bestätigt"
                            if position_ready
                            else "Bitte nach dem PDO-Mapping eine 1-mm-Testfahrt ausführen; "
                            "MAIN.StepperInternalPosition muss sich dabei ändern"
                        ),
                    )
                )
            calibration_ready = (
                conveyor_connected
                and conveyor.calibration_valid
                and conveyor.mm_per_full_step > 0.0
            )
            checks.append(
                PreflightCheck(
                    "conveyor_calibration",
                    "Förderbandkalibrierung",
                    calibration_ready,
                    (
                        f"{conveyor.mm_per_full_step:.6f} mm/Vollschritt"
                        if calibration_ready
                        else "in der SPS ungültig oder nicht lesbar"
                    ),
                )
            )
            drive_ready = (
                conveyor_connected
                and not conveyor.busy
                and not conveyor.error
                and conveyor.status_code not in {4, 5}
            )
            if conveyor.ready_to_execute:
                drive_detail = "bereit für Fahrauftrag"
            elif drive_ready:
                drive_detail = (
                    "Standby – der Positioniermodus wird vor der ersten Fahrt aktiviert"
                )
            else:
                drive_detail = (
                    f"Ready {conveyor.ready_to_execute}, Busy {conveyor.busy}, "
                    f"Fehler {conveyor.error}, SPS-Status {conveyor.status_code}"
                )
            checks.append(
                PreflightCheck(
                    "conveyor_drive",
                    "Förderbandantrieb",
                    drive_ready,
                    drive_detail,
                )
            )
            direction = (
                ""
                if self.conveyor is None
                else self.conveyor.config.conveyor_forward_direction
            )
            checks.append(
                PreflightCheck(
                    "conveyor_direction",
                    "Förderbandrichtung",
                    direction in {"left", "right"},
                    (
                        f"Vorwärts = {'Links' if direction == 'left' else 'Rechts'}"
                        if direction in {"left", "right"}
                        else "Links/Rechts als Vorwärtsrichtung bestätigen"
                    ),
                )
            )
            origin = None if self.conveyor is None else self.conveyor.origin_position
            if origin is None:
                origin = self._saved_conveyor_origin
            checks.append(
                PreflightCheck(
                    "conveyor_origin",
                    "Förderband-Nullpunkt",
                    origin is not None,
                    (
                        f"SPS-Position {origin} als 0 mm gesetzt"
                        if origin is not None
                        else "Bauteil hinten platzieren und ‚Aktuelle Position = 0 mm‘ drücken"
                    ),
                )
            )

        return tuple(checks)

    def _with_conveyor_quantization(
        self,
        points: tuple[CapturePoint, ...],
        settings: AcquisitionSettings,
    ) -> tuple[CapturePoint, ...]:
        if not settings.conveyor_enabled or self.conveyor is None:
            return points
        calibration = self.conveyor.status.mm_per_full_step
        if calibration <= 0.0:
            return points
        return tuple(
            replace(
                point,
                conveyor_actual_offset_mm=(
                    int(point.conveyor_offset_mm / calibration + 0.5) * calibration
                ),
            )
            for point in points
        )

    def _set_phase(self, phase: str, timeout_seconds: float) -> None:
        self._phase = phase
        self._deadline = time.monotonic() + timeout_seconds

    def _schedule(self, delay_ms: int, callback: Any, phase: str) -> None:
        self._set_phase(phase, max(5.0, delay_ms / 1000.0 + 2.0))
        try:
            self._phase_timer.timeout.disconnect()
        except TypeError:
            pass
        self._phase_timer.timeout.connect(callback)
        self._phase_timer.start(delay_ms)

    def _advance(self) -> None:
        if not self._running:
            return
        if self._index >= len(self._points):
            if self._ramp_pending_writes:
                self._set_phase("ramp_draining", 30.0)
                self.status_changed.emit(
                    f"Aufnahmen vollständig; warte auf "
                    f"{len(self._ramp_pending_writes)} Schreibvorgänge …"
                )
                return
            directory = str(self._session_directory or "")
            self._finish(f"Aufnahme abgeschlossen: {directory}", completed=True)
            return
        point = self._points[self._index]
        if self._current_robot_key != point.robot_key:
            if point.robot_control_mode == RobotCommandMode.ANGLE.value:
                target_text = f"UR-Winkel {point.robot_raw_value / 10.0:.1f} Grad"
            else:
                target_text = f"UR-Pose {point.pose}"
            self.status_changed.emit(f"Fahre {target_text} …")
            self._set_phase("robot", 45.0)
            requested = (
                self.robot.request_angle(point.robot_raw_value / 10.0)
                if point.robot_control_mode == RobotCommandMode.ANGLE.value
                else self.robot.request_pose(point.pose)
            )
            if not requested:
                self._fail("UR-Ziel konnte nicht angefordert werden.")
            return
        if (
            self._settings is not None
            and self._settings.conveyor_enabled
            and self._settings.conveyor_motion_mode == "stations"
        ):
            if self._current_conveyor_key != point.conveyor_key:
                self._move_conveyor(point)
                return
        self._prepare_point()

    def _move_conveyor(self, point: CapturePoint) -> None:
        if self.conveyor is None or self._settings is None:
            self._fail("Förderbandadapter ist nicht verfügbar.")
            return
        self.status_changed.emit(
            f"Fahre Förderband auf {point.conveyor_offset_mm:g} mm "
            f"({point.conveyor_direction}) …"
        )
        current = self.conveyor.status.logical_offset_mm
        distance = abs(point.conveyor_offset_mm - float(current or 0.0))
        timeout = distance / self._settings.conveyor_speed_mm_per_s + 5.0
        self._set_phase("conveyor", max(5.0, timeout))
        sequence = self.conveyor.request_offset(
            point.conveyor_offset_mm, self._settings.conveyor_speed_mm_per_s
        )
        if sequence is None:
            self._fail("Förderbandfahrt konnte nicht angefordert werden.")
        elif sequence == 0:
            self._current_conveyor_key = point.conveyor_key
            self._schedule(
                self._settings.conveyor_settle_ms,
                self._prepare_point,
                "conveyor_settle",
            )
        else:
            self._pending_conveyor_sequence = sequence

    def _prepare_point(self) -> None:
        point = self._points[self._index]
        if self._synchronized_conveyor():
            self._apply_exposure()
            return
        if point.ramp_sample_id is not None:
            self._apply_exposure()
            return
        target_lights = (point.light_1_brightness, point.light_2_brightness)
        if self._applied_lights != target_lights:
            self.status_changed.emit(
                f"Setze Panelhelligkeiten {target_lights[0]} % / {target_lights[1]} % …"
            )
            self._set_phase("lights", 8.0)
            self.light_1.set_brightness(target_lights[0])
            self.light_2.set_brightness(target_lights[1])
            self._maybe_lights_ready()
            return
        self._apply_exposure()

    def _apply_exposure(self) -> None:
        point = self._points[self._index]
        if point.exposure_time_us is None or self._applied_exposure == point.exposure_time_us:
            self._after_exposure_ready()
            return
        self.status_changed.emit(f"Setze Belichtungszeit {point.exposure_time_us} µs …")
        self._set_phase("exposure", 5.0)
        if not self.camera.set_exposure_time(point.exposure_time_us):
            self._fail("Belichtungszeit konnte nicht gesetzt werden.")

    def _after_exposure_ready(self) -> None:
        point = self._points[self._index]
        if self._synchronized_conveyor():
            key = self._sweep_key(point)
            if self._sweep_pass_key == key:
                self._wait_for_sweep_frame()
            else:
                self._start_sweep_sync(point)
            return
        if point.ramp_sample_id is not None:
            key = (point.robot_key, point.conveyor_key, point.exposure_time_us)
            if self._ramp_pass_key == key:
                self._wait_for_ramp_frame()
            else:
                self._start_ramp_sync(point)
            return
        self._wait_for_frame()

    def _wait_for_frame(self) -> None:
        self._frame_after = time.time()
        # The camera worker can recover a temporarily stalled GenTL stream twice.
        # Keep the capture point alive long enough for that recovery to complete.
        self._set_phase("frame", 20.0)
        self.status_changed.emit(
            f"Warte auf frisches Kamerabild ({self._index + 1}/{len(self._points)}) …"
        )

    def _on_camera_status(self, status: object) -> None:
        if not isinstance(status, CameraStatus):
            return
        self._camera_status = replace(status)
        if not self._running or self._phase != "exposure":
            return
        target = self._points[self._index].exposure_time_us
        if target is not None and status.exposure_time_us is not None:
            if abs(status.exposure_time_us - target) <= max(1.0, target * 0.001):
                self._applied_exposure = target
                assert self._settings is not None
                self._schedule(
                    self._settings.camera_settle_ms,
                    self._after_exposure_ready,
                    "exposure_settle",
                )

    def _on_robot_status(self, status: object) -> None:
        if not isinstance(status, RobotStatus):
            return
        self._robot_status = replace(status)
        if not self._running or self._phase != "robot":
            return
        point = self._points[self._index]
        acknowledged_target = (
            status.acknowledged_raw_value == point.robot_raw_value
            if point.robot_control_mode == RobotCommandMode.ANGLE.value
            else status.acknowledged_pose in {None, point.pose}
        )
        if (
            status.command_state_code == 3
            and not status.command_pending
            and status.requested_sequence is not None
            and status.acknowledged_sequence == status.requested_sequence
            and acknowledged_target
        ):
            self._current_robot_key = point.robot_key
            assert self._settings is not None
            self._schedule(
                self._settings.robot_settle_ms,
                self._advance,
                "robot_settle",
            )

    def _on_conveyor_status(self, status: object) -> None:
        if not isinstance(status, ConveyorStatus):
            return
        if self._alignment_required and self._pending_conveyor_sequence is not None:
            if status.error or status.status_code in {4, 5}:
                self._pending_conveyor_sequence = None
                self.error.emit("Korrekturfahrt des Förderbands ist fehlgeschlagen.")
            elif status.completed_sequence == self._pending_conveyor_sequence:
                self._pending_conveyor_sequence = None
                expected = self.expected_resume_offset_mm
                if expected is not None and self.conveyor is not None:
                    if self.conveyor.position_matches(expected):
                        self._set_alignment_required(False)
                        self.status_changed.emit(
                            "Förderbandposition korrigiert; die Aufnahme kann fortgesetzt werden."
                        )
            return
        if self._running and self._synchronized_conveyor():
            if status.error or status.status_code in {4, 5}:
                self._fail(f"Förderbandfehler (SPS-Status {status.status_code}).")
                return
            sequence = self._pending_conveyor_sequence
            if sequence is not None and status.movement_started_sequence == sequence:
                if self._phase == "sweep_out_start" and status.movement_started_at:
                    self._begin_sweep_timeline(status, "out")
                elif self._phase == "sweep_back_start" and status.movement_started_at:
                    self._begin_sweep_timeline(status, "back")
            if sequence is not None and status.completed_sequence == sequence:
                direction = "out" if not self._sweep_out_completed else "back"
                target = (
                    self._settings.conveyor_max_offset_mm
                    if direction == "out" and self._settings is not None
                    else 0.0
                )
                if self.conveyor is None or not self.conveyor.position_matches(target):
                    self._fail(
                        "Synchronisierte Förderbandfahrt quittiert, "
                        "Zielposition stimmt jedoch nicht überein."
                    )
                    return
                self._pending_conveyor_sequence = None
                if direction == "out":
                    self._sweep_out_completed = True
                    self._wait_for_sweep_frame()
                else:
                    self._sweep_return_completed = True
                    if self._sweep_samples_finished:
                        self._finish_sweep_pass()
                    else:
                        self._wait_for_sweep_frame()
            return
        if not self._running or self._phase != "conveyor":
            return
        if status.error or status.status_code in {4, 5}:
            self._fail(f"Förderbandfehler (SPS-Status {status.status_code}).")
            return
        if (
            self._pending_conveyor_sequence is not None
            and status.completed_sequence == self._pending_conveyor_sequence
        ):
            point = self._points[self._index]
            if self.conveyor is None or not self.conveyor.position_matches(
                point.conveyor_offset_mm
            ):
                self._fail("Förderbandfahrt quittiert, Zielposition stimmt jedoch nicht überein.")
                return
            self._pending_conveyor_sequence = None
            self._current_conveyor_key = point.conveyor_key
            assert self._settings is not None
            self._schedule(
                self._settings.conveyor_settle_ms,
                self._prepare_point,
                "conveyor_settle",
            )

    def _on_conveyor_move_failed(self, sequence: int, message: str) -> None:
        if self._pending_conveyor_sequence != sequence:
            return
        self._pending_conveyor_sequence = None
        if self._alignment_required:
            self.error.emit(f"Korrekturfahrt fehlgeschlagen: {message}")
            return
        if self._running:
            self._fail(message)

    def _on_light_status(self, index: int, status: object) -> None:
        if not isinstance(status, LightStatus):
            return
        self._light_statuses[index] = replace(status)
        self._maybe_power_ready()
        self._maybe_lights_ready()
        self._maybe_ramp_transition_ready()

    def _maybe_power_ready(self) -> None:
        if not self._running or self._phase != "power":
            return
        if all(status.connected and status.power is True for status in self._light_statuses):
            self._schedule(150, self._advance, "power_settle")

    def _maybe_lights_ready(self) -> None:
        if not self._running or self._phase != "lights":
            return
        point = self._points[self._index]
        targets = (point.light_1_brightness, point.light_2_brightness)
        ready = all(
            status.connected
            and status.values_are_confirmed_commands
            and status.brightness == target
            for status, target in zip(self._light_statuses, targets, strict=True)
        )
        if ready:
            self._applied_lights = targets
            assert self._settings is not None
            self._schedule(
                self._settings.light_settle_ms,
                self._apply_exposure,
                "light_settle",
            )

    def _reset_ramp_state(self) -> None:
        self._ramp_timer.stop()
        self._ramp_pass_key = None
        self._ramp_origin_monotonic = 0.0
        self._ramp_origin_wall = 0.0
        self._ramp_confirmation_marker = 0.0
        self._ramp_targets = (0, 0)
        self._ramp_sent = [False, False]
        self._ramp_actual_offset_s = None
        self._ramp_timing_error_s = None
        self._ramp_pending_writes.clear()
        self._ramp_completed_writes.clear()
        self._sweep_pass_key = None
        self._sweep_duration_s = 0.0
        self._sweep_effective_speed_mm_s = 0.0
        self._sweep_origin_monotonic = 0.0
        self._sweep_origin_wall = 0.0
        self._sweep_out_completed = False
        self._sweep_return_completed = False
        self._sweep_samples_finished = False
        self._sweep_resuming_midpass = False

    def _synchronized_conveyor(self) -> bool:
        return bool(
            self._settings is not None
            and self._settings.conveyor_enabled
            and self._settings.conveyor_motion_mode == "synchronized"
        )

    @staticmethod
    def _sweep_key(point: CapturePoint) -> tuple[tuple[str, int], int | None]:
        return point.robot_key, point.exposure_time_us

    def _start_sweep_sync(self, point: CapturePoint) -> None:
        assert self._settings is not None
        self._sweep_duration_s, self._sweep_effective_speed_mm_s, _ = (
            synchronized_sweep_profile(self._settings)
        )
        self._sweep_pass_key = self._sweep_key(point)
        self._sweep_out_completed = point.conveyor_direction == "back"
        self._sweep_return_completed = False
        self._sweep_samples_finished = False
        self._sweep_resuming_midpass = point.conveyor_station_id != 0
        self._ramp_targets = (point.light_1_brightness, point.light_2_brightness)
        self._ramp_confirmation_marker = time.time()
        self._ramp_sent = [False, False]
        self._set_phase("sweep_sync", RAMP_BLE_COMMAND_TIMEOUT_SECONDS)
        self.status_changed.emit(
            f"Synchronisiere Licht für Band-Sweep bei Sample "
            f"{point.conveyor_station_id + 1}: {self._ramp_targets[0]} % / "
            f"{self._ramp_targets[1]} % …"
        )
        self._ramp_timer.start()
        self._ramp_tick()

    def _request_sweep_leg(self, direction: str) -> None:
        if self.conveyor is None or self._settings is None:
            self._fail("Förderbandadapter ist nicht verfügbar.")
            return
        target = self._settings.conveyor_max_offset_mm if direction == "out" else 0.0
        sequence = self.conveyor.request_offset(target, self._sweep_effective_speed_mm_s)
        if sequence is None:
            self._fail("Synchronisierte Förderbandfahrt konnte nicht angefordert werden.")
            return
        if sequence == 0:
            # This is normally only possible when resuming exactly at a reversal point.
            if direction == "out":
                self._sweep_out_completed = True
                self._request_sweep_leg("back")
            else:
                self._sweep_return_completed = True
                if self._sweep_samples_finished:
                    self._finish_sweep_pass()
                else:
                    self._wait_for_sweep_frame()
            return
        self._pending_conveyor_sequence = sequence
        remaining = abs(
            target - float(self.conveyor.status.logical_offset_mm or 0.0)
        )
        timeout = remaining / max(0.1, self._sweep_effective_speed_mm_s) + 7.0
        self._set_phase(f"sweep_{direction}_start", max(7.0, timeout))
        self.status_changed.emit(
            f"Synchronisierte Fahrt {direction}: Ziel {target:g} mm mit "
            f"{self._sweep_effective_speed_mm_s:.2f} mm/s …"
        )

    def _begin_sweep_timeline(self, status: ConveyorStatus, direction: str) -> None:
        point = self._points[self._index]
        half = self._sweep_duration_s / 2.0
        if self._sweep_resuming_midpass:
            elapsed_at_start = float(point.planned_offset_s or 0.0)
        else:
            elapsed_at_start = 0.0 if direction == "out" else half
        started_wall = status.movement_started_at or time.time()
        now_wall = time.time()
        self._sweep_origin_wall = started_wall - elapsed_at_start
        self._sweep_origin_monotonic = time.monotonic() - (
            now_wall - self._sweep_origin_wall
        )
        self._sweep_resuming_midpass = False
        self._wait_for_sweep_frame()

    def _sweep_elapsed(self) -> float:
        if self._sweep_origin_monotonic <= 0.0:
            if self._phase == "sweep_back_start":
                return self._sweep_duration_s / 2.0
            return 0.0
        return max(0.0, time.monotonic() - self._sweep_origin_monotonic)

    def _wait_for_sweep_frame(self) -> None:
        if not self._running or self._index >= len(self._points):
            return
        point = self._points[self._index]
        if self._sweep_key(point) != self._sweep_pass_key:
            return
        if point.conveyor_direction == "back" and self._phase != "sweep_back_start":
            if not self._sweep_out_completed:
                self._set_phase("sweep_wait_out", 5.0)
                return
            if not self._sweep_return_completed and (
                self._pending_conveyor_sequence is None
                or self._phase not in {"sweep_back_start", "sweep_frame"}
            ):
                self._sweep_origin_monotonic = 0.0
                self._sweep_origin_wall = 0.0
                self._request_sweep_leg("back")
                return
        self._set_phase("sweep_frame", 2.0)
        self.status_changed.emit(
            f"Synchronisierte Aufnahme: UR {point.robot_raw_value} · "
            f"Band {point.conveyor_direction} {point.conveyor_offset_mm:.1f} mm · "
            f"Sample {point.conveyor_station_id + 1}"
        )

    def _finish_sweep_pass(self) -> None:
        self._ramp_targets = (0, 0)
        self._ramp_confirmation_marker = time.time()
        self._ramp_sent = [False, False]
        self._set_phase("ramp_finalize", RAMP_BLE_COMMAND_TIMEOUT_SECONDS)
        self._ramp_tick()

    def _start_ramp_sync(self, point: CapturePoint) -> None:
        # A fresh pass begins at sample zero (0/0). A resumed pass is explicitly
        # synchronized to its next missing deterministic sample.
        self._ramp_targets = (point.light_1_brightness, point.light_2_brightness)
        self._ramp_confirmation_marker = time.time()
        self._ramp_sent = [False, False]
        self._set_phase("ramp_sync", RAMP_BLE_COMMAND_TIMEOUT_SECONDS)
        self.status_changed.emit(
            f"Synchronisiere Licht-Rampe bei Sample {point.ramp_sample_id}: "
            f"{self._ramp_targets[0]} % / {self._ramp_targets[1]} % …"
        )
        self._ramp_timer.start()
        self._ramp_tick()

    def _ramp_tick(self) -> None:
        if not self._running:
            self._ramp_timer.stop()
            return
        if self._phase in {"ramp_sync", "ramp_finalize", "sweep_sync"}:
            for index, (adapter, target) in enumerate(
                zip((self.light_1, self.light_2), self._ramp_targets, strict=True)
            ):
                if not self._ramp_sent[index]:
                    self._ramp_sent[index] = adapter.try_set_ramp_brightness(target)
            self._maybe_ramp_transition_ready()
            return
        if self._sweep_pass_key is not None and self._phase in {
            "sweep_frame",
            "sweep_wait_out",
            "sweep_back_start",
            "sweep_wait_return",
        }:
            assert self._settings is not None
            point = self._points[self._index] if self._index < len(self._points) else None
            elapsed = self._sweep_elapsed()
            if self._settings.capture_mode == "ramp":
                scale = self._sweep_duration_s / self._settings.ramp_duration_s
                targets = (
                    triangle_brightness(
                        elapsed, self._settings.ramp_light_1_period_s * scale
                    ),
                    triangle_brightness(
                        elapsed, self._settings.ramp_light_2_period_s * scale
                    ),
                )
            elif point is not None and self._sweep_key(point) == self._sweep_pass_key:
                targets = (point.light_1_brightness, point.light_2_brightness)
            else:
                targets = self._ramp_targets
            self._send_timed_light_targets(targets)
            return
        if self._ramp_pass_key is None or self._phase not in {"ramp_frame", "ramp_writing"}:
            return
        assert self._settings is not None
        elapsed = max(0.0, time.monotonic() - self._ramp_origin_monotonic)
        targets = (
            triangle_brightness(elapsed, self._settings.ramp_light_1_period_s),
            triangle_brightness(elapsed, self._settings.ramp_light_2_period_s),
        )
        self._send_timed_light_targets(targets)

    def _send_timed_light_targets(self, targets: tuple[int, int]) -> None:
        for panel_number, (adapter, status, target) in enumerate(
            zip((self.light_1, self.light_2), self._light_statuses, targets, strict=True),
            start=1,
        ):
            if not status.connected:
                self._fail("Ein Panel hat während der Licht-Rampe die Verbindung verloren.")
                return
            if ramp_command_timed_out(adapter):
                self._fail(
                    f"Panel {panel_number} hat einen Rampenbefehl länger als "
                    f"{RAMP_BLE_COMMAND_TIMEOUT_SECONDS:g} Sekunden nicht bestätigt."
                )
                return
            if status.brightness != target:
                adapter.try_set_ramp_brightness(target)

    def _maybe_ramp_transition_ready(self) -> None:
        if not self._running or self._phase not in {
            "ramp_sync",
            "ramp_finalize",
            "sweep_sync",
        }:
            return
        ready = all(
            sent
            and status.connected
            and status.values_are_confirmed_commands
            and status.brightness == target
            and (status.last_command_confirmed_at or 0.0) >= self._ramp_confirmation_marker
            for sent, status, target in zip(
                self._ramp_sent, self._light_statuses, self._ramp_targets, strict=True
            )
        )
        if not ready:
            return
        if self._phase == "ramp_finalize":
            self._ramp_timer.stop()
            self._ramp_pass_key = None
            self._sweep_pass_key = None
            self._applied_lights = (0, 0)
            self._pending_conveyor_sequence = None
            self.status_changed.emit("Lichtvariation beendet; beide Panels bestätigt auf 0 %.")
            self._advance()
            return
        if self._phase == "sweep_sync":
            point = self._points[self._index]
            self._request_sweep_leg(point.conveyor_direction)
            return
        point = self._points[self._index]
        planned = float(point.planned_offset_s or 0.0)
        now_monotonic = time.monotonic()
        now_wall = time.time()
        self._ramp_origin_monotonic = now_monotonic - planned
        self._ramp_origin_wall = now_wall - planned
        self._ramp_pass_key = (point.robot_key, point.conveyor_key, point.exposure_time_us)
        self.status_changed.emit(
            f"Licht-Rampe gestartet: UR {point.robot_raw_value}, "
            f"{self._settings.ramp_duration_s:g} s bei "
            f"{self._settings.ramp_image_rate_fps} Bildern/s."
        )
        self._wait_for_ramp_frame()

    def _wait_for_ramp_frame(self) -> None:
        point = self._points[self._index]
        assert point.ramp_sample_id is not None
        self._set_phase("ramp_frame", 2.0)
        self.status_changed.emit(
            f"Rampe: Pose {point.pose} · Sample {point.ramp_sample_id + 1}/"
            f"{round(self._settings.ramp_duration_s * self._settings.ramp_image_rate_fps)} · "
            f"bestätigt {self._light_statuses[0].brightness} % / "
            f"{self._light_statuses[1].brightness} %"
        )

    def _finish_ramp_pass(self) -> None:
        self._ramp_targets = (0, 0)
        self._ramp_confirmation_marker = time.time()
        self._ramp_sent = [False, False]
        self._set_phase("ramp_finalize", RAMP_BLE_COMMAND_TIMEOUT_SECONDS)
        self._ramp_tick()

    def _on_frame(self, frame: object) -> None:
        if self._running and self._phase == "sweep_frame" and isinstance(frame, CameraFrame):
            point = self._points[self._index]
            planned = float(point.planned_offset_s or 0.0)
            actual = frame.timestamp - self._sweep_origin_wall
            if actual < planned:
                return
            timing_error = actual - planned
            if timing_error > SYNCHRONIZED_FRAME_LATE_TOLERANCE_SECONDS:
                self._fail(
                    f"Kamerabild für Band-Sample {point.conveyor_station_id} kam "
                    f"{timing_error * 1000:.0f} ms zu spät "
                    f"(maximal {SYNCHRONIZED_FRAME_LATE_TOLERANCE_SECONDS * 1000:.0f} ms; "
                    f"Vorschau {self._camera_status.preview_fps:.1f} FPS)."
                )
                return
            conveyor_status = (
                self.conveyor.status if self.conveyor is not None else ConveyorStatus()
            )
            if conveyor_status.logical_offset_mm is None:
                self._fail(
                    "Die aktuelle Förderbandposition ist während der Fahrt nicht lesbar. "
                    "Bitte das EL7047-PDO 'Actual position' mit "
                    "MAIN.StepperInternalPosition verknüpfen."
                )
                return
            self._ramp_actual_offset_s = actual
            self._ramp_timing_error_s = timing_error
            metadata = self._metadata(point, frame)
            if not self.writer.can_accept():
                self._fail(
                    "Der PNG-Writer kann mit der synchronisierten Bildrate nicht Schritt "
                    "halten (8 ausstehende Bilder)."
                )
                return
            capture_index = self._index
            try:
                self.writer.submit(capture_index, point, frame, metadata)
            except Exception as exc:
                self._fail(f"Speichern konnte nicht gestartet werden: {exc}")
                return
            self._ramp_pending_writes.add(capture_index)
            self._index += 1
            next_same_pass = (
                self._index < len(self._points)
                and self._sweep_key(self._points[self._index]) == self._sweep_pass_key
            )
            if next_same_pass:
                self._wait_for_sweep_frame()
            else:
                self._sweep_samples_finished = True
                if self._sweep_return_completed:
                    self._finish_sweep_pass()
                else:
                    remaining = self._sweep_duration_s / 2.0 + 7.0
                    self._set_phase("sweep_wait_return", remaining)
                    self.status_changed.emit(
                        "Alle Sweep-Bilder aufgenommen; warte auf Bandrückkehr zu 0 mm …"
                    )
            return
        if self._running and self._phase == "ramp_frame" and isinstance(frame, CameraFrame):
            point = self._points[self._index]
            planned = float(point.planned_offset_s or 0.0)
            actual = frame.timestamp - self._ramp_origin_wall
            if actual < planned:
                return
            timing_error = actual - planned
            if timing_error > 0.5:
                self._fail(
                    f"Kamerabild für Rampen-Sample {point.ramp_sample_id} kam "
                    f"{timing_error * 1000:.0f} ms zu spät (maximal 500 ms)."
                )
                return
            self._ramp_actual_offset_s = actual
            self._ramp_timing_error_s = timing_error
            metadata = self._metadata(point, frame)
            if not self.writer.can_accept():
                self._fail(
                    "Der PNG-Writer kann mit der Rampen-Bildrate nicht Schritt halten "
                    "(8 ausstehende Bilder)."
                )
                return
            capture_index = self._index
            try:
                self.writer.submit(capture_index, point, frame, metadata)
            except Exception as exc:
                self._fail(f"Speichern konnte nicht gestartet werden: {exc}")
                return
            self._ramp_pending_writes.add(capture_index)
            self._index += 1
            previous_key = (point.robot_key, point.conveyor_key, point.exposure_time_us)
            next_key = (
                (
                    self._points[self._index].robot_key,
                    self._points[self._index].conveyor_key,
                    self._points[self._index].exposure_time_us,
                )
                if self._index < len(self._points)
                else None
            )
            if next_key == previous_key:
                self._wait_for_ramp_frame()
            else:
                self._finish_ramp_pass()
            return
        if (
            not self._running
            or self._phase != "frame"
            or not isinstance(frame, CameraFrame)
            or frame.timestamp < self._frame_after
        ):
            return
        point = self._points[self._index]
        metadata = self._metadata(point, frame)
        self._set_phase("writing", 20.0)
        try:
            self.writer.submit(self._index, point, frame, metadata)
        except Exception as exc:
            self._fail(f"Speichern konnte nicht gestartet werden: {exc}")

    def _metadata(self, point: CapturePoint, frame: CameraFrame) -> dict[str, Any]:
        camera = self._camera_status
        robot = self._robot_status
        lights = self._light_statuses
        metadata = {
            "schema_version": 2,
            "captured_at": datetime.fromtimestamp(frame.timestamp).astimezone().isoformat(),
            "image": {
                "pixel_format": frame.pixel_format,
                "width": int(frame.image.shape[1]),
                "height": int(frame.image.shape[0]),
            },
            "camera": {
                "model": camera.model,
                "serial_number": camera.serial_number,
                "ip_address": camera.ip_address,
                "exposure_time_us": camera.exposure_time_us,
                "gain": camera.gain,
                "configured_frame_rate_fps": camera.camera_fps,
                "measured_preview_fps": camera.preview_fps,
            },
            "robot": {
                "control_mode": point.robot_control_mode,
                "requested_raw_value": point.robot_raw_value,
                "requested_pose_id": (
                    point.pose
                    if point.robot_control_mode == RobotCommandMode.POSE_ID.value
                    else None
                ),
                "requested_angle_deg": (
                    point.robot_raw_value / 10.0
                    if point.robot_control_mode == RobotCommandMode.ANGLE.value
                    else None
                ),
                "acknowledged_pose_id": robot.acknowledged_pose,
                "acknowledged_angle_deg": robot.acknowledged_angle_deg,
                "command_sequence": robot.acknowledged_sequence,
                "robot_mode": robot.robot_mode,
                "safety_mode": robot.safety_mode,
                "speed_scaling": robot.speed_scaling,
                "joint_positions_rad": list(robot.joint_positions),
                "tcp_pose": list(robot.tcp_pose),
            },
            "lights": {
                "panel_1": self._light_metadata(
                    lights[0], point.light_1_brightness, frame.timestamp
                ),
                "panel_2": self._light_metadata(
                    lights[1], point.light_2_brightness, frame.timestamp
                ),
            },
            "sequence": {
                "index": self._index + 1,
                "total": len(self._points),
            },
        }
        if self._settings is not None and self._settings.conveyor_enabled:
            conveyor = self.conveyor.status if self.conveyor is not None else ConveyorStatus()
            move = conveyor.last_move
            metadata["conveyor"] = {
                "plc_ip": None if self.conveyor is None else self.conveyor.config.plc_ip,
                "ams_net_id": (
                    None if self.conveyor is None else self.conveyor.config.plc_ams_net_id
                ),
                "station_id": point.conveyor_station_id,
                "direction": point.conveyor_direction,
                "motion_mode": point.conveyor_motion_mode,
                "origin_internal_position": self._saved_conveyor_origin,
                "internal_position": conveyor.internal_position,
                "internal_start_position": (
                    None if move is None else move.start_internal_position
                ),
                "internal_end_position": (
                    conveyor.internal_position
                    if move is not None and move.sequence == 0
                    else conveyor.completed_internal_position
                ),
                "nominal_offset_mm": point.conveyor_offset_mm,
                "quantized_target_offset_mm": (
                    point.conveyor_actual_offset_mm
                    if point.conveyor_actual_offset_mm is not None
                    else conveyor.actual_target_offset_mm
                ),
                "measured_logical_offset_mm": conveyor.logical_offset_mm,
                "calibration_mm_per_full_step": conveyor.mm_per_full_step,
                "target_full_steps": None if move is None else move.target_full_steps,
                "delta_full_steps": None if move is None else move.delta_full_steps,
                "speed_mm_per_s": self._settings.conveyor_speed_mm_per_s,
                "command_sequence": None if move is None else move.sequence,
                "commanded_at": None if move is None else move.commanded_at,
                "acknowledged_at": (
                    move.commanded_at
                    if move is not None and move.sequence == 0
                    else conveyor.completed_at
                ),
                "movement_started_at": (
                    None
                    if conveyor.movement_started_at is None
                    else datetime.fromtimestamp(conveyor.movement_started_at)
                    .astimezone()
                    .isoformat()
                ),
                "movement_started_sequence": conveyor.movement_started_sequence,
                "position_sampled_at": (
                    None
                    if conveyor.sampled_at is None
                    else datetime.fromtimestamp(conveyor.sampled_at).astimezone().isoformat()
                ),
                "movement_acknowledged": (
                    move is not None
                    and (move.sequence == 0 or conveyor.completed_sequence == move.sequence)
                ),
                "status_code": conveyor.status_code,
                "busy": conveyor.busy,
                "error": conveyor.error,
            }
            if point.conveyor_motion_mode == "synchronized":
                metadata["conveyor"]["synchronized_capture"] = {
                    "sample_id": point.conveyor_station_id,
                    "planned_active_motion_offset_s": point.planned_offset_s,
                    "actual_active_motion_offset_s": self._ramp_actual_offset_s,
                    "timing_error_ms": (
                        None
                        if self._ramp_timing_error_s is None
                        else self._ramp_timing_error_s * 1000.0
                    ),
                    "planned_position_mm": point.conveyor_offset_mm,
                    "measured_position_mm": conveyor.logical_offset_mm,
                    "effective_speed_mm_per_s": self._sweep_effective_speed_mm_s,
                    "active_round_trip_duration_s": self._sweep_duration_s,
                }
        if point.ramp_sample_id is not None:
            metadata["ramp"] = {
                "sample_id": point.ramp_sample_id,
                "planned_offset_s": point.planned_offset_s,
                "actual_capture_offset_s": self._ramp_actual_offset_s,
                "timing_error_ms": None
                if self._ramp_timing_error_s is None
                else self._ramp_timing_error_s * 1000.0,
                "panel_1_nominal_target_percent": point.light_1_brightness,
                "panel_2_nominal_target_percent": point.light_2_brightness,
            }
        return metadata

    @staticmethod
    def _light_metadata(
        status: LightStatus, requested_brightness: int, captured_at: float
    ) -> dict[str, Any]:
        confirmed_at = status.last_command_confirmed_at
        return {
            "name": status.name,
            "address": status.address,
            "requested_brightness_percent": requested_brightness,
            "confirmed_brightness_percent": status.brightness,
            "last_confirmed_command_percent": status.brightness,
            "last_command_duration_ms": status.last_command_duration_ms,
            "last_command_confirmed_at": None
            if confirmed_at is None
            else datetime.fromtimestamp(confirmed_at).astimezone().isoformat(),
            "confirmed_command_age_ms": None
            if confirmed_at is None
            else max(0.0, (captured_at - confirmed_at) * 1000.0),
            "mode": status.mode,
            "cct_kelvin": status.cct_kelvin if status.mode == "CCT" else None,
            "hue_degrees": status.hue if status.mode == "HSI" else None,
            "saturation_percent": status.saturation if status.mode == "HSI" else None,
            "power": status.power,
            "values_are_confirmed_commands": status.values_are_confirmed_commands,
        }

    def _on_saved(self, token: int, index: int, image_path: str) -> None:
        if token != self._writer_token:
            return
        if index in self._ramp_pending_writes:
            self._ramp_pending_writes.remove(index)
            self._ramp_completed_writes.add(index)
            while self._checkpoint_index in self._ramp_completed_writes:
                self._ramp_completed_writes.remove(self._checkpoint_index)
                self._checkpoint_index += 1
            self._write_manifest(
                "running" if self._running else "interrupted",
                f"Bild {index + 1} gespeichert." if self._running else self._last_finish_message,
            )
            self.progress_changed.emit(self._checkpoint_index, len(self._points))
            if self._running and self._phase == "ramp_draining" and not self._ramp_pending_writes:
                self._advance()
            return
        if (
            not self._running
            or self._phase not in {"writing", "ramp_writing"}
            or index != self._index
        ):
            return
        self._index += 1
        self._checkpoint_index = self._index
        self._write_manifest("running", f"Bild {self._index} gespeichert.")
        self.progress_changed.emit(self._index, len(self._points))
        self.status_changed.emit(f"Gespeichert: {image_path}")
        self._advance()

    def _on_write_failed(self, token: int, index: int, message: str) -> None:
        if (
            token == self._writer_token
            and self._running
            and (index == self._index or index in self._ramp_pending_writes)
        ):
            self._ramp_pending_writes.discard(index)
            self._fail(f"Speichern fehlgeschlagen: {message}")

    def _check_timeout(self) -> None:
        if self._running and time.monotonic() > self._deadline:
            if self._phase in {"ramp_sync", "ramp_finalize", "sweep_sync"}:
                self._fail(
                    f"Ein Panel hat den Lichtbefehl länger als "
                    f"{RAMP_BLE_COMMAND_TIMEOUT_SECONDS:g} Sekunden nicht bestätigt."
                )
            elif self._phase == "ramp_writing":
                self._fail(
                    "Das Speichern des PNG/YAML-Paars ist zu langsam für die Rampen-Bildrate."
                )
            elif self._phase == "sweep_frame":
                self._fail(
                    "Kein frisches Kamerabild zum geplanten Band-Sample empfangen."
                )
            elif self._phase == "ramp_frame":
                self._fail("Kein frisches Kamerabild zum geplanten Rampenzeitpunkt empfangen.")
            elif self._phase == "ramp_draining":
                self._fail(
                    "Ausstehende PNG/YAML-Dateien konnten nicht rechtzeitig gespeichert werden."
                )
            elif self._phase in {
                "conveyor",
                "sweep_out_start",
                "sweep_back_start",
                "sweep_wait_out",
                "sweep_wait_return",
            }:
                if self.conveyor is not None:
                    self.conveyor.stop_motion()
                self._fail("Zeitüberschreitung während der Förderbandfahrt; Stop wurde gesendet.")
            else:
                self._fail(f"Zeitüberschreitung in Phase '{self._phase}'.")

    def _fail(self, message: str) -> None:
        self.error.emit(message)
        self._finish(f"Aufnahme wegen Fehler beendet: {message}", resumable=True)

    def _finish(
        self,
        message: str,
        *,
        completed: bool = False,
        resumable: bool = False,
    ) -> None:
        restore_exposure = self._settings is not None and self._settings.exposure_enabled
        self._last_finish_message = message
        self._running = False
        self._phase = "idle"
        self._phase_timer.stop()
        self._ramp_timer.stop()
        self._watchdog.stop()
        if self.conveyor is not None and self._settings is not None:
            if self._settings.conveyor_enabled:
                self.conveyor.release_control()
        if restore_exposure:
            self.camera.restore_exposure()
        can_resume = resumable and self._checkpoint_index < len(self._points)
        self._write_manifest("interrupted" if can_resume else "completed", message)
        self._set_resume_available(can_resume)
        self.running_changed.emit(False)
        self.status_changed.emit(message)
        if completed:
            self.completed.emit(message)
