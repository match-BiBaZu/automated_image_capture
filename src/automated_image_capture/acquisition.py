from __future__ import annotations

import json
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
from automated_image_capture.hardware.light import LightAdapter
from automated_image_capture.hardware.robot import ALLOWED_POSES, RobotAdapter
from automated_image_capture.models import (
    CameraFrame,
    CameraStatus,
    ConnectionState,
    LightStatus,
    RobotStatus,
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


@dataclass(slots=True, frozen=True)
class AcquisitionSettings:
    output_directory: Path
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

    def validated(self) -> AcquisitionSettings:
        if not str(self.output_directory):
            raise ValueError("Bitte einen Speicherort auswählen.")
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
        return self


@dataclass(slots=True, frozen=True)
class CapturePoint:
    pose: int
    light_1_brightness: int
    light_2_brightness: int
    exposure_time_us: int | None


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
    return tuple(
        CapturePoint(pose, light_1, light_2, exposure)
        for pose in poses_between(settings.pose_start, settings.pose_end)
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
        for exposure in exposures
    )


class DatasetWriter(QObject):
    saved = pyqtSignal(int, int, str)
    failed = pyqtSignal(int, int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-writer")
        self._session_directory: Path | None = None
        self._session_token = 0

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
        stem = (
            f"img_{index + 1:06d}_ur{point.pose}_"
            f"p1-{point.light_1_brightness:03d}_p2-{point.light_2_brightness:03d}_{exposure}"
        )
        return (
            self._session_directory / f"{stem}.png",
            self._session_directory / f"{stem}.yaml",
        )

    def flush(self, timeout_seconds: float = 30.0) -> None:
        self._executor.submit(lambda: None).result(timeout=timeout_seconds)

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
        image = np.ascontiguousarray(frame.image.copy())
        future = self._executor.submit(
            self._write_pair,
            image_path,
            yaml_path,
            image,
            metadata,
        )

        def done(result: Any) -> None:
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
        rgb_image: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", bgr)
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

    def __init__(
        self,
        camera: CameraAdapter,
        robot: RobotAdapter,
        light_1: LightAdapter,
        light_2: LightAdapter,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera = camera
        self.robot = robot
        self.light_1 = light_1
        self.light_2 = light_2
        self.writer = DatasetWriter(self)
        self._settings: AcquisitionSettings | None = None
        self._points: tuple[CapturePoint, ...] = ()
        self._index = 0
        self._running = False
        self._phase = "idle"
        self._deadline = 0.0
        self._current_pose: int | None = None
        self._applied_lights: tuple[int, int] | None = None
        self._applied_exposure: int | None = None
        self._frame_after = 0.0
        self._session_directory: Path | None = None
        self._writer_token = 0
        self._resume_available = False
        self._camera_status = CameraStatus()
        self._robot_status = RobotStatus()
        self._light_statuses = [LightStatus(), LightStatus()]

        self._phase_timer = QTimer(self)
        self._phase_timer.setSingleShot(True)
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(100)
        self._watchdog.timeout.connect(self._check_timeout)

        camera.frame_ready.connect(self._on_frame)
        camera.status_changed.connect(self._on_camera_status)
        robot.status_changed.connect(self._on_robot_status)
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
        return max(0, len(self._points) - self._index)

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
            session = self.writer.start_session(settings.output_directory)
        except Exception as exc:
            self.error.emit(str(exc) or type(exc).__name__)
            return False

        if self._resume_available:
            self._write_manifest("abandoned", "Durch eine neue Sitzung ersetzt.")
        self._settings = settings
        self._points = points
        self._index = 0
        self._running = True
        self._current_pose = None
        self._applied_lights = None
        self._applied_exposure = None
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
        self._finish(
            "Aufnahme gestoppt. Eine bereits gestartete UR-Fahrt wird nicht abgebrochen.",
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
        except Exception as exc:
            self.error.emit(str(exc) or type(exc).__name__)
            return False

        self._running = True
        self._current_pose = None
        self._applied_lights = None
        self._applied_exposure = None
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
                self._session_directory = manifest.parent
                self._writer_token = self.writer.session_token
                self._reconcile_saved_points()
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
        names = set(AcquisitionSettings.__dataclass_fields__)
        values = {name: payload[name] for name in names}
        values["output_directory"] = Path(str(values["output_directory"]))
        return AcquisitionSettings(**values).validated()

    def _write_manifest(self, status: str, message: str) -> None:
        if self._settings is None or self._session_directory is None:
            return
        settings = asdict(self._settings)
        settings["output_directory"] = str(self._settings.output_directory)
        payload = {
            "schema_version": 1,
            "status": status,
            "next_index": self._index,
            "total": len(self._points),
            "settings": settings,
            "message": message,
            "updated_at": datetime.now().astimezone().isoformat(),
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
        if self.camera.state is not ConnectionState.CONNECTED:
            raise RuntimeError("Die Kamera ist nicht verbunden.")
        if self.light_1.state is not ConnectionState.CONNECTED:
            raise RuntimeError("Licht 1 ist nicht verbunden.")
        if self.light_2.state is not ConnectionState.CONNECTED:
            raise RuntimeError("Licht 2 ist nicht verbunden.")
        robot_ready = (
            self._robot_status.rtde_connected
            and self._robot_status.command_channel_connected
            and self._robot_status.robot_mode.upper() == "RUNNING"
            and self._robot_status.safety_mode.upper() in {"NORMAL", "REDUCED"}
            and "PLAYING" in self._robot_status.program_state.upper()
            and "BIBAZU" in self._robot_status.loaded_program.upper()
            and self._robot_status.command_state_code in {1, 3, -1}
        )
        if not robot_ready:
            raise RuntimeError("Der UR-Handshake ist nicht bereit oder BiBaZu_GUI läuft nicht.")
        if settings.exposure_enabled:
            if not self._camera_status.exposure_writable:
                raise RuntimeError("Die Kamera erlaubt aktuell keine manuelle Belichtungszeit.")
            minimum = self._camera_status.exposure_min_us
            maximum = self._camera_status.exposure_max_us
            low = min(settings.exposure_start_us, settings.exposure_end_us)
            high = max(settings.exposure_start_us, settings.exposure_end_us)
            if minimum is not None and low < minimum:
                raise RuntimeError(
                    f"Belichtungszeit liegt unter dem Kameraminimum {minimum:.0f} µs."
                )
            if maximum is not None and high > maximum:
                raise RuntimeError(
                    f"Belichtungszeit liegt über dem Kameramaximum {maximum:.0f} µs."
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
            directory = str(self._session_directory or "")
            self._finish(f"Aufnahme abgeschlossen: {directory}", completed=True)
            return
        point = self._points[self._index]
        if self._current_pose != point.pose:
            self.status_changed.emit(f"Fahre UR-Pose {point.pose} …")
            self._set_phase("robot", 45.0)
            if not self.robot.request_pose(point.pose):
                self._fail("UR-Pose konnte nicht angefordert werden.")
            return
        self._prepare_point()

    def _prepare_point(self) -> None:
        point = self._points[self._index]
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
            self._wait_for_frame()
            return
        self.status_changed.emit(f"Setze Belichtungszeit {point.exposure_time_us} µs …")
        self._set_phase("exposure", 5.0)
        if not self.camera.set_exposure_time(point.exposure_time_us):
            self._fail("Belichtungszeit konnte nicht gesetzt werden.")

    def _wait_for_frame(self) -> None:
        self._frame_after = time.time()
        self._set_phase("frame", 5.0)
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
                    self._wait_for_frame,
                    "exposure_settle",
                )

    def _on_robot_status(self, status: object) -> None:
        if not isinstance(status, RobotStatus):
            return
        self._robot_status = replace(status)
        if not self._running or self._phase != "robot":
            return
        point = self._points[self._index]
        if (
            status.command_state_code == 3
            and not status.command_pending
            and status.requested_sequence is not None
            and status.acknowledged_sequence == status.requested_sequence
            and status.acknowledged_pose in {None, point.pose}
        ):
            self._current_pose = point.pose
            assert self._settings is not None
            self._schedule(
                self._settings.robot_settle_ms,
                self._prepare_point,
                "robot_settle",
            )

    def _on_light_status(self, index: int, status: object) -> None:
        if not isinstance(status, LightStatus):
            return
        self._light_statuses[index] = replace(status)
        self._maybe_power_ready()
        self._maybe_lights_ready()

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

    def _on_frame(self, frame: object) -> None:
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
        return {
            "schema_version": 1,
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
            },
            "robot": {
                "requested_pose_id": point.pose,
                "acknowledged_pose_id": robot.acknowledged_pose,
                "command_sequence": robot.acknowledged_sequence,
                "robot_mode": robot.robot_mode,
                "safety_mode": robot.safety_mode,
                "speed_scaling": robot.speed_scaling,
                "joint_positions_rad": list(robot.joint_positions),
                "tcp_pose": list(robot.tcp_pose),
            },
            "lights": {
                "panel_1": self._light_metadata(lights[0], point.light_1_brightness),
                "panel_2": self._light_metadata(lights[1], point.light_2_brightness),
            },
            "sequence": {
                "index": self._index + 1,
                "total": len(self._points),
            },
        }

    @staticmethod
    def _light_metadata(status: LightStatus, requested_brightness: int) -> dict[str, Any]:
        return {
            "name": status.name,
            "address": status.address,
            "requested_brightness_percent": requested_brightness,
            "confirmed_brightness_percent": status.brightness,
            "mode": status.mode,
            "cct_kelvin": status.cct_kelvin if status.mode == "CCT" else None,
            "hue_degrees": status.hue if status.mode == "HSI" else None,
            "saturation_percent": status.saturation if status.mode == "HSI" else None,
            "power": status.power,
            "values_are_confirmed_commands": status.values_are_confirmed_commands,
        }

    def _on_saved(self, token: int, index: int, image_path: str) -> None:
        if (
            token != self._writer_token
            or not self._running
            or self._phase != "writing"
            or index != self._index
        ):
            return
        self._index += 1
        self._write_manifest("running", f"Bild {self._index} gespeichert.")
        self.progress_changed.emit(self._index, len(self._points))
        self.status_changed.emit(f"Gespeichert: {image_path}")
        self._advance()

    def _on_write_failed(self, token: int, index: int, message: str) -> None:
        if token == self._writer_token and self._running and index == self._index:
            self._fail(f"Speichern fehlgeschlagen: {message}")

    def _check_timeout(self) -> None:
        if self._running and time.monotonic() > self._deadline:
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
        restore_exposure = (
            self._settings is not None and self._settings.exposure_enabled
        )
        self._running = False
        self._phase = "idle"
        self._phase_timer.stop()
        self._watchdog.stop()
        if restore_exposure:
            self.camera.restore_exposure()
        can_resume = resumable and self._index < len(self._points)
        self._write_manifest("interrupted" if can_resume else "completed", message)
        self._set_resume_available(can_resume)
        self.running_changed.emit(False)
        self.status_changed.emit(message)
        if completed:
            self.completed.emit(message)
