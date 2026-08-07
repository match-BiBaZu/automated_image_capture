from __future__ import annotations

import ipaddress
import queue
import re
import threading
import time
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from automated_image_capture.hardware.base import DeviceAdapter
from automated_image_capture.models import CameraFrame, CameraStatus, ConnectionState
from automated_image_capture.settings import AppSettings

FETCH_TIMEOUT_SECONDS = 0.5
FETCH_TIMEOUT_MARGIN_SECONDS = 0.5
STREAM_DEGRADED_AFTER_SECONDS = 3.0
STREAM_RESTART_AFTER_SECONDS = 5.0
MAX_STREAM_RESTART_ATTEMPTS = 2
# Kept for callers of the small retry-policy helper. The worker itself uses
# elapsed time, because a long exposure also increases the individual fetch timeout.
MAX_CONSECUTIVE_FETCH_TIMEOUTS = 10


def camera_fetch_timeout_seconds(exposure_time_us: float | None) -> float:
    """Allow one exposure plus transfer margin before treating a fetch as late."""
    if exposure_time_us is None:
        return FETCH_TIMEOUT_SECONDS
    return max(
        FETCH_TIMEOUT_SECONDS,
        max(0.0, float(exposure_time_us)) / 1_000_000.0
        + FETCH_TIMEOUT_MARGIN_SECONDS,
    )


def _normalize_to_uint8(image: np.ndarray, pixel_format: str) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    match = re.search(r"(10|12|14|16)", pixel_format)
    source_bits = int(match.group(1)) if match else image.dtype.itemsize * 8
    maximum = float((1 << source_bits) - 1)
    return np.clip(image.astype(np.float32) * (255.0 / maximum), 0, 255).astype(np.uint8)


def convert_to_rgb(
    data: np.ndarray | Any,
    width: int,
    height: int,
    pixel_format: str,
) -> np.ndarray:
    """Convert common PFNC camera formats into a contiguous RGB8 preview image."""
    array = np.asarray(data)
    fmt = str(pixel_format)
    lower = fmt.lower()

    if "packed" in lower or lower.endswith("p"):
        raise ValueError(f"Gepacktes Pixelformat wird noch nicht unterstützt: {fmt}")

    if lower.startswith("mono"):
        if array.size != width * height:
            raise ValueError(f"Unerwartete Datenmenge für {fmt}: {array.size}")
        mono = _normalize_to_uint8(array.reshape(height, width), fmt)
        return np.ascontiguousarray(cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB))

    bayer_codes = {
        "bayerrg": cv2.COLOR_BayerRG2RGB,
        "bayerbg": cv2.COLOR_BayerBG2RGB,
        "bayergr": cv2.COLOR_BayerGR2RGB,
        "bayergb": cv2.COLOR_BayerGB2RGB,
    }
    for prefix, code in bayer_codes.items():
        if lower.startswith(prefix):
            if array.size != width * height:
                raise ValueError(f"Unerwartete Datenmenge für {fmt}: {array.size}")
            bayer = _normalize_to_uint8(array.reshape(height, width), fmt)
            return np.ascontiguousarray(cv2.cvtColor(bayer, code))

    if lower.startswith("rgb8"):
        return np.ascontiguousarray(array.reshape(height, width, 3).astype(np.uint8))
    if lower.startswith("bgr8"):
        bgr = array.reshape(height, width, 3).astype(np.uint8)
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    raise ValueError(f"Nicht unterstütztes Pixelformat: {fmt}")


def _device_field(device: Any, name: str, default: str = "") -> str:
    value = getattr(device, name, default)
    if callable(value):
        value = value()
    return str(value) if value is not None else default


def _node_value(node_map: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(node_map, name).value
    except Exception:
        return default


def _find_node(node_map: Any, *names: str) -> Any:
    for name in names:
        try:
            return getattr(node_map, name)
        except Exception:
            continue
    return None


def _node_number(node: Any, attribute: str) -> float | None:
    try:
        return float(getattr(node, attribute))
    except Exception:
        return None


def _node_writable(node: Any) -> bool:
    if node is None:
        return False
    try:
        from genicam.genapi import is_writable

        return bool(is_writable(node))
    except Exception:
        access_mode = str(getattr(node, "access_mode", "")).upper()
        return access_mode in {"RW", "WO", "4"}


def _format_camera_ip(value: Any, fallback: str) -> str:
    if isinstance(value, int):
        try:
            return str(ipaddress.IPv4Address(value))
        except ipaddress.AddressValueError:
            return fallback
    return str(value) if value else fallback


def camera_error_message(error: BaseException) -> str:
    message = str(error) or type(error).__name__
    busy_tokens = (
        "resource",
        "in use",
        "access denied",
        "exclusive access",
        "operation is not allowed",
        "busy",
        "control channel is locked",
    )
    if any(token in message.lower() for token in busy_tokens):
        message += (
            " Schließen Sie den Baumer Camera Explorer. Falls er bereits geschlossen ist, "
            "Kamera in der GUI trennen und erneut verbinden."
        )
    return message


def is_camera_fetch_timeout(error: BaseException) -> bool:
    """Recognize GenTL timeouts even when the producer returns an empty message."""
    description = f"{type(error).__name__} {error}".lower()
    return "timeout" in description or "timed out" in description


def should_retry_camera_fetch(error: BaseException, consecutive_timeouts: int) -> bool:
    return (
        is_camera_fetch_timeout(error)
        and consecutive_timeouts < MAX_CONSECUTIVE_FETCH_TIMEOUTS
    )


class CameraWorker(QObject):
    state_changed = pyqtSignal(object)
    status_changed = pyqtSignal(object)
    frame_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    event_message = pyqtSignal(str)
    finished = pyqtSignal()
    exposure_applied = pyqtSignal(float)
    exposure_failed = pyqtSignal(str)

    def __init__(self, config: AppSettings) -> None:
        super().__init__()
        self._config = config
        self._stop = threading.Event()
        self._exposure_commands: queue.Queue[float | None] = queue.Queue()

    def enqueue_exposure(self, exposure_time_us: float) -> None:
        self._exposure_commands.put(float(exposure_time_us))

    def enqueue_exposure_restore(self) -> None:
        self._exposure_commands.put(None)

    def stop(self) -> None:
        self._stop.set()

    def _select_device(self, devices: list[Any]) -> Any:
        if self._config.camera_serial:
            for device in devices:
                if _device_field(device, "serial_number") == self._config.camera_serial:
                    return device
        for device in devices:
            if self._config.camera_ip in repr(device):
                return device
        if len(devices) == 1:
            return devices[0]
        descriptions = ", ".join(
            f"{_device_field(d, 'model', '?')} ({_device_field(d, 'serial_number', '?')})"
            for d in devices
        )
        raise RuntimeError(
            "Mehrere Kameras gefunden und keine eindeutige Auswahl möglich: " + descriptions
        )

    @pyqtSlot()
    def run(self) -> None:
        harvester = None
        acquirer = None
        exposure_node: Any = None
        exposure_auto_node: Any = None
        original_exposure: float | None = None
        original_exposure_auto: str | None = None
        try:
            self.state_changed.emit(ConnectionState.DISCOVERING)
            self.event_message.emit("Suche über den Baumer GenTL-Producer …")
            from harvesters.core import Harvester

            harvester = Harvester()
            harvester.add_file(
                self._config.camera_cti_path,
                check_existence=True,
                check_validity=True,
            )
            harvester.update()
            devices = list(harvester.device_info_list)
            if not devices:
                raise RuntimeError(
                    "Keine GenTL-Kamera gefunden. Netzwerk, Stromversorgung und CTI-Pfad prüfen."
                )

            selected = self._select_device(devices)
            self.state_changed.emit(ConnectionState.CONNECTING)
            self.event_message.emit(
                f"Öffne {_device_field(selected, 'model', 'Baumer-Kamera')} "
                f"({_device_field(selected, 'serial_number', 'ohne Seriennummer')}) …"
            )
            acquirer = harvester.create(selected)
            node_map = acquirer.remote_device.node_map
            exposure_node = _find_node(node_map, "ExposureTime", "ExposureTimeAbs")
            exposure_auto_node = _find_node(node_map, "ExposureAuto")
            original_exposure = _node_number(exposure_node, "value")
            exposure_auto = str(_node_value(node_map, "ExposureAuto", "–"))
            original_exposure_auto = exposure_auto
            width = int(_node_value(node_map, "Width", 0))
            height = int(_node_value(node_map, "Height", 0))
            pixel_format = str(_node_value(node_map, "PixelFormat", "Unbekannt"))
            camera_fps_raw = _node_value(
                node_map,
                "AcquisitionFrameRate",
                _node_value(node_map, "ResultingFrameRate", None),
            )
            camera_fps = float(camera_fps_raw) if camera_fps_raw is not None else None
            camera_ip = _format_camera_ip(
                _node_value(node_map, "GevCurrentIPAddress", None), self._config.camera_ip
            )
            status = CameraStatus(
                model=_device_field(selected, "model", "Baumer"),
                serial_number=_device_field(selected, "serial_number", "–"),
                ip_address=camera_ip,
                width=width,
                height=height,
                pixel_format=pixel_format,
                camera_fps=camera_fps,
                exposure_time_us=original_exposure,
                exposure_min_us=_node_number(exposure_node, "min"),
                exposure_max_us=_node_number(exposure_node, "max"),
                exposure_writable=exposure_node is not None
                and (
                    _node_writable(exposure_node)
                    or _node_writable(exposure_auto_node)
                ),
                exposure_auto=exposure_auto,
                gain=(
                    float(_node_value(node_map, "Gain", 0.0))
                    if _node_value(node_map, "Gain", None) is not None
                    else None
                ),
            )
            self.status_changed.emit(status)

            acquirer.start()
            self.state_changed.emit(ConnectionState.CONNECTED)
            self.event_message.emit("Liveaufnahme gestartet.")
            preview_interval = 1.0 / max(1, self._config.preview_max_fps)
            last_preview = 0.0
            fps_window_start = time.monotonic()
            preview_frames = 0
            consecutive_fetch_timeouts = 0
            stream_restart_attempts = 0
            outage_started: float | None = None
            restart_window_started: float | None = None
            stream_degraded = False

            while not self._stop.is_set():
                try:
                    while True:
                        requested_exposure = self._exposure_commands.get_nowait()
                        if exposure_node is None or not status.exposure_writable:
                            raise RuntimeError(
                                "Die Belichtungszeit ist an dieser Kamera nicht manuell "
                                "verstellbar."
                            )
                        restoring = requested_exposure is None
                        if restoring:
                            if original_exposure is None:
                                raise RuntimeError("Ursprüngliche Belichtungszeit ist unbekannt.")
                            requested_exposure = original_exposure
                        current_auto = str(
                            getattr(exposure_auto_node, "value", status.exposure_auto)
                        )
                        if current_auto.lower() not in {"off", "–", "none"}:
                            if exposure_auto_node is None or not _node_writable(
                                exposure_auto_node
                            ):
                                raise RuntimeError(
                                    "ExposureAuto ist aktiv und kann nicht deaktiviert werden."
                                )
                            exposure_auto_node.value = "Off"
                        minimum = status.exposure_min_us or requested_exposure
                        maximum = status.exposure_max_us or requested_exposure
                        requested_exposure = max(minimum, min(maximum, requested_exposure))
                        if not _node_writable(exposure_node):
                            raise RuntimeError(
                                "ExposureTime ist nach dem Abschalten von ExposureAuto "
                                "nicht beschreibbar."
                            )
                        exposure_node.value = requested_exposure
                        applied = float(exposure_node.value)
                        if (
                            restoring
                            and exposure_auto_node is not None
                            and original_exposure_auto is not None
                            and original_exposure_auto.lower() not in {"off", "–", "none"}
                        ):
                            exposure_auto_node.value = original_exposure_auto
                        status.exposure_time_us = applied
                        status.exposure_auto = str(
                            getattr(exposure_auto_node, "value", status.exposure_auto)
                        )
                        self.status_changed.emit(status)
                        self.exposure_applied.emit(applied)
                except queue.Empty:
                    pass
                except Exception as exc:
                    self.exposure_failed.emit(str(exc) or type(exc).__name__)

                try:
                    fetch_timeout = camera_fetch_timeout_seconds(status.exposure_time_us)
                    with acquirer.fetch(timeout=fetch_timeout) as buffer:
                        if stream_degraded:
                            self.state_changed.emit(ConnectionState.CONNECTED)
                            self.event_message.emit(
                                "Kamerastream automatisch wiederhergestellt."
                            )
                        consecutive_fetch_timeouts = 0
                        stream_restart_attempts = 0
                        outage_started = None
                        restart_window_started = None
                        stream_degraded = False
                        component = buffer.payload.components[0]
                        now = time.monotonic()
                        if now - last_preview < preview_interval:
                            continue
                        fmt = str(getattr(component, "data_format", pixel_format))
                        rgb = convert_to_rgb(
                            component.data,
                            int(component.width),
                            int(component.height),
                            fmt,
                        ).copy()
                    last_preview = now
                    preview_frames += 1
                    self.frame_ready.emit(CameraFrame(rgb, fmt, time.time()))
                    elapsed = now - fps_window_start
                    if elapsed >= 1.0:
                        status.preview_fps = preview_frames / elapsed
                        status.exposure_time_us = _node_number(exposure_node, "value")
                        status.exposure_auto = str(
                            getattr(exposure_auto_node, "value", status.exposure_auto)
                        )
                        self.status_changed.emit(status)
                        fps_window_start = now
                        preview_frames = 0
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    if is_camera_fetch_timeout(exc):
                        consecutive_fetch_timeouts += 1
                        now = time.monotonic()
                        if outage_started is None:
                            outage_started = now
                            restart_window_started = now
                        if consecutive_fetch_timeouts == 1:
                            self.event_message.emit(
                                "Ein Kamerabuffer blieb aus; der Stream wird weiter abgefragt."
                            )
                        outage_seconds = now - outage_started
                        if (
                            not stream_degraded
                            and outage_seconds >= STREAM_DEGRADED_AFTER_SECONDS
                        ):
                            stream_degraded = True
                            status.preview_fps = 0.0
                            self.status_changed.emit(status)
                            self.state_changed.emit(ConnectionState.DEGRADED)
                            self.event_message.emit(
                                "Kamerastream vorübergehend unterbrochen; "
                                "Wiederherstellung läuft."
                            )
                        assert restart_window_started is not None
                        if now - restart_window_started < STREAM_RESTART_AFTER_SECONDS:
                            continue
                        if stream_restart_attempts < MAX_STREAM_RESTART_ATTEMPTS:
                            stream_restart_attempts += 1
                            self.event_message.emit(
                                "Starte den Kamera-Datenstrom neu "
                                f"(Versuch {stream_restart_attempts}/"
                                f"{MAX_STREAM_RESTART_ATTEMPTS}) …"
                            )
                            try:
                                acquirer.stop()
                            except Exception as stop_error:
                                raise RuntimeError(
                                    "Kamerastream konnte vor dem Neustart nicht sauber "
                                    "angehalten werden. Kamera kurz trennen und erneut verbinden."
                                ) from stop_error
                            # Some GenTL producers release announced buffers just after
                            # stop() returns. Starting in the same tick can otherwise race
                            # that release and produce a misleading BusyException.
                            time.sleep(0.1)
                            acquirer.start()
                            consecutive_fetch_timeouts = 0
                            restart_window_started = time.monotonic()
                            continue
                        raise RuntimeError(
                            "Kamerastream liefert seit "
                            f"{outage_seconds:.1f} Sekunden kein Bild; "
                            "die automatische Wiederherstellung ist fehlgeschlagen."
                        ) from exc
                    raise
        except Exception as exc:
            self.state_changed.emit(ConnectionState.ERROR)
            self.error.emit(camera_error_message(exc))
        finally:
            if acquirer is not None:
                if exposure_node is not None and original_exposure is not None:
                    try:
                        current_auto = str(
                            getattr(exposure_auto_node, "value", original_exposure_auto)
                        )
                        if (
                            current_auto.lower() not in {"off", "–", "none"}
                            and exposure_auto_node is not None
                            and _node_writable(exposure_auto_node)
                        ):
                            exposure_auto_node.value = "Off"
                        exposure_node.value = original_exposure
                        if exposure_auto_node is not None and original_exposure_auto is not None:
                            exposure_auto_node.value = original_exposure_auto
                    except Exception:
                        pass
                try:
                    acquirer.stop()
                except Exception:
                    pass
                try:
                    acquirer.destroy()
                except Exception:
                    pass
            if harvester is not None:
                try:
                    harvester.reset()
                except Exception:
                    pass
            if self._stop.is_set():
                self.state_changed.emit(ConnectionState.DISCONNECTED)
                self.event_message.emit("Verbindung getrennt.")
            self.finished.emit()


class CameraAdapter(DeviceAdapter):
    frame_ready = pyqtSignal(object)
    exposure_applied = pyqtSignal(float)

    def __init__(self, config: AppSettings, parent: QObject | None = None) -> None:
        super().__init__("Kamera", parent)
        self.config = config
        self._thread: QThread | None = None
        self._worker: CameraWorker | None = None
        self._status = CameraStatus()

    def connect(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread(self)
        self._worker = CameraWorker(self.config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.state_changed.connect(self._set_state)
        self._worker.status_changed.connect(self._camera_status)
        self._worker.frame_ready.connect(self._forward_frame)
        self._worker.exposure_applied.connect(self.exposure_applied)
        self._worker.exposure_failed.connect(self._emit_error)
        self._worker.error.connect(self._emit_error)
        self._worker.event_message.connect(self._emit_event)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread)
        self._thread.start()

    @pyqtSlot()
    def _clear_thread(self) -> None:
        self._worker = None
        self._thread = None

    @pyqtSlot(object)
    def _forward_frame(self, frame: object) -> None:
        self.frame_ready.emit(frame)

    @pyqtSlot(object)
    def _camera_status(self, status: object) -> None:
        if isinstance(status, CameraStatus):
            self._status = status
        self._forward_status(status)

    def set_exposure_time(self, exposure_time_us: float) -> bool:
        if self._worker is None or self.state is not ConnectionState.CONNECTED:
            self._emit_error("Belichtungszeit kann nur bei verbundener Kamera geändert werden.")
            return False
        if not self._status.exposure_writable:
            self._emit_error("Die Kamera meldet keine manuell verstellbare Belichtungszeit.")
            return False
        self._worker.enqueue_exposure(exposure_time_us)
        return True

    def restore_exposure(self) -> bool:
        if self._worker is None or self.state is not ConnectionState.CONNECTED:
            return False
        self._worker.enqueue_exposure_restore()
        return True

    def disconnect(self) -> None:
        worker, thread = self._worker, self._thread
        if worker is not None:
            worker.stop()
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(3000):
                self._emit_error("Kamera-Worker wurde nicht innerhalb von 3 Sekunden beendet.")
        if self.state is not ConnectionState.ERROR:
            self._set_state(ConnectionState.DISCONNECTED)
