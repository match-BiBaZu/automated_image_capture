from __future__ import annotations

import ipaddress
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
    )
    if any(token in message.lower() for token in busy_tokens):
        message += " Schließen Sie den Baumer Camera Explorer und versuchen Sie es erneut."
    return message


class CameraWorker(QObject):
    state_changed = pyqtSignal(object)
    status_changed = pyqtSignal(object)
    frame_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    event_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: AppSettings) -> None:
        super().__init__()
        self._config = config
        self._stop = threading.Event()

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
            )
            self.status_changed.emit(status)

            acquirer.start()
            self.state_changed.emit(ConnectionState.CONNECTED)
            self.event_message.emit("Liveaufnahme gestartet.")
            preview_interval = 1.0 / max(1, self._config.preview_max_fps)
            last_preview = 0.0
            fps_window_start = time.monotonic()
            preview_frames = 0

            while not self._stop.is_set():
                try:
                    with acquirer.fetch(timeout=0.5) as buffer:
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
                        self.status_changed.emit(status)
                        fps_window_start = now
                        preview_frames = 0
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    if "timeout" not in str(exc).lower():
                        raise
        except Exception as exc:
            self.state_changed.emit(ConnectionState.ERROR)
            self.error.emit(camera_error_message(exc))
        finally:
            if acquirer is not None:
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

    def __init__(self, config: AppSettings, parent: QObject | None = None) -> None:
        super().__init__("Kamera", parent)
        self.config = config
        self._thread: QThread | None = None
        self._worker: CameraWorker | None = None

    def connect(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread(self)
        self._worker = CameraWorker(self.config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.state_changed.connect(self._set_state)
        self._worker.status_changed.connect(self._forward_status)
        self._worker.frame_ready.connect(self._forward_frame)
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
