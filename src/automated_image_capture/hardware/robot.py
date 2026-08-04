from __future__ import annotations

import socket
import threading
import time
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from automated_image_capture.hardware.base import DeviceAdapter
from automated_image_capture.models import ConnectionState, RobotStatus
from automated_image_capture.settings import AppSettings

ROBOT_MODES = {
    -1: "NO_CONTROLLER",
    0: "DISCONNECTED",
    1: "CONFIRM_SAFETY",
    2: "BOOTING",
    3: "POWER_OFF",
    4: "POWER_ON",
    5: "IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "UPDATING_FIRMWARE",
}

SAFETY_MODES = {
    1: "NORMAL",
    2: "REDUCED",
    3: "PROTECTIVE_STOP",
    4: "RECOVERY",
    5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION",
    9: "FAULT",
    10: "VALIDATE_JOINT_ID",
    11: "UNDEFINED",
    12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
    13: "THREE_POSITION_ENABLING_STOP",
}


class DashboardReadClient:
    """Minimal line-based client restricted to read-only dashboard commands."""

    ALLOWED_COMMANDS = frozenset(
        {"robotmode", "safetymode", "programState", "is in remote control", "PolyscopeVersion"}
    )

    def __init__(self, host: str, timeout: float = 2.0) -> None:
        self._host = host
        self._timeout = timeout
        self._socket: socket.socket | None = None
        self._reader: Any = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        self.close()
        self._socket = socket.create_connection((self._host, 29999), timeout=self._timeout)
        self._socket.settimeout(self._timeout)
        self._reader = self._socket.makefile("r", encoding="utf-8", newline="\n")
        greeting = self._reader.readline()
        if not greeting:
            raise ConnectionError("Dashboard-Server hat keine Begrüßung gesendet.")

    def query(self, command: str) -> str:
        if command not in self.ALLOWED_COMMANDS:
            raise ValueError(f"Nicht freigegebener Dashboard-Befehl: {command}")
        if self._socket is None or self._reader is None:
            raise ConnectionError("Dashboard-Server ist nicht verbunden.")
        self._socket.sendall((command + "\n").encode("ascii"))
        response = self._reader.readline()
        if not response:
            raise ConnectionError("Dashboard-Verbindung wurde geschlossen.")
        return response.strip()

    def close(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except OSError:
                pass
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._reader = None
        self._socket = None


def _dashboard_value(response: str) -> str:
    if ":" in response:
        return response.split(":", 1)[1].strip()
    return response.strip()


def _safe_call(target: Any, method: str, default: Any = None) -> Any:
    try:
        return getattr(target, method)()
    except Exception:
        return default


class RobotWorker(QObject):
    state_changed = pyqtSignal(object)
    status_changed = pyqtSignal(object)
    error = pyqtSignal(str)
    event_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: AppSettings) -> None:
        super().__init__()
        self._config = config
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @pyqtSlot()
    def run(self) -> None:
        rtde: Any = None
        dashboard: DashboardReadClient | None = None
        last_rtde_attempt = 0.0
        last_dashboard_attempt = 0.0
        last_dashboard_poll = 0.0
        last_reported_error = ""
        dashboard_values: dict[str, str] = {}
        was_rtde_connected = False
        was_dashboard_connected = False
        self.state_changed.emit(ConnectionState.CONNECTING)
        self.event_message.emit("Starte read-only RTDE- und Dashboard-Verbindungen …")

        try:
            from rtde_receive import RTDEReceiveInterface

            while not self._stop.is_set():
                now = time.monotonic()
                rtde_error = ""
                dashboard_error = ""

                if rtde is None and now - last_rtde_attempt >= 5.0:
                    last_rtde_attempt = now
                    try:
                        rtde = RTDEReceiveInterface(self._config.robot_ip, 10.0)
                        if not rtde.isConnected():
                            raise ConnectionError("RTDE-Handshake nicht erfolgreich.")
                        if not was_rtde_connected:
                            self.event_message.emit("RTDE-Monitor verbunden (nur Lesen).")
                        was_rtde_connected = True
                    except Exception as exc:
                        rtde_error = f"RTDE: {exc}"
                        rtde = None

                if dashboard is None and now - last_dashboard_attempt >= 5.0:
                    last_dashboard_attempt = now
                    try:
                        dashboard = DashboardReadClient(self._config.robot_ip)
                        dashboard.connect()
                        if not was_dashboard_connected:
                            self.event_message.emit(
                                "Dashboard-Server verbunden (nur Statusbefehle)."
                            )
                        was_dashboard_connected = True
                    except Exception as exc:
                        dashboard_error = f"Dashboard: {exc}"
                        if dashboard is not None:
                            dashboard.close()
                        dashboard = None

                if dashboard is not None and now - last_dashboard_poll >= 1.0:
                    last_dashboard_poll = now
                    try:
                        for command, key in (
                            ("robotmode", "robot_mode"),
                            ("safetymode", "safety_mode"),
                            ("programState", "program_state"),
                            ("is in remote control", "remote_control"),
                            ("PolyscopeVersion", "polyscope_version"),
                        ):
                            dashboard_values[key] = _dashboard_value(dashboard.query(command))
                    except Exception as exc:
                        dashboard_error = f"Dashboard: {exc}"
                        dashboard.close()
                        dashboard = None

                status = RobotStatus(
                    rtde_connected=rtde is not None,
                    dashboard_connected=dashboard is not None,
                    robot_mode=dashboard_values.get("robot_mode", "–"),
                    safety_mode=dashboard_values.get("safety_mode", "–"),
                    remote_control=dashboard_values.get("remote_control", "–"),
                    program_state=dashboard_values.get("program_state", "–"),
                    polyscope_version=dashboard_values.get("polyscope_version", "–"),
                )

                if rtde is not None:
                    try:
                        robot_mode = int(rtde.getRobotMode())
                        safety_mode = int(rtde.getSafetyMode())
                        status.robot_mode = ROBOT_MODES.get(robot_mode, str(robot_mode))
                        status.safety_mode = SAFETY_MODES.get(safety_mode, str(safety_mode))
                        status.speed_scaling = float(rtde.getSpeedScaling())
                        status.joint_positions = tuple(float(v) for v in rtde.getActualQ())
                        status.tcp_pose = tuple(float(v) for v in rtde.getActualTCPPose())
                    except Exception as exc:
                        rtde_error = f"RTDE: {exc}"
                        try:
                            rtde.disconnect()
                        except Exception:
                            pass
                        rtde = None
                        status.rtde_connected = False

                if status.rtde_connected and status.dashboard_connected:
                    state = ConnectionState.CONNECTED
                elif status.rtde_connected or status.dashboard_connected:
                    state = ConnectionState.DEGRADED
                else:
                    state = ConnectionState.ERROR
                self.state_changed.emit(state)
                self.status_changed.emit(status)

                combined_error = "; ".join(filter(None, (rtde_error, dashboard_error)))
                if combined_error and combined_error != last_reported_error:
                    self.error.emit(combined_error)
                    last_reported_error = combined_error
                elif not combined_error:
                    last_reported_error = ""

                self._stop.wait(0.1)
        except Exception as exc:
            self.state_changed.emit(ConnectionState.ERROR)
            self.error.emit(f"ur_rtde konnte nicht geladen oder gestartet werden: {exc}")
        finally:
            if rtde is not None:
                try:
                    rtde.disconnect()
                except Exception:
                    pass
            if dashboard is not None:
                dashboard.close()
            self.state_changed.emit(ConnectionState.DISCONNECTED)
            self.event_message.emit(
                "Verbindungen getrennt; es wurden keine Roboterbefehle gesendet."
            )
            self.finished.emit()


class RobotAdapter(DeviceAdapter):
    def __init__(self, config: AppSettings, parent: QObject | None = None) -> None:
        super().__init__("UR16e", parent)
        self.config = config
        self._thread: QThread | None = None
        self._worker: RobotWorker | None = None

    def connect(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread(self)
        self._worker = RobotWorker(self.config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.state_changed.connect(self._set_state)
        self._worker.status_changed.connect(self._forward_status)
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

    def disconnect(self) -> None:
        worker, thread = self._worker, self._thread
        if worker is not None:
            worker.stop()
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(3000):
                self._emit_error("UR-Worker wurde nicht innerhalb von 3 Sekunden beendet.")
        self._set_state(ConnectionState.DISCONNECTED)
