from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from automated_image_capture.hardware.base import DeviceAdapter
from automated_image_capture.models import ConnectionState, RobotCommandMode, RobotStatus
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

ALLOWED_POSES = (
    155,
    160,
    170,
    180,
    190,
    200,
    210,
    1155,
    1170,
    1185,
    1200,
    2155,
    2170,
    2185,
    2200,
)
POSE_INPUT_REGISTER = 42
SEQUENCE_INPUT_REGISTER = 43
ACK_SEQUENCE_OUTPUT_REGISTER = 42
COMMAND_STATE_OUTPUT_REGISTER = 43
CURRENT_POSE_OUTPUT_REGISTER = 41
ANGLE_MIN_TENTHS = 155
ANGLE_MAX_TENTHS = 210

COMMAND_STATES = {
    0: "UR-Programm nicht bereit",
    1: "Bereit",
    2: "Fährt",
    3: "Pose erreicht",
    -1: "Ungültige Pose abgelehnt",
}


class DashboardReadClient:
    """Minimal line-based client restricted to read-only dashboard commands."""

    ALLOWED_COMMANDS = frozenset(
        {
            "robotmode",
            "safetymode",
            "programState",
            "get loaded program",
            "is in remote control",
            "PolyscopeVersion",
        }
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


def _create_rtde_receive(interface: Any, host: str) -> Any:
    """Open the receive recipe using external-client registers 42-46."""
    return interface(host, 10.0, [], False, True)


class RobotWorker(QObject):
    state_changed = pyqtSignal(object)
    status_changed = pyqtSignal(object)
    error = pyqtSignal(str)
    event_message = pyqtSignal(str)
    finished = pyqtSignal()
    command_failed = pyqtSignal(int, str)

    def __init__(self, config: AppSettings) -> None:
        super().__init__()
        self._config = config
        self._stop = threading.Event()
        self._commands: queue.Queue[tuple[RobotCommandMode, int, int]] = queue.Queue()
        self._requested_mode = RobotCommandMode.POSE_ID
        self._requested_value: int | None = None
        self._requested_sequence: int | None = None

    def enqueue_pose_command(self, pose: int, sequence: int) -> None:
        """Thread-safe mailbox; register writes happen exclusively in run()."""
        self._commands.put((RobotCommandMode.POSE_ID, pose, sequence))

    def enqueue_angle_command(self, angle_tenths: int, sequence: int) -> None:
        self._commands.put((RobotCommandMode.ANGLE, angle_tenths, sequence))

    def stop(self) -> None:
        self._stop.set()

    @pyqtSlot()
    def run(self) -> None:
        rtde: Any = None
        rtde_io: Any = None
        dashboard: DashboardReadClient | None = None
        last_rtde_attempt = 0.0
        last_io_attempt = 0.0
        last_dashboard_attempt = 0.0
        last_dashboard_poll = 0.0
        last_reported_error = ""
        dashboard_values: dict[str, str] = {}
        was_rtde_connected = False
        was_dashboard_connected = False
        self.state_changed.emit(ConnectionState.CONNECTING)
        self.event_message.emit("Starte RTDE-Monitor, Pose-Register und Dashboard-Verbindung …")

        try:
            from rtde_io import RTDEIOInterface
            from rtde_receive import RTDEReceiveInterface

            while not self._stop.is_set():
                now = time.monotonic()
                rtde_error = ""
                io_error = ""
                dashboard_error = ""

                if rtde is None and now - last_rtde_attempt >= 5.0:
                    last_rtde_attempt = now
                    try:
                        rtde = _create_rtde_receive(
                            RTDEReceiveInterface,
                            self._config.robot_ip,
                        )
                        if not rtde.isConnected():
                            raise ConnectionError("RTDE-Handshake nicht erfolgreich.")
                        if not was_rtde_connected:
                            self.event_message.emit("RTDE-Monitor verbunden (nur Lesen).")
                        was_rtde_connected = True
                    except Exception as exc:
                        rtde_error = f"RTDE: {exc}"
                        rtde = None

                if rtde_io is None and now - last_io_attempt >= 5.0:
                    last_io_attempt = now
                    try:
                        rtde_io = RTDEIOInterface(
                            self._config.robot_ip,
                            False,
                            True,
                        )
                        if not rtde_io.isConnected():
                            raise ConnectionError("RTDE-I/O-Handshake nicht erfolgreich.")
                        self.event_message.emit(
                            "Pose-Auswahlkanal verbunden (nur RTDE-Register 42/43)."
                        )
                    except Exception as exc:
                        io_error = f"Pose-Auswahlkanal: {exc}"
                        rtde_io = None

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
                            ("get loaded program", "loaded_program"),
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
                    command_channel_connected=rtde_io is not None,
                    robot_mode=dashboard_values.get("robot_mode", "–"),
                    safety_mode=dashboard_values.get("safety_mode", "–"),
                    remote_control=dashboard_values.get("remote_control", "–"),
                    program_state=dashboard_values.get("program_state", "–"),
                    loaded_program=dashboard_values.get("loaded_program", "–"),
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
                        status.acknowledged_sequence = int(
                            rtde.getOutputIntRegister(ACK_SEQUENCE_OUTPUT_REGISTER)
                        )
                        status.command_state_code = int(
                            rtde.getOutputIntRegister(COMMAND_STATE_OUTPUT_REGISTER)
                        )
                        status.command_state = COMMAND_STATES.get(
                            status.command_state_code,
                            f"Unbekannt ({status.command_state_code})",
                        )
                        acknowledged_value = int(
                            rtde.getOutputIntRegister(CURRENT_POSE_OUTPUT_REGISTER)
                        )
                        status.command_mode = self._requested_mode
                        status.acknowledged_raw_value = acknowledged_value
                        if self._requested_mode is RobotCommandMode.ANGLE:
                            status.acknowledged_angle_deg = (
                                acknowledged_value / 10.0
                                if ANGLE_MIN_TENTHS <= acknowledged_value <= ANGLE_MAX_TENTHS
                                else None
                            )
                        else:
                            status.acknowledged_pose = (
                                acknowledged_value if acknowledged_value in ALLOWED_POSES else None
                            )
                    except Exception as exc:
                        rtde_error = f"RTDE: {exc}"
                        try:
                            rtde.disconnect()
                        except Exception:
                            pass
                        rtde = None
                        status.rtde_connected = False

                if rtde_io is not None:
                    active_sequence: int | None = None
                    try:
                        while True:
                            mode, value, sequence = self._commands.get_nowait()
                            active_sequence = sequence
                            if not rtde_io.setInputIntRegister(POSE_INPUT_REGISTER, value):
                                raise ConnectionError("Pose-Register wurde nicht bestätigt.")
                            if not rtde_io.setInputIntRegister(
                                SEQUENCE_INPUT_REGISTER, sequence
                            ):
                                raise ConnectionError("Befehlsregister wurde nicht bestätigt.")
                            self._requested_mode = mode
                            self._requested_value = value
                            self._requested_sequence = sequence
                            self.event_message.emit(
                                f"UR-Ziel {value} als Auswahl #{sequence} "
                                "an das UR-Programm übergeben."
                            )
                    except queue.Empty:
                        pass
                    except Exception as exc:
                        io_error = f"Pose-Auswahlkanal: {exc}"
                        if active_sequence is not None:
                            self.command_failed.emit(active_sequence, str(exc))
                        try:
                            rtde_io.disconnect()
                        except Exception:
                            pass
                        rtde_io = None

                status.command_mode = self._requested_mode
                status.requested_raw_value = self._requested_value
                if self._requested_mode is RobotCommandMode.ANGLE:
                    status.requested_angle_deg = (
                        None if self._requested_value is None else self._requested_value / 10.0
                    )
                else:
                    status.requested_pose = self._requested_value
                status.requested_sequence = self._requested_sequence
                status.command_pending = (
                    self._requested_sequence is not None
                    and status.acknowledged_sequence != self._requested_sequence
                )

                if (
                    status.rtde_connected
                    and status.dashboard_connected
                    and status.command_channel_connected
                ):
                    state = ConnectionState.CONNECTED
                elif status.rtde_connected or status.dashboard_connected:
                    state = ConnectionState.DEGRADED
                else:
                    state = ConnectionState.ERROR
                self.state_changed.emit(state)
                self.status_changed.emit(status)

                combined_error = "; ".join(
                    filter(None, (rtde_error, io_error, dashboard_error))
                )
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
            if rtde_io is not None:
                try:
                    rtde_io.disconnect()
                except Exception:
                    pass
            if dashboard is not None:
                dashboard.close()
            self.state_changed.emit(ConnectionState.DISCONNECTED)
            self.event_message.emit("UR-Verbindungen und Pose-Auswahlkanal getrennt.")
            self.finished.emit()


class RobotAdapter(DeviceAdapter):
    def __init__(self, config: AppSettings, parent: QObject | None = None) -> None:
        super().__init__("UR16e", parent)
        self.config = config
        self._thread: QThread | None = None
        self._worker: RobotWorker | None = None
        self._pending_sequence: int | None = None
        self._next_sequence = int(time.time() * 1000) & 0x7FFFFFFF

    def connect(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread(self)
        self._worker = RobotWorker(self.config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.state_changed.connect(self._set_state)
        self._worker.status_changed.connect(self._forward_status)
        self._worker.command_failed.connect(self._command_failed)
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

    def request_pose(self, pose: int) -> bool:
        return self._request_pose_legacy(pose)

    def request_angle(self, angle_deg: float) -> bool:
        angle_tenths = round(float(angle_deg) * 10.0)
        if abs(float(angle_deg) * 10.0 - angle_tenths) > 1e-6:
            self._emit_error("UR-Winkel sind nur in Schritten von 0,1 Grad zulässig.")
            return False
        if not ANGLE_MIN_TENTHS <= angle_tenths <= ANGLE_MAX_TENTHS:
            self._emit_error("Der UR-Winkel muss zwischen 15,5 und 21,0 Grad liegen.")
            return False
        if self._worker is None or self._thread is None or not self._thread.isRunning():
            self._emit_error(
                "Ein UR-Winkel kann nur bei bestehender Verbindung angefordert werden."
            )
            return False
        if self._pending_sequence is not None:
            self._emit_error("Das vorherige UR-Ziel wurde noch nicht bestätigt.")
            return False
        self._next_sequence = (self._next_sequence + 1) & 0x7FFFFFFF
        if self._next_sequence == 0:
            self._next_sequence = 1
        self._pending_sequence = self._next_sequence
        self._worker.enqueue_angle_command(angle_tenths, self._next_sequence)
        self._emit_event(
            f"Winkel {angle_tenths / 10.0:.1f} Grad angefordert "
            f"(Auswahl #{self._next_sequence})."
        )
        return True

    def _request_pose_legacy(self, pose: int) -> bool:
        if pose not in ALLOWED_POSES:
            self._emit_error(f"Pose {pose} ist nicht freigegeben.")
            return False
        if self._worker is None or self._thread is None or not self._thread.isRunning():
            self._emit_error("Pose kann nur bei bestehender UR-Verbindung angefordert werden.")
            return False
        if self._pending_sequence is not None:
            self._emit_error("Die vorherige Pose wurde noch nicht vom UR-Programm bestätigt.")
            return False
        self._next_sequence = (self._next_sequence + 1) & 0x7FFFFFFF
        if self._next_sequence == 0:
            self._next_sequence = 1
        self._pending_sequence = self._next_sequence
        self._worker.enqueue_pose_command(pose, self._next_sequence)
        self._emit_event(f"Pose {pose} angefordert (Auswahl #{self._next_sequence}).")
        return True

    @pyqtSlot(object)
    def _forward_status(self, status: object) -> None:
        if isinstance(status, RobotStatus) and (
            self._pending_sequence is not None
            and status.acknowledged_sequence == self._pending_sequence
        ):
            self._pending_sequence = None
            status.command_pending = False
        super()._forward_status(status)

    @pyqtSlot(int, str)
    def _command_failed(self, sequence: int, message: str) -> None:
        if self._pending_sequence == sequence:
            self._pending_sequence = None
        self._emit_error(f"Pose-Auswahl #{sequence} fehlgeschlagen: {message}")

    def disconnect(self) -> None:
        worker, thread = self._worker, self._thread
        if worker is not None:
            worker.stop()
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(3000):
                self._emit_error("UR-Worker wurde nicht innerhalb von 3 Sekunden beendet.")
        self._pending_sequence = None
        self._set_state(ConnectionState.DISCONNECTED)
