from __future__ import annotations

import queue
import socket
import threading
import time
from dataclasses import replace
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from automated_image_capture.hardware.base import DeviceAdapter
from automated_image_capture.models import ConnectionState, ConveyorMove, ConveyorStatus
from automated_image_capture.settings import AppSettings

PLC_PORT_TC3 = 851
ADS_TIMEOUT_MS = 500
ADS_RECONNECT_SECONDS = 2.0
IDLE_POLL_SECONDS = 0.5
ACTIVE_POLL_SECONDS = 0.1
INCREMENTS_PER_FULL_STEP = 64
UINT32_MODULUS = 1 << 32

STATUS_SYMBOLS = {
    "MAIN.CalibrationBusy": "busy",
    "MAIN.CalibrationError": "error",
    "MAIN.CalibrationStatusCode": "status_code",
    "MAIN.StepperPosReadyToExecute": "ready_to_execute",
    "MAIN.StepperInternalPosition": "internal_position",
    "MAIN.GuiConveyorMmPerFullStep": "mm_per_full_step",
    "MAIN.GuiConveyorCalibrationValid": "calibration_valid",
    "MAIN.GuiConveyorReverse": "conveyor_reverse",
}


def signed_u32_delta(value: int, origin: int) -> int:
    """Return a wrap-safe signed delta between two PLC UDINT positions."""
    delta = (int(value) - int(origin)) % UINT32_MODULUS
    return delta - UINT32_MODULUS if delta >= UINT32_MODULUS // 2 else delta


def full_steps_for_offset(offset_mm: float, mm_per_full_step: float) -> int:
    if mm_per_full_step <= 0.0:
        raise ValueError("Die Förderbandkalibrierung ist ungültig.")
    if offset_mm < 0.0:
        raise ValueError("Der Förderbandoffset darf nicht negativ sein.")
    return int(float(offset_mm) / float(mm_per_full_step) + 0.5)


def speed_in_full_steps(speed_mm_per_s: float, mm_per_full_step: float) -> float:
    if speed_mm_per_s <= 0.0:
        raise ValueError("Die Förderbandgeschwindigkeit muss positiv sein.")
    if mm_per_full_step <= 0.0:
        raise ValueError("Die Förderbandkalibrierung ist ungültig.")
    return max(1.0, min(500.0, float(speed_mm_per_s) / float(mm_per_full_step)))


def effective_direction_sign(plc_direction: str, conveyor_reverse: bool) -> int:
    if plc_direction not in {"left", "right"}:
        raise ValueError("Die SPS-Richtung muss 'left' oder 'right' sein.")
    negative = plc_direction == "left"
    if conveyor_reverse:
        negative = not negative
    return -1 if negative else 1


def diagnose_plc_network(host: str, port: int = 48898, timeout: float = 0.4) -> str:
    """Give a useful hint without modifying routes or network settings."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return ""
    except OSError as exc:
        return f"SPS-IP {host} ist auf ADS/TCP {port} nicht erreichbar: {exc}"


class ConveyorWorker(QObject):
    state_changed = pyqtSignal(object)
    status_changed = pyqtSignal(object)
    error = pyqtSignal(str)
    event_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: AppSettings) -> None:
        super().__init__()
        self._config = config
        self._stop = threading.Event()
        self._commands: queue.Queue[tuple[str, object | None]] = queue.Queue()
        self._requested_move: ConveyorMove | None = None
        self._completed_sequence: int | None = None
        self._completed_internal_position: int | None = None
        self._completed_at: float | None = None
        self._saw_busy = False
        self._move_not_before = 0.0

    def enqueue_move(self, move: ConveyorMove) -> None:
        self._commands.put(("move", move))

    def enqueue_stop(self) -> None:
        self._commands.put(("stop", None))

    def enqueue_release(self) -> None:
        self._commands.put(("release", None))

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _safe_values() -> dict[str, object]:
        return {
            "MAIN.GuiCalibrationStop": True,
            "MAIN.GuiConveyorCalibrationMode": False,
            "MAIN.GuiConveyorEnabled": False,
        }

    def _write_values(self, plc: Any, values: dict[str, object], pyads: Any) -> None:
        del pyads
        errors = plc.write_list_by_name(values, cache_symbol_info=True)
        failed = {
            name: error
            for name, error in errors.items()
            if error and str(error).lower() != "no error"
        }
        if failed:
            raise RuntimeError(f"ADS-Sammelschreibzugriff fehlgeschlagen: {failed}")

    def _handle_command(self, plc: Any, command: str, payload: object | None, pyads: Any) -> None:
        if command == "stop":
            self._write_values(plc, {"MAIN.GuiCalibrationStop": True}, pyads)
            self.event_message.emit("Förderband-Stop an die SPS gesendet.")
            return
        if command == "release":
            self._write_values(plc, self._safe_values(), pyads)
            self._requested_move = None
            self._saw_busy = False
            self.event_message.emit("Förderbandsteuerung freigegeben und gestoppt.")
            return
        if command != "move" or not isinstance(payload, ConveyorMove):
            return
        if self._requested_move is not None:
            raise RuntimeError("Eine Förderbandfahrt ist bereits aktiv.")
        direction_symbol = (
            "MAIN.GuiCalibrationMoveLeft"
            if payload.plc_direction == "left"
            else "MAIN.GuiCalibrationMoveRight"
        )
        self._write_values(
            plc,
            {
                "MAIN.GuiConveyorEnabled": False,
                "MAIN.GuiConveyorCalibrationMode": True,
                "MAIN.GuiCalibrationJogSteps": abs(payload.delta_full_steps),
                "MAIN.GuiCalibrationJogSpeedFullStepsPerSec": payload.speed_full_steps_per_s,
                direction_symbol: True,
            },
            pyads,
        )
        self._requested_move = payload
        self._saw_busy = False
        self._move_not_before = time.monotonic() + 0.05
        self.event_message.emit(
            f"Fahrt #{payload.sequence}: {payload.requested_offset_mm:g} mm "
            f"({abs(payload.delta_full_steps)} Vollschritte, {payload.plc_direction})."
        )

    @pyqtSlot()
    def run(self) -> None:
        plc: Any = None
        pyads: Any = None
        last_attempt = 0.0
        last_error = ""
        self.state_changed.emit(ConnectionState.CONNECTING)
        try:
            import pyads as imported_pyads

            pyads = imported_pyads
            pyads.set_timeout(ADS_TIMEOUT_MS)
            while not self._stop.is_set():
                now = time.monotonic()
                if plc is None and now - last_attempt >= ADS_RECONNECT_SECONDS:
                    last_attempt = now
                    try:
                        plc = pyads.Connection(
                            self._config.plc_ams_net_id,
                            self._config.plc_port,
                            self._config.plc_ip,
                        )
                        plc.open()
                        plc.read_state()
                        self.state_changed.emit(ConnectionState.CONNECTED)
                        self.event_message.emit(
                            f"TwinCAT ADS verbunden ({self._config.plc_ip}, "
                            f"{self._config.plc_ams_net_id}:{self._config.plc_port})."
                        )
                        last_error = ""
                    except Exception as exc:
                        if plc is not None:
                            try:
                                plc.close()
                            except Exception:
                                pass
                        plc = None
                        hint = diagnose_plc_network(self._config.plc_ip)
                        message = hint or f"ADS-Verbindung/Route fehlgeschlagen: {exc}"
                        if message != last_error:
                            self.error.emit(message)
                            last_error = message
                        self.state_changed.emit(ConnectionState.ERROR)

                if plc is None:
                    self._stop.wait(0.1)
                    continue

                try:
                    while True:
                        command, payload = self._commands.get_nowait()
                        self._handle_command(plc, command, payload, pyads)
                except queue.Empty:
                    pass

                try:
                    values = plc.read_list_by_name(list(STATUS_SYMBOLS), cache_symbol_info=True)
                    status = ConveyorStatus(
                        connected=True,
                        calibration_valid=bool(values["MAIN.GuiConveyorCalibrationValid"]),
                        mm_per_full_step=float(values["MAIN.GuiConveyorMmPerFullStep"]),
                        ready_to_execute=bool(values["MAIN.StepperPosReadyToExecute"]),
                        busy=bool(values["MAIN.CalibrationBusy"]),
                        error=bool(values["MAIN.CalibrationError"]),
                        status_code=int(values["MAIN.CalibrationStatusCode"]),
                        internal_position=int(values["MAIN.StepperInternalPosition"]),
                        conveyor_reverse=bool(values["MAIN.GuiConveyorReverse"]),
                        requested_offset_mm=(
                            None
                            if self._requested_move is None
                            else self._requested_move.requested_offset_mm
                        ),
                        actual_target_offset_mm=(
                            None
                            if self._requested_move is None
                            else self._requested_move.actual_offset_mm
                        ),
                        requested_sequence=(
                            None if self._requested_move is None else self._requested_move.sequence
                        ),
                        completed_sequence=self._completed_sequence,
                        completed_internal_position=self._completed_internal_position,
                        completed_at=self._completed_at,
                        last_move=self._requested_move,
                    )
                    if status.busy or status.status_code in {1, 2}:
                        self._saw_busy = True
                    if self._requested_move is not None:
                        if status.error or status.status_code in {4, 5}:
                            failed = self._requested_move
                            self._requested_move = None
                            self._saw_busy = False
                            self.error.emit(
                                f"Förderbandfahrt #{failed.sequence} von der SPS abgelehnt "
                                f"(Status {status.status_code})."
                            )
                        elif (
                            status.status_code == 3
                            and not status.busy
                            and time.monotonic() >= self._move_not_before
                        ):
                            self._completed_sequence = self._requested_move.sequence
                            self._completed_internal_position = status.internal_position
                            self._completed_at = time.time()
                            status.completed_sequence = self._completed_sequence
                            status.completed_internal_position = self._completed_internal_position
                            status.completed_at = self._completed_at
                            status.last_move = self._requested_move
                            self.event_message.emit(
                                f"Förderbandfahrt #{self._completed_sequence} abgeschlossen."
                            )
                            self._requested_move = None
                            self._saw_busy = False
                    self.status_changed.emit(status)
                    self.state_changed.emit(ConnectionState.CONNECTED)
                    last_error = ""
                except Exception as exc:
                    message = f"ADS-Lese-/Schreibfehler oder fehlendes SPS-Symbol: {exc}"
                    if message != last_error:
                        self.error.emit(message)
                        last_error = message
                    try:
                        plc.close()
                    except Exception:
                        pass
                    plc = None
                    self.state_changed.emit(ConnectionState.ERROR)
                self._stop.wait(
                    ACTIVE_POLL_SECONDS if self._requested_move is not None else IDLE_POLL_SECONDS
                )
        except Exception as exc:
            self.state_changed.emit(ConnectionState.ERROR)
            self.error.emit(f"pyads konnte nicht geladen oder gestartet werden: {exc}")
        finally:
            if plc is not None and pyads is not None:
                try:
                    self._write_values(plc, self._safe_values(), pyads)
                except Exception:
                    pass
                try:
                    plc.close()
                except Exception:
                    pass
            self.state_changed.emit(ConnectionState.DISCONNECTED)
            self.event_message.emit("TwinCAT-ADS-Verbindung getrennt.")
            self.finished.emit()


class ConveyorAdapter(DeviceAdapter):
    def __init__(self, config: AppSettings, parent: QObject | None = None) -> None:
        super().__init__("Förderband", parent)
        self.config = config
        self._thread: QThread | None = None
        self._worker: ConveyorWorker | None = None
        self._status = ConveyorStatus(forward_direction=config.conveyor_forward_direction)
        self._origin_position: int | None = None
        self._next_sequence = int(time.time() * 1000) & 0x7FFFFFFF

    @property
    def status(self) -> ConveyorStatus:
        return replace(self._status)

    @property
    def origin_position(self) -> int | None:
        return self._origin_position

    def connect(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread(self)
        self._worker = ConveyorWorker(self.config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.state_changed.connect(self._set_state)
        self._worker.status_changed.connect(self._on_status)
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

    def set_forward_direction(self, direction: str) -> None:
        if direction not in {"left", "right"}:
            raise ValueError("Bitte zuerst Links oder Rechts als Vorwärtsrichtung festlegen.")
        self._status.forward_direction = direction
        self.config.conveyor_forward_direction = direction
        self._decorate_status()
        self.status_changed.emit(replace(self._status))

    def set_current_as_origin(self) -> bool:
        if self._status.internal_position is None or self._status.busy:
            self._emit_error("Der Förderband-Nullpunkt kann aktuell nicht übernommen werden.")
            return False
        self._origin_position = self._status.internal_position
        self._status.origin_position = self._origin_position
        self._status.logical_offset_mm = 0.0
        self.status_changed.emit(replace(self._status))
        self._emit_event(f"Aktuelle SPS-Position {self._origin_position} als 0 mm übernommen.")
        return True

    def restore_origin(self, internal_position: int) -> None:
        self._origin_position = int(internal_position) % UINT32_MODULUS
        self._decorate_status()

    def _decorate_status(self) -> None:
        self._status.forward_direction = self.config.conveyor_forward_direction
        self._status.origin_position = self._origin_position
        if self._origin_position is None or self._status.internal_position is None:
            self._status.logical_offset_mm = None
            return
        direction = self._status.forward_direction
        if direction not in {"left", "right"}:
            self._status.logical_offset_mm = None
            return
        sign = effective_direction_sign(direction, self._status.conveyor_reverse)
        increments = signed_u32_delta(self._status.internal_position, self._origin_position)
        self._status.logical_offset_mm = (
            increments * sign / INCREMENTS_PER_FULL_STEP * self._status.mm_per_full_step
        )

    @pyqtSlot(object)
    def _on_status(self, status: object) -> None:
        if not isinstance(status, ConveyorStatus):
            return
        previous_move = self._status.last_move
        self._status = replace(status)
        if self._status.last_move is None:
            self._status.last_move = previous_move
        if self._status.last_move is not None:
            if self._status.requested_offset_mm is None:
                self._status.requested_offset_mm = self._status.last_move.requested_offset_mm
            if self._status.actual_target_offset_mm is None:
                self._status.actual_target_offset_mm = self._status.last_move.actual_offset_mm
        if (
            self._status.requested_sequence is not None
            and self._status.completed_sequence == self._status.requested_sequence
        ):
            self._status.requested_sequence = None
        self._decorate_status()
        self.status_changed.emit(replace(self._status))

    def _ready_worker(self) -> ConveyorWorker | None:
        if (
            self._worker is None
            or self._thread is None
            or not self._thread.isRunning()
            or self.state is not ConnectionState.CONNECTED
        ):
            self._emit_error("Das Förderband ist nicht über ADS verbunden.")
            return None
        return self._worker

    def request_offset(self, offset_mm: float, speed_mm_per_s: float) -> int | None:
        worker = self._ready_worker()
        if worker is None:
            return None
        if self._origin_position is None:
            self._emit_error("Bitte zuerst die aktuelle Förderbandposition als 0 mm übernehmen.")
            return None
        if not self._status.calibration_valid or self._status.mm_per_full_step <= 0.0:
            self._emit_error("Die Förderbandkalibrierung in der SPS ist ungültig.")
            return None
        if self._status.busy or self._status.requested_sequence is not None:
            self._emit_error("Eine Förderbandfahrt ist bereits aktiv.")
            return None
        forward = self.config.conveyor_forward_direction
        if forward not in {"left", "right"}:
            self._emit_error("Bitte zuerst die Vorwärtsrichtung des Förderbands bestätigen.")
            return None
        target_steps = full_steps_for_offset(offset_mm, self._status.mm_per_full_step)
        sign = effective_direction_sign(forward, self._status.conveyor_reverse)
        current_position = self._status.internal_position
        if current_position is None:
            current_position = self._origin_position
        current_increments = signed_u32_delta(current_position, self._origin_position)
        current_steps = round(current_increments * sign / INCREMENTS_PER_FULL_STEP)
        delta_steps = target_steps - current_steps
        if delta_steps == 0:
            move = ConveyorMove(
                sequence=0,
                requested_offset_mm=float(offset_mm),
                actual_offset_mm=target_steps * self._status.mm_per_full_step,
                target_full_steps=target_steps,
                delta_full_steps=0,
                speed_mm_per_s=float(speed_mm_per_s),
                speed_full_steps_per_s=speed_in_full_steps(
                    speed_mm_per_s, self._status.mm_per_full_step
                ),
                plc_direction=forward,
                start_internal_position=current_position,
                commanded_at=time.time(),
            )
            self._status.requested_offset_mm = float(offset_mm)
            self._status.actual_target_offset_mm = target_steps * self._status.mm_per_full_step
            self._status.logical_offset_mm = self._status.actual_target_offset_mm
            self._status.last_move = move
            self.status_changed.emit(replace(self._status))
            return 0
        plc_direction = forward if delta_steps > 0 else ("left" if forward == "right" else "right")
        self._next_sequence = (self._next_sequence + 1) & 0x7FFFFFFF or 1
        move = ConveyorMove(
            sequence=self._next_sequence,
            requested_offset_mm=float(offset_mm),
            actual_offset_mm=target_steps * self._status.mm_per_full_step,
            target_full_steps=target_steps,
            delta_full_steps=delta_steps,
            speed_mm_per_s=float(speed_mm_per_s),
            speed_full_steps_per_s=speed_in_full_steps(
                speed_mm_per_s, self._status.mm_per_full_step
            ),
            plc_direction=plc_direction,
            start_internal_position=self._status.internal_position,
            commanded_at=time.time(),
        )
        self._status.requested_sequence = move.sequence
        self._status.requested_offset_mm = move.requested_offset_mm
        self._status.actual_target_offset_mm = move.actual_offset_mm
        self._status.last_move = move
        self.status_changed.emit(replace(self._status))
        worker.enqueue_move(move)
        return move.sequence

    def jog(
        self,
        direction: str,
        distance_mm: float = 1.0,
        speed_mm_per_s: float = 10.0,
    ) -> int | None:
        if direction not in {"left", "right"}:
            raise ValueError("Ungültige Förderbandrichtung.")
        worker = self._ready_worker()
        if worker is None:
            return None
        if not self._status.calibration_valid or self._status.mm_per_full_step <= 0.0:
            self._emit_error("Die Förderbandkalibrierung in der SPS ist ungültig.")
            return None
        if self._status.busy or self._status.requested_sequence is not None:
            self._emit_error("Eine Förderbandfahrt ist bereits aktiv.")
            return None
        steps = max(1, full_steps_for_offset(distance_mm, self._status.mm_per_full_step))
        self._next_sequence = (self._next_sequence + 1) & 0x7FFFFFFF or 1
        move = ConveyorMove(
            sequence=self._next_sequence,
            requested_offset_mm=float(distance_mm),
            actual_offset_mm=steps * self._status.mm_per_full_step,
            target_full_steps=steps,
            delta_full_steps=steps,
            speed_mm_per_s=float(speed_mm_per_s),
            speed_full_steps_per_s=speed_in_full_steps(
                speed_mm_per_s, self._status.mm_per_full_step
            ),
            plc_direction=direction,
            start_internal_position=self._status.internal_position,
            commanded_at=time.time(),
        )
        self._status.requested_sequence = move.sequence
        self._status.last_move = move
        self.status_changed.emit(replace(self._status))
        worker.enqueue_move(move)
        return move.sequence

    def stop_motion(self) -> None:
        if self._worker is not None:
            self._worker.enqueue_stop()

    def release_control(self) -> None:
        if self._worker is not None:
            self._worker.enqueue_release()

    def position_matches(self, offset_mm: float, tolerance_full_steps: int = 1) -> bool:
        if self._status.logical_offset_mm is None or self._status.mm_per_full_step <= 0.0:
            return False
        expected = full_steps_for_offset(offset_mm, self._status.mm_per_full_step)
        actual = round(self._status.logical_offset_mm / self._status.mm_per_full_step)
        return abs(actual - expected) <= tolerance_full_steps

    def disconnect(self) -> None:
        worker, thread = self._worker, self._thread
        if worker is not None:
            worker.enqueue_release()
            worker.stop()
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(2000):
                self._emit_error("Förderband-Worker wurde nicht innerhalb von 2 Sekunden beendet.")
        self._set_state(ConnectionState.DISCONNECTED)
